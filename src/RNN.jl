
# Sepcifying a action-conditional RNN Cell
using Flux
# using OMEinsum
using KernelAbstractions
using OMEinsum
using Tullio

"""
    RNN(in::Integer, out::Integer, σ = tanh)
The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.
"""
# RNN(a...; ka...) = Flux.Recur(AC_RNNCell(a...; ka...))
# Flux.Recur(m::AC_RNNCell) = Flux.Recur(m, m.state0)

# Flux.trainable(m::AC_RNNCell) = if hidden_learnable(m)
#     (Wx = m.Wi, Wh = m.Wh, b = m.b, state0 = m.state0)
# else
#     (Wx = m.Wi, Wh = m.Wh, b = m.b)
# end

abstract type AbstractActionRNN end
_needs_action_input(m::M) where {M<:AbstractActionRNN} = true

"""
    ARNNCell

    An RNN cell which explicitily transforms the hidden state of the recurrent neural network according to action.

"""
struct ARNNCell{F, A, V, H} <: AbstractActionRNN
    σ::F
    Wx::A
    Wh::A
    b::V
    state0::H
end

ARNNCell(num_ext_features, num_actions, num_hidden;
         init=Flux.glorot_uniform,
         initb=(args...;kwargs...) -> Flux.zeros(args...),
         init_state=Flux.zeros,
         σ_int=tanh,
         hs_learnable=true) =
    ARNNCell(σ_int,
             init(num_actions, num_hidden, num_ext_features; ignore_dims=1),
             init(num_actions, num_hidden, num_hidden; ignore_dims=1),
             initb(num_hidden, num_actions),
             init_state(num_hidden, 1))

# Flux.hidden(m::ARNNCell) = m.h
Flux.@functor ARNNCell
ARNN(args...; kwargs...) = Flux.Recur(ARNNCell(args...; kwargs...))
Flux.Recur(m::ARNNCell) = Flux.Recur(m, m.state0)


function (m::ARNNCell)(h, x::Tuple{I, A}) where {I<:Integer, A<:AbstractArray{<:AbstractFloat}}

    @inbounds new_h =
        m.σ.((@view m.Wx[x[1], :, :])*x[2] + (@view m.Wh[x[1], :, :])*h + m.b[:, x[1]])

    return new_h, new_h
end

function (m::ARNNCell)(h, x::Tuple{I, A}) where {I<:Array{<:Integer, 1}, A<:AbstractArray{<:AbstractFloat, 2}}


    # @show typeof(h), typeof(x)
    
    a = x[1]
    o = x[2]

    Wx = m.Wx[a, :, :]
    Wh = m.Wh[a, :, :]

    @ein wx[i, k] := Wx[k, i, j] * o[j, k]
    @ein wh[i, k] := Wh[k, i, j] * h[j, k]

    new_h = m.σ.(wx .+ wh .+ m.b[:, a])

    return new_h, new_h
end

function (m::ARNNCell)(h, x::Tuple{TA, A}) where {TA<:AbstractArray{<:AbstractFloat, 2}, A}
    a = x[1]
    o = x[2]

    @show size(h)
    out_h = _contract(m.Wh, h, a)

    new_h = m.σ.(out_h + m.b*a)
    new_h, new_h
end

function reduce_func(m::ARNNCell, x, h, i)
    @inbounds x[1][i]*view(m.Wx, :, :, i)*x[2] +
        x[1][i]*view(m.Wh, :, :, i)*h
end

function Base.show(io::IO, l::ARNNCell)
  print(io, "ARNNCell(", size(l.Wx, 2), ", ", size(l.Wx, 3), ", ", size(l.Wx, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end



@doc raw"""
    FactorizedARNNCell

    An RNN cell which explicitily transforms the hidden state of the recurrent neural network according to action.
    
    ```math
        W (x_{t+1}Wx \odot a_{t}Wa)^T
    ```

"""
mutable struct FacARNNCell{F, A, V, H} <: AbstractActionRNN
    σ::F
    W::A
    Wx::A
    Wh::A
    Wa::A
    b::V
    state0::H
end

# FacARNNCell(num_ext_features, num_actions, num_hidden; init=Flux.glorot_uniform, σ_int=tanh) =
FacARNNCell(in, actions, out, factors, activation=tanh; hs_learnable=true, init=Flux.glorot_uniform, initb=Flux.zeros, init_state=Flux.zeros) = 
    FacARNNCell(activation,
                init(out, factors),
                init(factors, in),
                init(factors, out),
                init(factors, actions),
                initb(out, actions),
                init_state(out, 1))

function FacARNNCell_tensor(in, actions, out, factors, activation=tanh;
                   init=Flux.glorot_uniform,
                   initb=Flux.zeros,
                   init_state=Flux.zeros,
                   hs_learnable=true) 


end

FacARNN(args...; kwargs...) = Flux.Recur(FacARNNCell(args...; kwargs...))
Flux.Recur(cell::FacARNNCell) = Flux.Recur(cell, cell.state0)

function get_Wabya(Wa, a)
    if a isa Int
        Wa[:, a]
    elseif eltype(a) <: Int
        Wa[:, a]
    else
        Wa*a
    end
end

function (m::FacARNNCell)(h, x::Tuple{A, O}) where {A, O}
    W = m.W; Wx = m.Wx; Wh = m.Wh; Wa = m.Wa; a = x[1]; o = x[2]; b = m.b
    new_h = m.σ.(W*((Wx*o .+ Wh*h) .* get_Wabya(Wa, a)) .+ b[:, a])
    return new_h, new_h
end

# Flux.hidden(m::FacARNNCell) = m.h
Flux.@functor FacARNNCell

# Flux.trainable(m::FacARNNCell) = if  hidden_learnable(m)
#     (W = m.W, Wx = m.Wx, Wh = m.Wh, Wa = m.Wa, b = m.b, h = m.h)
# else
#     (W = m.W, Wx = m.Wx, Wh = m.Wh, Wa = m.Wa, b = m.b)
# end
