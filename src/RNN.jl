
# Sepcifying a action-conditional RNN Cell
using Flux
# using OMEinsum
using Tullio


hidden_learnable(rnn) = true
hidden_learnable(rnn::Flux.Recur) = hidden_learnable(rnn.cell)

Flux.trainable(m::Flux.Recur) = if hidden_learnable(m)
    (cell = m.cell, init=m.init)
else
    (cell = m.cell,)
end

mutable struct RNNCell{F,A,V}
    σ::F
    Wi::A
    Wh::A
    b::V
    h::V
    h_learnable::Bool
end

hidden_learnable(rnn::RNNCell) = rnn.h_learnable

RNNCell(in::Integer, out::Integer, σ = tanh;
        init = glorot_uniform, hs_learnable=true) =
  RNNCell(σ, init(out, in), init(out, out),
          init(out), Flux.zeros(out), hs_learnable)

function (m::RNNCell)(h, x)
  σ, Wi, Wh, b = m.σ, m.Wi, m.Wh, m.b
  h = σ.(Wi*x .+ Wh*h .+ b)
  return h, h
end

Flux.hidden(m::RNNCell) = m.h
Flux.@functor RNNCell

function Base.show(io::IO, l::RNNCell)
  print(io, "RNNCell(", size(l.Wi, 2), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    RNN(in::Integer, out::Integer, σ = tanh)
The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.
"""
RNN(a...; ka...) = Flux.Recur(RNNCell(a...; ka...))

Flux.trainable(m::RNNCell) = if hidden_learnable(m)
    (Wx = m.Wi, Wh = m.Wh, b = m.b, h = m.h)
else
    (Wx = m.Wi, Wh = m.Wh, b = m.b)
end


abstract type AbstractActionRNN end

_needs_action_input(m::M) where {M<:AbstractActionRNN} = true


"""
    ARNNCell

    An RNN cell which explicitily transforms the hidden state of the recurrent neural network according to action.

"""
mutable struct ARNNCell{F, A, V, H} <: AbstractActionRNN
    σ::F
    Wx::A
    Wh::A
    b::V
    h::H
    islearnable::Bool
end

ARNNCell(num_ext_features, num_actions, num_hidden; init=Flux.glorot_uniform, σ_int=tanh, hs_learnable=true) =
    ARNNCell(σ_int,
             init(num_hidden, num_ext_features, num_actions),
             init(num_hidden, num_hidden, num_actions),
             zeros(Float32, num_hidden, num_actions),
             Flux.zeros(num_hidden),
             hs_learnable)

hidden_learnable(rnn::ARNNCell) = rnn.islearnable

Flux.hidden(m::ARNNCell) = m.h
Flux.@functor ARNNCell
ARNN(args...; kwargs...) = Flux.Recur(ARNNCell(args...; kwargs...))
Flux.trainable(m::ARNNCell) = if  hidden_learnable(m)
    (Wx = m.Wx, Wh = m.Wh, b = m.b, h = m.h)
else
    (Wx = m.Wx, Wh = m.Wh, b = m.b)
end

function (m::ARNNCell)(h, x::Tuple{I, A}) where {I<:Integer, A<:AbstractArray{<:AbstractFloat}}

    @inbounds new_h =
        m.σ.((@view m.Wx[:, :, x[1]])*x[2] + (@view m.Wh[:, :, x[1]])*h + (@view m.b[:, x[1]]))

    return new_h, new_h
end

function (m::ARNNCell)(h, x::Tuple{I, A}) where {I<:Array{<:Integer, 1}, A<:Array{<:AbstractFloat, 2}}
    # @info "Here"
    wx = ein_mul((@view m.Wx[:, :, x[1]]), x[2])
    wh = ein_mul((@view m.Wh[:, :, x[1]]), h)
    
    @inbounds new_h =
        m.σ.(wx + wh + (m.b[:, x[1]]))

    return new_h, new_h
end

function ein_mul(x::AbstractArray{<:Number, 3}, y::AbstractArray{<:Number, 2})
    @tullio ret[i, k] := x[i, j, k]*y[j,k]
end


function _contract(W::AbstractArray{<:Number, 3}, x1::AbstractArray{<:Number, 2}, x2::AbstractArray{<:Number, 2})
    sze_W = size(W)
    Wx2 = reshape(reshape(W, :, sze_W[end])*x2, sze_W[1:2]..., :)
    throw("Not Implemented")
    # @ein ret[i, l] := Wx2[i, j, l]*x1[j, l]
end

function _contract(W::AbstractArray{<:Number, 3}, x1::AbstractArray{<:Number, 1}, x2::AbstractArray{<:Number, 2})
    pdW = permutedims(W, (1, 3, 2))
    sze_W = size(pdW)
    Wx1 = reshape(reshape(pdW, :, sze_W[end])*x1, sze_W[1:2]...)
    Wx1*x2
end

function (m::ARNNCell)(h, x::Tuple{TA, A}) where {TA<:AbstractArray{<:AbstractFloat, 2}, A}
    a = x[1]
    o = x[2]

    @show size(h)
    out_h = _contract(m.Wh, h, a)

    new_h = m.σ.(out_h + m.b*a)
    new_h, new_h
end

function (m::ARNNCell)(h, x::Tuple{Array{<:Integer, 1}, A}) where {A}
    if length(size(h)) == 1
        @inbounds new_h = m.σ.(
            hcat((((@view m.Wx[:, :, x[1][i]])*x[2][:, i]) +
                  ((@view m.Wh[:, :, x[1][i]])*h) +
                  (@view m.b[:, x[1][i]]) for i in 1:length(x[1]))...))
        return new_h, new_h
    else
        @inbounds new_h = m.σ.(
            hcat((((@view m.Wx[:, :, x[1][i]])*x[2][:, i]) +
                  ((@view m.Wh[:, :, x[1][i]])*h[:, i]) +
                  (@view m.b[:, x[1][i]]) for i in 1:length(x[1]))...))

        return new_h, new_h
    end
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
    h::H
    hs_learnable::Bool
end

# FacARNNCell(num_ext_features, num_actions, num_hidden; init=Flux.glorot_uniform, σ_int=tanh) =
FacARNNCell(in, actions, out, factors, activation=tanh; hs_learnable=true, init=Flux.glorot_uniform) = 
    FacARNNCell(activation,
                init(out, factors),
                init(factors, in),
                init(factors, out),
                init(factors, actions),
                Flux.zeros(out, actions),
                Flux.zeros(out),
                hs_learnable)

FacARNN(args...; kwargs...) = Flux.Recur(FacARNNCell(args...; kwargs...))

hidden_learnable(m::FacARNNCell) = m.hs_learnable

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
    new_h = m.σ.(W*((Wx*o + Wh*h) .* get_Wabya(Wa, a)) + b[:, a])
    return new_h, new_h
end

Flux.hidden(m::FacARNNCell) = m.h
Flux.@functor FacARNNCell

Flux.trainable(m::FacARNNCell) = if  hidden_learnable(m)
    (W = m.W, Wx = m.Wx, Wh = m.Wh, Wa = m.Wa, b = m.b, h = m.h)
else
    (W = m.W, Wx = m.Wx, Wh = m.Wh, Wa = m.Wa, b = m.b)
end
