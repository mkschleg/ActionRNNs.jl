
# Sepcifying a action-conditional RNN Cell
using Flux
using Tullio

struct AARNNCell{F,A,V,S} <: AbstractActionRNN
    σ::F
    Wi::A
    Wa::A
    Wh::A
    b::V
    state0::S
end

AARNNCell(in::Integer,
          na::Integer,
          out::Integer,
          σ=tanh;
          init=Flux.glorot_uniform,
          initb=Flux.zeros,
          init_state=Flux.zeros) = 
              AARNNCell(σ,
                        init(out, in),
                        init(out, na),
                        init(out, out),
                        initb(out),
                        init_state(out, 1))

function (m::AARNNCell)(h, x::Tuple{A, X}) where {A, X}
    σ, Wi, Wa, Wh, b = m.σ, m.Wi, m.Wa, m.Wh, m.b

    o = x[2]
    a = x[1]

    new_h = σ.(Wi*o .+ get_waa(Wa, a) .+ Wh*h .+ b)
    sz = size(o)
    return new_h, new_h#reshape(h, :, sz[2:end]...)
end

Flux.@functor AARNNCell

function Base.show(io::IO, l::AARNNCell)
  print(io, "AARNNCell(", size(l.Wi, 2), ", ", size(l.Wa), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    AARNN(in::Integer, out::Integer, σ = tanh)
The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.
"""
AARNN(a...; ka...) = Flux.Recur(AARNNCell(a...; ka...))
Flux.Recur(m::AARNNCell) = Flux.Recur(m, m.state0)



"""
    MARNNCell

    An RNN cell which explicitily transforms the hidden state of the recurrent neural network according to action using multiplicative updates.

"""
struct MARNNCell{F, A, V, H} <: AbstractActionRNN
    σ::F
    Wx::A
    Wh::A
    b::V
    state0::H
end

MARNNCell(num_ext_features, num_actions, num_hidden;
         init=glorot_uniform,
         initb=(args...;kwargs...) -> Flux.zeros(args...),
         init_state=Flux.zeros,
         σ_int=tanh) =
    MARNNCell(σ_int,
             init(num_actions, num_hidden, num_ext_features; ignore_dims=1),
             init(num_actions, num_hidden, num_hidden; ignore_dims=1),
             initb(num_hidden, num_actions),
             init_state(num_hidden, 1))

Flux.@functor MARNNCell
MARNN(args...; kwargs...) = Flux.Recur(MARNNCell(args...; kwargs...))
Flux.Recur(m::MARNNCell) = Flux.Recur(m, m.state0)


function (m::MARNNCell)(h, x::Tuple{A, X}) where {A, X} # where {I<:Array{<:Integer, 1}, A<:AbstractArray{<:AbstractFloat, 2}}
    Wx, Wh, b, σ = m.Wx, m.Wh, m.b, m.σ

    a = x[1]
    o = x[2]

    wx = contract_WA(m.Wx, a, o)
    wh = contract_WA(m.Wh, a, h)
    ba = get_waa(m.b, a)

    new_h = m.σ.(wx .+ wh .+ ba)

    return new_h, new_h
end

function Base.show(io::IO, l::MARNNCell)
  print(io, "MARNNCell(", size(l.Wx, 2), ", ", size(l.Wx, 3), ", ", size(l.Wx, 1))
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
mutable struct FacMARNNCell{F, A, V, H} <: AbstractActionRNN
    σ::F
    W::A
    Wx::A
    Wh::A
    Wa::A
    b::V
    state0::H
end

# FacMARNNCell(num_ext_features, num_actions, num_hidden; init=Flux.glorot_uniform, σ_int=tanh) =
FacMARNNCell(in, actions, out, factors, activation=tanh;
             hs_learnable=true, init=Flux.glorot_uniform,
             initb=Flux.zeros, init_state=Flux.zeros) =
    FacMARNNCell(activation,
                init(out, factors; ignore_dims=2),
                init(factors, in),
                init(factors, out),
                init(factors, actions),
                initb(out, actions),
                 init_state(out, 1))

function FacMARNNCell_init2(in, actions, out, factors, activation=tanh;
                            hs_learnable=true, init=glorot_uniform,
                            initb=Flux.zeros, init_state=Flux.zeros)
    W = init(actions, out, in)
    
    FacMARNNCell(activation,
                init(out, factors; ignore_dims=2),
                init(factors, in),
                init(factors, out),
                init(factors, actions),
                initb(out, actions),
                init_state(out, 1))
end

FacMARNN(args...; kwargs...) = Flux.Recur(FacMARNNCell(args...; kwargs...))
Flux.Recur(cell::FacMARNNCell) = Flux.Recur(cell, cell.state0)
Flux.@functor FacMARNNCell

function get_Wabya(Wa, a)
    if a isa Int
        Wa[:, a]
    elseif eltype(a) <: Int
        Wa[:, a]
    else
        Wa*a
    end
end

function (m::FacMARNNCell)(h, x::Tuple{A, O}) where {A, O}
    W = m.W; Wx = m.Wx; Wh = m.Wh; Wa = m.Wa; a = x[1]; o = x[2]; b = m.b
    new_h = m.σ.(W*((Wx*o .+ Wh*h) .* get_Wabya(Wa, a)) .+ b[:, a])
    return new_h, new_h
end

# Flux.hidden(m::FacMARNNCell) = m.h

mutable struct FacTucMARNNCell{F, T, A, V, H} <: AbstractActionRNN
    σ::F
    Wg::T
    Wa::A
    Wh::A
    Wxx::A
    Wxh::A
    b::V
    state0::H
end

FacTucMARNNCell(in, actions, out, action_factors, out_factors, in_factors,
                activation=tanh; hs_learnable=true, init=Flux.glorot_uniform,
                initb=Flux.zeros, init_state=Flux.zeros) =
    FacTucMARNNCell(activation,
                init(action_factors, out_factors, in_factors),
                init(action_factors, actions),
                init(out, out_factors),
                init(in_factors, in),
                init(in_factors, out),
                initb(out, actions),
                init_state(out, 1))

FacTucMARNN(args...; kwargs...) = Flux.Recur(FacTucMARNNCell(args...; kwargs...))
Flux.Recur(m::FacTucMARNNCell) = Flux.Recur(m, m.state0)
Flux.@functor FacTucMARNNCell

function (m::FacTucMARNNCell)(h, x::Tuple{A, X}) where {A, X} # where {I<:Array{<:Integer, 1},A<:AbstractArray{<:AbstractFloat, 2}}

    Wg, Wa, Wh, Wxx, Wxh, b, σ = m.Wg, m.Wa, m.Wh, m.Wxx, m.Wxh, m.b, m.σ

    a = x[1]
    o = x[2]

#     wx = contract_tuc(Wg, waa, Wh, Wxx*o)
#     wh = if size(h, 2) == 1
#         contract_tuc(Wg, waa, Wh, Wxh*h[:])
#     else
#         contract_tuc(Wg, waa, Wh, Wxh*h)
#     end

    waa = get_waa(Wa, a)
    wx = Wh * (contract_Wga(Wg, waa) * (Wxx*o))
    wh = if size(h, 2) == 1
        Wh * (contract_Wga(Wg, waa) * (Wxh*h[:]))
    else
        Wh * (contract_Wga(Wg, waa) * (Wxh*h))
    end
    ba = get_waa(b, a)

    new_h = if size(h, 2) == 1
        reshape(σ.(wx .+ wh .+ ba), :, 1)
    else
        σ.(wx .+ wh .+ ba)
    end

    return new_h, new_h
end

contract_Wga(Wg, Wa::AbstractVector{<:Number}) =
    @tullio ret[q, r] := Wg[p, q, r] * Wa[p]

contract_tuc(Wg, Wa::AbstractVector{<:Number}, Wh::AbstractMatrix{<:Number},Wx::AbstractVector{<:Number}) =
    @tullio ret[j] := Wg[p, q, r] * Wa[p] * Wh[q, j] * Wx[r]

contract_tuc(Wg, Wa::AbstractVector{<:Number}, Wh::AbstractMatrix{<:Number},Wx::AbstractMatrix{<:Number}) =
    @tullio ret[j, x] := Wg[p, q, r] * Wa[p] * Wh[q, j] * Wx[r, x]

# tucker_compose(Wg, Wa::AbstractMatrix{<:Number}, Wh, Wx) =
#     @tullio ret[i, j, k] := Wg[p, q, r] * Wa[p, i] * Wh[q, j] * Wx[r, k]
