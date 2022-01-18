
# Sepcifying a action-conditional RNN Cell
using Flux
using Tullio
import TensorToolbox: cp_als

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
    return new_h, reshape(new_h, :, sz[2:end]...)
end

Flux.@functor AARNNCell

function Base.show(io::IO, l::AARNNCell)
  print(io, "AARNNCell(", size(l.Wi, 2), ", ", size(l.Wa), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    AARNN(in::Integer, actions::Integer, out::Integer, σ = tanh)
Like an RNN cell, except takes a tuple (action, observation) as input. The action is used with
[`get_waa`](@ref) with results added to the usual update.

The update is as follows:
    `σ.(Wi*o .+ get_waa(Wa, a) .+ Wh*h .+ b)`
"""
AARNN(a...; ka...) = Flux.Recur(AARNNCell(a...; ka...))
Flux.Recur(m::AARNNCell) = Flux.Recur(m, m.state0)


struct MARNNCell{F, A, V, H} <: AbstractActionRNN
    σ::F
    Wx::A
    Wh::A
    b::V
    state0::H
end

MARNNCell(in, actions, out;
         init=glorot_uniform,
         initb=(args...;kwargs...) -> Flux.zeros(args...),
         init_state=Flux.zeros,
         σ_int=tanh) =
    MARNNCell(σ_int,
             init(actions, out, in; ignore_dims=1),
             init(actions, out, out; ignore_dims=1),
             initb(out, actions),
             init_state(out, 1))



function (m::MARNNCell)(h, x::Tuple{A, X}) where {A, X} # where {I<:Array{<:Integer, 1}, A<:AbstractArray{<:AbstractFloat, 2}}

    Wx, Wh, b, σ = m.Wx, m.Wh, m.b, m.σ

    a = x[1]
    o = x[2]

    wx = contract_WA(m.Wx, a, o)
    wh = contract_WA(m.Wh, a, h)
    ba = get_waa(m.b, a)

    new_h = σ.(wx .+ wh .+ ba)
    
    sz = size(o)
    if new_h isa AbstractVector
        new_h = reshape(new_h, :, 1)
    end
    return new_h, reshape(new_h, :, sz[2:end]...)

end

Flux.@functor MARNNCell

"""
    MARNN(in::Integer, actions::Integer, out::Integer, σ = tanh)
This cell incorporates the action as a multiplicative operation. We use 
[`contract_WA`](@ref) and [`get_waa`](@ref) to handle this.

The update is as follows:
```julia
new_h = σ.(contract_WA(m.Wx, a, o) .+ contract_WA(m.Wh, a, h) .+ get_waa(m.b, a))
```
"""
MARNN(args...; kwargs...) = Flux.Recur(MARNNCell(args...; kwargs...))
Flux.Recur(m::MARNNCell) = Flux.Recur(m, m.state0)

function Base.show(io::IO, l::MARNNCell)
  print(io, "MARNNCell(", size(l.Wx, 2), ", ", size(l.Wx, 3), ", ", size(l.Wx, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end


# Combo




mutable struct FacMARNNCell{F, A, V, H} <: AbstractActionRNN
    σ::F
    W::A
    Wx::A
    Wh::A
    Wa::A
    b::V
    state0::H
end

function FacMARNNCell(args...;
                      init_style="ignore",
                      kwargs...)

    init_cell_name = "FacMARNNCell_$(init_style)"
    rnn_init = getproperty(ActionRNNs, Symbol(init_cell_name))
    rnn_init(args...; kwargs...)
end

FacMARNNCell_standard(in, actions, out, factors, activation=tanh; hs_learnable=true, init=Flux.glorot_uniform, initb=Flux.zeros, init_state=Flux.zeros) = 
    FacMARNNCell(activation,
                 init(out, factors),
                 init(factors, in),
                 init(factors, out),
                 init(factors, actions),
                 initb(out, actions),
                 init_state(out, 1))

FacMARNNCell_ignore(in, actions, out, factors, activation=tanh; hs_learnable=true, init=Flux.glorot_uniform, initb=Flux.zeros, init_state=Flux.zeros) = 
    FacMARNNCell(activation,
                 init(out, factors; ignore_dims=2),
                 init(factors, in),
                 init(factors, out),
                 init(factors, actions),
                 initb(out, actions),
                 init_state(out, 1))

function FacMARNNCell_tensor(in, actions, out, factors, activation=tanh;
                            hs_learnable=true, init=glorot_uniform,
                            initb=Flux.zeros, init_state=Flux.zeros)

    W_t = init(actions, out, in+out; ignore_dims=1)
    W_d = cp_als(W_t, factors)

    W_a, W_o, W_hi = W_d.fmat
    W_o .*= W_d.lambda'
    
    FacMARNNCell(activation,
                 Float32.(W_o),
                 Float32.(transpose(W_hi[1:in, :])),
                 Float32.(transpose(W_hi[(in+1):end, :])),
                 Float32.(transpose(W_a)),
                 initb(out, actions),
                 init_state(out, 1))
end


"""
    FacMARNN(in::Integer, actions::Integer, out::Integer, factors, σ = tanh; init_style="ignore")
This cell incorporates the action as a multiplicative operation, but as a factored approximation of the multiplicative version.
This cell uses [`get_waa`](@ref). Uses [CP decomposition](https://en.wikipedia.org/wiki/Tensor_rank_decomposition).

The update is as follows:
```julia
   new_h = m.σ.(W*((Wx*o .+ Wh*h) .* get_waa(Wa, a)) .+ get_waa(m.b, a))
```

Three init_styles:
- standard: using init and initb w/o any keywords
- ignore: `W = init(out, factors, ignore_dims=2)`
- tensor: Decompose `W_t = init(actions, out, in+out; ignore_dims=1)` to get `W_o, W_a, W_hi` using `TensorToolbox.cp_als`.


"""
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
    W = m.W; Wx = m.Wx; Wh = m.Wh; Wa = m.Wa; a = x[1]; o = x[2]; b = get_waa(m.b, a)
    new_h = m.σ.(W*((Wx*o .+ Wh*h) .* get_waa(Wa, a)) .+ b)
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

function FacTucMARNNCell(args...;
                         init_style="standard",
                         kwargs...)

    init_cell_name = "FacTucMARNNCell_$(init_style)"
    rnn_init = getproperty(ActionRNNs, Symbol(init_cell_name))
    rnn_init(args...; kwargs...)
end

FacTucMARNNCell_standard(in, actions, out, action_factors, out_factors, in_factors,
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

FacTucMARNNCell_ignore(in, actions, out, action_factors, out_factors, in_factors,
                activation=tanh; hs_learnable=true, init=Flux.glorot_uniform,
                initb=Flux.zeros, init_state=Flux.zeros) =
    FacTucMARNNCell(activation,
                init(action_factors, out_factors, in_factors),
                init(action_factors, actions; ignore_dims=2),
                init(out, out_factors),
                init(in_factors, in),
                init(in_factors, out),
                initb(out, actions; ignore_dims=2),
                init_state(out, 1))

"""
    FacTucMARNN(in::Integer, actions::Integer, out::Integer, action_factors, out_factors, in_factors, σ = tanh; init_style="ignore")
This cell incorporates the action as a multiplicative operation, but as a factored approximation of the multiplicative version.
This cell uses [`get_waa`](@ref). Uses [Tucker decomposition](https://en.wikipedia.org/wiki/Tucker_decomposition).

Three init_styles:
- standard: using init and initb w/o any keywords
- ignore: `Wa = init(action_factors, actions; ignore_dims=2)`

"""
FacTucMARNN(args...; kwargs...) = Flux.Recur(FacTucMARNNCell(args...; kwargs...))
Flux.Recur(m::FacTucMARNNCell) = Flux.Recur(m, m.state0)
Flux.@functor FacTucMARNNCell

function (m::FacTucMARNNCell)(h, x::Tuple{A, X}) where {A, X}

    Wg, Wa, Wh, Wxx, Wxh, b, σ = m.Wg, m.Wa, m.Wh, m.Wxx, m.Wxh, m.b, m.σ

    a = x[1]
    o = x[2]

    waa = get_Wabya(Wa, a)

#     wx = contract_tuc(Wg, waa, Wh, Wxx*o)
#     wh = if size(h, 2) == 1
#         contract_tuc(Wg, waa, Wh, Wxh*h[:])
#     else
#         contract_tuc(Wg, waa, Wh, Wxh*h)
#     end

    wx, wh = if a isa Int
        Wh * (contract_Wga(Wg, waa) * (Wxx*o)), Wh * (contract_Wga(Wg, waa) * (Wxh*h[:]))
    else
        Wh * contract_Wgax(Wg, waa, Wxx*o), Wh * contract_Wgax(Wg, waa, Wxh*h)
    end
    ba = get_waa(b, a)

    new_h = σ.(wx .+ wh .+ ba)
    if a isa Int new_h = reshape(new_h , :, 1) end

    return new_h, new_h
end

