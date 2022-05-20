

function additive_rnn_inner(X::Tuple, h, σ, Wi, Wa, Wh, b)
    o = X[2]
    a = X[1]
    
    σ.(Wi*o .+ get_waa(Wa, a) .+ Wh*h .+ b)
end

additive_rnn_inner(X::Tuple, h, Wi, Wa, Wh, b) =
    additive_rnn_inner(X::Tuple, h, identity, Wi, Wa, Wh, b)



struct MixRNNCell{F,A,V,S} <: AbstractActionRNN
    σ::F
    θ::V
    Wi::Vector{A}
    Wa::Vector{A}
    Wh::Vector{A}
    b::Vector{V}
    state0::S
end

MixRNNCell(in::Int,
           na::Int,
           out::Int,
           num_experts::Int,
           σ=tanh;
           init=Flux.glorot_uniform,
           initb=Flux.zeros,
           init_state=Flux.zeros) = 
               MixRNNCell(σ,
                            # Flux.ones(2),
                            Flux.ones(num_experts),
                            [init(out, in) for _ in 1:num_experts],
                            [init(out, na) for _ in 1:num_experts],
                            [init(out, out) for _ in 1:num_experts],
                            [initb(out) for _ in 1:num_experts],
                            init_state(out, 1))




function (m::MixRNNCell)(h, x::Tuple{A, X}) where {A, X}
    σ, θ, Wi, Wa, Wh, b = m.σ, m.θ, m.Wi, m.Wa, m.Wh, m.b

    o = x[2]
    a = x[1]

    # additive
    new_hs = additive_rnn_inner.((x,), (h,), σ, Wi, Wa, Wh, b)
    # mix
    new_h = sum(θ .* new_hs) ./ sum(θ)

    sz = size(o)
    return new_h, reshape(new_h, :, sz[2:end]...)
end


Flux.@functor MixRNNCell

function Base.show(io::IO, l::MixRNNCell)
  print(io, "MixRNNCell()")
end

"""
    MixRNN(in, actions, out, num_experts, σ = tanh)

Mixing between `num_experts` [`AARNN`](@ref) cells. Uses the weighting

```julia
h′ = sum(θ[i] .* expert_h′[i] for i in 1:length(θ)) ./ sum(θ)
```

"""
MixRNN(a...; ka...) = Flux.Recur(MixRNNCell(a...; ka...))
Flux.Recur(m::MixRNNCell) = Flux.Recur(m, m.state0)


struct MixElRNNCell{F,A,V,S} <: AbstractActionRNN
    σ::F
    θ::Vector{V}
    Wi::Vector{A}
    Wa::Vector{A}
    Wh::Vector{A}
    b::Vector{V}
    state0::S
end

function MixElRNNCell(in::Int,
                      na::Int,
                      out::Int,
                      num_experts::Int,
                      σ=tanh;
                      init=Flux.glorot_uniform,
                      initb=Flux.zeros,
                      init_state=Flux.zeros)
    
    MixElRNNCell(σ,
                 # Flux.ones(2),
                 [Flux.ones(out) for _ in 1:num_experts],
                 [init(out, in) for _ in 1:num_experts],
                 [init(out, na) for _ in 1:num_experts],
                 [init(out, out) for _ in 1:num_experts],
                 [initb(out) for _ in 1:num_experts],
                 init_state(out, 1))
end


function (m::MixElRNNCell)(h, x::Tuple{A, X}) where {A, X}
    σ, θ, Wi, Wa, Wh, b = m.σ, m.θ, m.Wi, m.Wa, m.Wh, m.b

    o = x[2]
    a = x[1]

    # additive
    new_hs = additive_rnn_inner.((x,), (h,), σ, Wi, Wa, Wh, b)
    # mix
    new_h = sum(θ[i] .* new_hs[i] for i in 1:length(θ)) ./ sum(θ)

    sz = size(o)
    return new_h, reshape(new_h, :, sz[2:end]...)
end


Flux.@functor MixElRNNCell

function Base.show(io::IO, l::MixElRNNCell)
  print(io, "MixElRNNCell()")
end


"""
    MixElRNN(in, actions, out, num_experts, σ = tanh)

Mixing between `num_experts` [`AARNN`](@ref) cells. Uses the weighting

```julia
h′ = sum(θ[i] .* expert_h′[i] for i in 1:length(θ)) ./ sum(θ)
```

(here θ[i] is a vector).

"""
MixElRNN(a...; ka...) = Flux.Recur(MixElRNNCell(a...; ka...))
Flux.Recur(m::MixElRNNCell) = Flux.Recur(m, m.state0)
