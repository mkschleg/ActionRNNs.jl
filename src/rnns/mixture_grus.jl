
function additive_gru_inner(x, h, Wi, Wa, Wh, b)

    o = size(h, 1)

    a = x[1]
    obs = x[2]

    gx, gh = Wi*obs, Wh*h
    ga = get_waa(Wa, a)
    
    r = σ.(gate(gx, o, 1) .+ gate(ga, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
    z = σ.(gate(gx, o, 2) .+ gate(ga, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
    h̃ = tanh.(gate(gx, o, 3) .+ gate(ga, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
    (1 .- z) .* h̃ .+ z .* h
end


struct MixGRUCell{A,V,S}  <: AbstractActionRNN
    θ::V
    Wi::Vector{A}
    Wa::Vector{A}
    Wh::Vector{A}
    b::Vector{V}
    state0::S
end

function MixGRUCell(in, 
                    na, 
                    out,
                    num_experts; 
                    init = Flux.glorot_uniform, 
                    initb = Flux.zeros, 
                    init_state = Flux.zeros)
    MixGRUCell(
        Flux.ones(num_experts),
        [init(out * 3, in) for _ in 1:num_experts],
        [init(out * 3, na) for _ in 1:num_experts],
        [init(out * 3, out) for _ in 1:num_experts],
        [initb(out * 3) for _ in 1:num_experts],
        init_state(out,1))
end




function (m::MixGRUCell)(h, x::Tuple{A, O}) where {A, O}
    o = size(h, 1)

    a = x[1]
    obs = x[2]

    hps = additive_gru_inner.((x,), (h,), m.Wi, m.Wa, m.Wh, m.b)
    
    # adding together state
    h′ = sum(m.θ .* hps) ./ sum(m.θ)
    # h′ = (m.w[1]*h′1 + m.w[2]*h′2) ./ sum(m.w)

    sz = size(obs)
    return h′, reshape(h′, :, sz[2:end]...)

end

Flux.@functor MixGRUCell

Base.show(io::IO, l::MixGRUCell) =
  print(io, "MixGRUCell()")

"""
    MixGRU(in::Integer, out::Integer)
[Additive Action Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""

MixGRU(a...; ka...) = Flux.Recur(MixGRUCell(a...; ka...))
Flux.Recur(m::MixGRUCell) = Flux.Recur(m, m.state0)



struct MixElGRUCell{A,V,S}  <: AbstractActionRNN
    θ::Vector{V}
    Wi::Vector{A}
    Wa::Vector{A}
    Wh::Vector{A}
    b::Vector{V}
    state0::S
end

function MixElGRUCell(in, 
                    na, 
                    out,
                    num_experts; 
                    init = Flux.glorot_uniform, 
                    initb = Flux.zeros, 
                    init_state = Flux.zeros)
    MixElGRUCell(
        [Flux.ones(out) for _ in 1:num_experts],
        [init(out * 3, in) for _ in 1:num_experts],
        [init(out * 3, na) for _ in 1:num_experts],
        [init(out * 3, out) for _ in 1:num_experts],
        [initb(out * 3) for _ in 1:num_experts],
        init_state(out,1))
end

function (m::MixElGRUCell)(h, x::Tuple{A, O}) where {A, O}
    o = size(h, 1)

    a = x[1]
    obs = x[2]

    hps = additive_gru_inner.((x,), (h,), m.Wi, m.Wa, m.Wh, m.b)
    
    # adding together state
    h′ = sum(m.θ[i] .* hps[i] for i in 1:length(m.θ)) ./ sum(m.θ)
    # h′ = (m.w[1]*h′1 + m.w[2]*h′2) ./ sum(m.w)

    sz = size(obs)
    return h′, reshape(h′, :, sz[2:end]...)

end

Flux.@functor MixElGRUCell

Base.show(io::IO, l::MixElGRUCell) =
  print(io, "MixElGRUCell()")

"""
    MixElGRU(in::Integer, out::Integer)
[Additive Action Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""

MixElGRU(a...; ka...) = Flux.Recur(MixElGRUCell(a...; ka...))
Flux.Recur(m::MixElGRUCell) = Flux.Recur(m, m.state0)
