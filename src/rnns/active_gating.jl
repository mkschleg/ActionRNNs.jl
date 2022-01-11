struct AGMoERNNCell{F, G,A,V,S} <: AbstractActionRNN
    σ::F
    gating_network::G
    Wi::Vector{A}
    Wa::Vector{A}
    Wh::Vector{A}
    b::Vector{V}
    state0::S
end

AGMoERNNCell(in::Int,
             na::Int,
             out::Int,
             num_experts::Int,
             gating_net,
             σ=tanh;
             init=Flux.glorot_uniform,
             initb=Flux.zeros,
             init_state=Flux.zeros) = 
                 AGMoERNNCell(σ,
                              # Flux.ones(2),
                              gating_net,
                              [init(out, in) for _ in 1:num_experts],
                              [init(out, na) for _ in 1:num_experts],
                              [init(out, out) for _ in 1:num_experts],
                              [initb(out) for _ in 1:num_experts],
                              init_state(out, 1))




function (m::AGMoERNNCell)(h, x::Tuple{A, X}) where {A, X}
    σ, G, Wi, Wa, Wh, b = m.σ, m.gating_network, m.Wi, m.Wa, m.Wh, m.b

    o = x[2]
    a = x[1]

    # additive
    new_hs = additive_rnn_inner.((x,), (h,), σ, Wi, Wa, Wh, b)
    θ = G(h, x)
    # mix
    new_h = sum(θ .* new_hs) ./ sum(θ)

    sz = size(o)
    return new_h, reshape(new_h, :, sz[2:end]...)
end


Flux.@functor AGMoERNNCell

function Base.show(io::IO, l::AGMoERNNCell)
  print(io, "AGMoERNNCell()")
end

"""
    AGMoERNN(in::Integer, out::Integer, σ = tanh)
The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.
"""

AGMoERNN(a...; ka...) = Flux.Recur(AGMoERNNCell(a...; ka...))
Flux.Recur(m::AGMoERNNCell) = Flux.Recur(m, m.state0)


struct AGMoEGRUCell{G,A,V,S}  <: AbstractActionRNN
    gating_network::G
    Wi::Vector{A}
    Wa::Vector{A}
    Wh::Vector{A}
    b::Vector{V}
    state0::S
end

function AGMoEGRUCell(in, 
                      na, 
                      out,
                      num_experts,
                      gating_net; 
                      init = Flux.glorot_uniform, 
                      initb = Flux.zeros, 
                      init_state = Flux.zeros)
    AGMoEGRUCell(
        gating_net,
        [init(out * 3, in) for _ in 1:num_experts],
        [init(out * 3, na) for _ in 1:num_experts],
        [init(out * 3, out) for _ in 1:num_experts],
        [initb(out * 3) for _ in 1:num_experts],
        init_state(out,1))
end




function (m::AGMoEGRUCell)(h, x::Tuple{A, O}) where {A, O}
    o = size(h, 1)

    a = x[1]
    obs = x[2]

    hps = additive_gru_inner.((x,), (h,), m.Wi, m.Wa, m.Wh, m.b)
    θ = m.gating_network(h, x)
    # adding together state
    h′ = sum(θ .* hps) ./ sum(θ)
    # h′ = (m.w[1]*h′1 + m.w[2]*h′2) ./ sum(m.w)

    sz = size(obs)
    return h′, reshape(h′, :, sz[2:end]...)

end

Flux.@functor AGMoEGRUCell

Base.show(io::IO, l::AGMoEGRUCell) =
  print(io, "AGMoEGRUCell()")

"""
    AGMoEGRU(in::Integer, out::Integer)
[Additive Action Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""

AGMoEGRU(a...; ka...) = Flux.Recur(AGMoEGRUCell(a...; ka...))
Flux.Recur(m::AGMoEGRUCell) = Flux.Recur(m, m.state0)
