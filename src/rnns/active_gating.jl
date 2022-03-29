

function _AGMoE_inner(cell, h, x)
    inner_func = get_inner_rnn_func(cell)
    if hasproperty(cell, :σ)
        _AGMoE_inner(inner_func, h, x, cell.gating_network, cell.Wi, cell.Wa, cell.Wh, cell.b, cell.σ)
    else
        _AGMoE_inner(inner_func, h, x, cell.gating_network, cell.Wi, cell.Wa, cell.Wh, cell.b)
    end
end

function _AGMoE_inner(rnn_inner_func, h, x, G, Wi, Wa, Wh, b, σ=nothing)

    a = x[1]
    o = x[2]
    
    new_hs = if isnothing(σ)
        rnn_inner_func.((x,), (h,), Wi, Wa, Wh, b)
    else
        new_hs = rnn_inner_func.((x,), (h,), σ, Wi, Wa, Wh, b)
    end

    ginput = if size(h, 2) !== size(o, 2) && size(h, 2) == 1
        vcat(h * ones(eltype(h), 1, size(o, 2)), o)
    else
        vcat(h, o)
    end
    
    θ = G((a, ginput))

    # assume softmax output from G
    if θ isa AbstractMatrix
        sum([h .* θ' for (h, θ) in zip(new_hs, eachslice(θ, dims=1))])
    else θ isa AbstractVector
        sum(θ .* new_hs)
    end

end


struct AGMoERNNCell{F, G,A,V,S} <: AbstractActionRNN
    σ::F
    gating_network::G
    # rnncells
    
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

get_inner_rnn_func(::AGMoERNNCell) = additive_rnn_inner

function (m::AGMoERNNCell)(h, x::Tuple{A, X}) where {A, X}

    new_h = _AGMoE_inner(m, h, x)

    o = x[2]
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

get_inner_rnn_func(::AGMoEGRUCell) = additive_gru_inner



function (m::AGMoEGRUCell)(h, x::Tuple{A, O}) where {A, O}
    
    new_h = _AGMoE_inner(m, h, x)

    o = x[2]
    sz = size(o)

    return new_h, reshape(new_h, :, sz[2:end]...)
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
