

function _AGMoEActPost_inner(cell, h, x)
    inner_func = get_inner_rnn_func(cell)
    _AGMoE_inner(inner_func, h, x, cell.gating_network, cell.Wi, cell.Wa, cell.Wh, cell.b, cell.σ)
end

function _AGMoEActPost_inner(rnn_inner_func, h, x, G, Wi, Wa, Wh, b, σ)

    a = x[1]
    o = x[2]

    rnn_inner_func.((x,), (h,), Wi, Wa, Wh, b)

    ginput = if size(h, 2) !== size(o, 2) && size(h, 2) == 1
        vcat(h * ones(eltype(h), 1, size(o, 2)), o)
    else
        vcat(h, o)
    end
    
    θ = G((a, ginput))

    # assume softmax output from G
    if θ isa AbstractMatrix
        tθ = θ'
        σ.(sum([h .* tθ for (h, θ) in zip(new_hs, eachslice(θ, dims=1))]))
    else θ isa AbstractVector
        σ.(sum(θ .* new_hs))
    end

end


struct AGMoERNNActPostCell{F, G,A,V,S} <: AbstractActionRNN
    σ::F
    gating_network::G
    # rnncells
    
    Wi::Vector{A}
    Wa::Vector{A}
    Wh::Vector{A}
    b::Vector{V}

    state0::S
end

AGMoERNNActPostCell(in::Int,
             na::Int,
             out::Int,
             num_experts::Int,
             gating_net,
             σ=tanh;
             init=Flux.glorot_uniform,
             initb=Flux.zeros,
             init_state=Flux.zeros) = 
                 AGMoERNNActPostCell(σ,
                              # Flux.ones(2),
                              gating_net,
                              [init(out, in) for _ in 1:num_experts],
                              [init(out, na) for _ in 1:num_experts],
                              [init(out, out) for _ in 1:num_experts],
                              [initb(out) for _ in 1:num_experts],
                              init_state(out, 1))

get_inner_rnn_func(::AGMoERNNActPostCell) = additive_rnn_inner

function (m::AGMoERNNActPostCell)(h, x::Tuple{A, X}) where {A, X}

    new_h = _AGMoEActPost_inner(m, h, x)

    o = x[2]
    sz = size(o)

    return new_h, reshape(new_h, :, sz[2:end]...)
end

Flux.@functor AGMoERNNActPostCell

function Base.show(io::IO, l::AGMoERNNActPostCell)
  print(io, "AGMoERNNActPostCell()")
end

"""
    AGMoERNNActPost(in::Integer, out::Integer, σ = tanh)
The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.
"""

AGMoERNNActPost(a...; ka...) = Flux.Recur(AGMoERNNActPostCell(a...; ka...))
Flux.Recur(m::AGMoERNNActPostCell) = Flux.Recur(m, m.state0)
