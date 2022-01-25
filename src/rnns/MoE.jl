


import ChainRulesCore

struct MoERNNCell{F,M,G,V,S} <: AbstractActionRNN
    σ::F
    experts::Vector{M}
    gating_network::G
    w::V
    k::Int
    state0::S
end

Flux.@functor MoE

function (l::MoERNNCell)(h, X::Tuple{A, X}) where {A, X}

    a = X[1]
    obs = X[2]
    
    g_out = l.gating_network(h, X)
    stk_out = soft_top_k(g_out, l.k)
    sum([stk_out[i] * l.experts[i](h, X) ])
    
    # ranks = soft_top_k(W*y, l.k)

    experts = l.experts[top_k_experts]
    reduce(vcat, [expert(x) for expert in experts])
    
end

function soft_top_k(x::AbstractVector, k)
    srt = partialsortperm(x, 1:k)
    ret = fill(-Inf, size(x))
    ret[srt] .= x[srt]
    softmax(ret)
end

function soft_top_k(X::AbstractMatrix, k)
    ranks = [partialsortperm(@view X[:, i], 1:k) for i in 1:size(X, 2)]
    ret = fill(-Inf, size(X))
    for i in 1:size(X, 2)
        ret[ranks[i], i] .= X[ranks[i], i]
    end
    softmax(ret; dims=1)
end

function ChainRulesCore.rrule(::typeof(soft_top_k), x, k)
    srt = partialsortperm(x, 1:k)
    xnew = fill(-Inf, size(x))
    xnew[srt] .= x[srt]
    Y = softmax(xnew)
    function soft_top_k_pullback(∇y)
	∇f = NoTangent()
	∇x = ∇softmax(unthunk(∇y), xnew, Y)
	∇k = NoTangent()
	∇f, ∇x, ∇k
    end

    Y, soft_top_k_pullback
end
