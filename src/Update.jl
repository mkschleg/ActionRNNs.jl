
using Statistics
using LinearAlgebra: Diagonal
# import Flux.Tracker.update!
using Flux.Optimise: apply!
using Flux
using Zygote: dropgrad, ignore, Buffer

tderror(v_t, c, γ_tp1, ṽ_tp1) =
    (v_t .- (c .+ γ_tp1.*ṽ_tp1))

# function offpolicy_tdloss(ρ_t::Array{T, 1},
#                           v_t::TrackedArray,
#                           c::Array{T, 1},
#                           γ_tp1::Array{T, 1},
#                           ṽ_tp1::Array{T, 1}) where {T<:AbstractFloat}
function offpolicy_tdloss(ρ_t, v_t, c, γ_tp1, ṽ_tp1)
    target = dropgrad(Float32.(c + γ_tp1.*ṽ_tp1))
    sum(ρ_t.*((v_t - target).^2)) * (1//(2*length(ρ_t)))
end

tdloss(v_t, c, γ_tp1, ṽ_tp1) =
    (1//2)*length(c)*Flux.mse(v_t, Flux.data(c .+ γ_tp1.*ṽ_tp1))



abstract type LearningUpdate end

function update!(out_model, rnn::Flux.Recur{T},
                 horde::H,
                 opt, lu::LearningUpdate, h_init,
                 state_seq, env_state_tp1,
                 action_t=nothing, b_prob=1.0;
                 kwargs...) where {T, H<:GVFHordes.AbstractHorde} end

struct TD <: LearningUpdate end

function update!(chain,
                 horde::H,
                 opt,
                 lu::TD,
                 h_init,
                 state_seq,
                 env_state_tp1,
                 action_t=nothing,
                 b_prob=1.0;
                 kwargs...) where {T, H<:GVFHordes.AbstractHorde}

    reset!(chain, h_init)
    n = length(state_seq)
    grads = gradient(Flux.params(chain)) do
        
        preds = map(chain, state_seq)
        v_tp1 = dropgrad(preds[n])
        cumulants, discounts, π_prob = dropgrad(get(horde, nothing, action_t, env_state_tp1, v_tp1))
        ρ = dropgrad(Float32.(π_prob./b_prob))
        offpolicy_tdloss(ρ, preds[n-1], cumulants, discounts, v_tp1)
        
    end
    reset!(chain, h_init)
    for weights in Flux.params(chain)
        Flux.update!(opt, weights, grads[weights])
    end
end

struct QLearning <: LearningUpdate
    γ::Float32
end

get_cart_idx(a, l) = CartesianIndex.(a, 1:l)

function q_learning_loss(q_t, a_t, r, terminal, γ, q_tp1)
    q_tp1_max = Flux.data(maximum(q_tp1))
    return (q_t[a_t] - (r .+ γ*(1 - terminal)*q_tp1_max)).^2
end

function update!(chain,
                 opt,
                 lu::QLearning,
                 h_init,
                 state_seq,
                 action_t,
                 reward,
                 terminal)

    reset!(chain, h_init)

    grads = Flux.Tracker.gradient(Flux.params(chain)) do
        preds = chain.(state_seq)
        for i in 1:n
            preds[i] = chain(state_seq[i])
        end
        q_tp1 = dropgrad(preds[n])
        q_learning_loss(preds[end-1], action_t, reward, terminal, lu.γ, q_tp1)
    end
    reset!(chain, h_init)
    for weights in Flux.params(chain)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end
