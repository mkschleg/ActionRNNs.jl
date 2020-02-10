
using Statistics
using LinearAlgebra: Diagonal
import Flux.Tracker.update!
import RLCore: AbstractHorde

using Flux.Optimise: apply!

tderror(v_t, c, γ_tp1, ṽ_tp1) =
    (v_t .- (c .+ γ_tp1.*ṽ_tp1))

function offpolicy_tdloss(ρ_t::Array{T, 1},
                          v_t::TrackedArray,
                          c::Array{T, 1},
                          γ_tp1::Array{T, 1},
                          ṽ_tp1::Array{T, 1}) where {T<:AbstractFloat}
    target = T.(c .+ γ_tp1.*ṽ_tp1)
    return (T(0.5))*sum(ρ_t.*((v_t .- target).^2)) * (1//length(ρ_t))
end

tdloss(v_t, c, γ_tp1, ṽ_tp1) =
    (1//2)*length(c)*Flux.mse(v_t, Flux.data(c .+ γ_tp1.*ṽ_tp1))



abstract type LearningUpdate end

function update!(out_model, rnn::Flux.Recur{T},
                 horde::H,
                 opt, lu::LearningUpdate, h_init,
                 state_seq, env_state_tp1,
                 action_t=nothing, b_prob=1.0;
                 kwargs...) where {T, H<:AbstractHorde} end

struct TD <: LearningUpdate end

function update!(out_model, rnn::Flux.Recur{T},
                 horde::H,
                 opt, lu::TD, h_init,
                 state_seq, env_state_tp1,
                 action_t=nothing, b_prob=1.0;
                 kwargs...) where {T, H<:AbstractHorde}

    reset!(rnn, h_init)
    rnn_out = rnn.(state_seq)
    preds = out_model.(rnn_out)
    cumulants, discounts, π_prob = get(horde, nothing, action_t, env_state_tp1, Flux.data(preds[end]))
    ρ = Float32.(π_prob./b_prob)
    δ = offpolicy_tdloss(ρ, preds[end-1], Float32.(cumulants), Float32.(discounts), Flux.data(preds[end]))

    grads = Flux.Tracker.gradient(()->δ, Flux.params(out_model, rnn))
    reset!(rnn, h_init)
    for weights in Flux.params(out_model, rnn)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end

function update!(chain,
                 horde::H,
                 opt,
                 lu::TD,
                 h_init,
                 state_seq,
                 env_state_tp1,
                 action_t=nothing,
                 b_prob=1.0;
                 kwargs...) where {T, H<:AbstractHorde}

    reset!(chain, h_init)
    preds = chain.(state_seq)
    cumulants, discounts, π_prob = get(horde, nothing, action_t, env_state_tp1, Flux.data(preds[end]))
    ρ = Float32.(π_prob./b_prob)
    δ = offpolicy_tdloss(ρ, preds[end-1], Float32.(cumulants), Float32.(discounts), Flux.data(preds[end]))

    grads = Flux.Tracker.gradient(()->δ, Flux.params(chain))
    reset!(chain, h_init)
    for weights in Flux.params(chain)
        Flux.Tracker.update!(opt, weights, grads[weights])
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
# function loss(lu::QLearning, model, s_t, a_t, s_tp1, r, terminal, target_model)
    
#     action_idx = get_cart_idx(a_t, length(terminal))
#     q_tp1 = if target_model isa Nothing
#         dropgrad(maximum(model(s_tp1); dims=1)[1, :])
#     else
#         dropgrad(maximum(target_model(s_tp1); dims=1)[1, :])
#     end
#     q_t = @view model(s_t)[action_idx]
    
#     return mean(huber_loss.(q_t, dropgrad(r .+ (1 .- terminal).*lu.γ.*q_tp1)))
# end


function update!(chain,
                 opt,
                 lu::QLearning,
                 h_init,
                 state_seq,
                 action_t,
                 reward,
                 terminal)

    reset!(chain, h_init)
    preds = chain.(state_seq)

    # δ = offpolicy_tdloss(ρ, preds[end-1], Float32.(cumulants), Float32.(discounts), Flux.data(preds[end]))
    δ = q_learning_loss(preds[end-1], action_t, reward, terminal, lu.γ, preds[end])

    grads = Flux.Tracker.gradient(()->δ, Flux.params(chain))
    reset!(chain, h_init)
    for weights in Flux.params(chain)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end
