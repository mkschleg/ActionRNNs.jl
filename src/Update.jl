
using Statistics
using LinearAlgebra: Diagonal
import Flux.Tracker.update!

using Flux.Optimise: apply!

abstract type LearningUpdate end

function update!(out_model, rnn::Flux.Recur{T},
                 horde::H,
                 opt, lu::LearningUpdate, h_init,
                 state_seq, env_state_tp1,
                 action_t=nothing, b_prob=1.0;
                 kwargs...) where {T, H<:AbstractHorde} end

struct TD <: LearningUpdate
end

function update!(out_model, rnn::Flux.Recur{T},
                 horde::H,
                 opt, lu::TD, h_init,
                 state_seq, env_state_tp1,
                 action_t=nothing, b_prob=1.0;
                 kwargs...) where {T, H<:AbstractHorde}

    reset!(rnn, h_init)
    rnn_out = rnn.(state_seq)
    preds = out_model.(rnn_out)
    cumulants, discounts, π_prob = get(horde, action_t, env_state_tp1, Flux.data(preds[end]))
    ρ = Float32.(π_prob./b_prob)
    δ = offpolicy_tdloss(ρ, preds[end-1], Float32.(cumulants), Float32.(discounts), Flux.data(preds[end]))

    grads = Flux.Tracker.gradient(()->δ, Flux.params(out_model, rnn))
    reset!(rnn, h_init)
    for weights in Flux.params(out_model, rnn)
        Flux.Tracker.update!(opt, weights, grads[weights])
    end
end

