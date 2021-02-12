
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
    sum(ρ_t.*((v_t - target).^2)) * (1//(2*size(ρ_t)[1]))
end

function offpolicy_tdloss_batch(ρ_t, v_t, c, γ_tp1, ṽ_tp1)
    println(size(ρ_t), size(v_t), size(c), size(γ_tp1), size(ṽ_tp1))
    target = dropgrad(Float32.(c + γ_tp1.*ṽ_tp1))
    sum(ρ_t.*((v_t - target).^2)) * (1//(2*size(ρ_t)[1]))
end

# tdloss(v_t, c, γ_tp1, ṽ_tp1) =
#     (1//2)*length(c)*Flux.mse(v_t, Flux.data(c .+ γ_tp1.*ṽ_tp1))



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
        if !(grads[weights] === nothing)
            Flux.update!(opt, weights, grads[weights])
        end
    end
end


function update_batch!(chain,
                       horde::H,
                       opt,
                       lu::TD,
                       h_init,
                       state_seq,
                       env_state_tp1,
                       action_t=nothing,
                       b_prob=1.0;
                       kwargs...) where {H<:GVFHordes.AbstractHorde}

    reset!(chain, h_init)

    n = length(state_seq)
    preds = map(chain, state_seq)
    v_tp1 = preds[n]

    params = if length(size(v_tp1)) == 1
        dropgrad([get(horde, nothing, action_t[1], env_state_tp1, v_tp1)])
    elseif length(size(env_state_tp1)) == 1
        dropgrad([get(horde, nothing, action_t[i], env_state_tp1[i], v_tp1[:, i]) for i in 1:(size(env_state_tp1)[2])])
    else
        dropgrad([get(horde, nothing, action_t[i], env_state_tp1[:, i], v_tp1[:, i]) for i in 1:(size(env_state_tp1)[2])])
    end

    cumulants = dropgrad(Flux.batch(Base.getindex.(params, 1)))
    discounts = dropgrad(Flux.batch(Base.getindex.(params, 2)))
    ρ = if length(size(env_state_tp1)) == 1
        dropgrad(Flux.batch([Base.getindex(params[1], 3)./b_prob[1]]))
    else
        dropgrad(Flux.batch([Base.getindex(params[i], 3)./b_prob[i] for i in 1:(size(env_state_tp1)[2])]))
    end
    
    ℒ = 0.0f0
    reset!(chain, h_init)
    grads = gradient(Flux.params(chain)) do
        
        preds = map(chain, state_seq)
        v_tp1 = dropgrad(preds[n])

        ℒ = offpolicy_tdloss(ρ, preds[n-1], cumulants, discounts, v_tp1)
    end
    reset!(chain, h_init)
    for weights in Flux.params(chain)
        if !(grads[weights] === nothing)
            Flux.update!(opt, weights, grads[weights])
        end
    end
    ℒ
end


struct QLearning <: LearningUpdate
    γ::Float32
end

get_cart_idx(a, l) = CartesianIndex.(a, 1:l)

function q_learning_loss(q_t, a_t, r, terminal, γ, q_tp1)
    target = dropgrad(r .+ γ*(1-terminal)*maximum(q_tp1))
    return (q_t[a_t] - target).^2
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
    n = length(state_seq)
    grads = gradient(Flux.params(chain)) do
        
        preds = map(chain, state_seq)

        q_tp1 = dropgrad(preds[n])
        # cumulants, discounts, π_prob = dropgrad(get(horde, nothing, action_t, env_state_tp1, v_tp1))
        # ρ = dropgrad(Float32.(π_prob./b_prob))
        # offpolicy_tdloss(ρ, preds[n-1], cumulants, discounts, v_tp1)
        q_learning_loss(preds[end-1], action_t, reward, terminal, lu.γ, q_tp1)
    end
    
    Flux.reset!(chain)
    for weights in Flux.params(chain)
        if !(grads[weights] === nothing) && !(weights isa Flux.Zeros)
            Flux.update!(opt, weights, grads[weights])
        end
    end

    
end

function update_batch!(chain,
                       opt,
                       lu::QLearning,
                       h_init,
                       state_seq,
                       reward,
                       terminal,
                       action_t)

    
    ℒ = 0.0f0
    reset!(chain, h_init)
    γ = lu.γ

    # println(action_t)
    a_t = [CartesianIndex(action_t[i], i) for i in 1:length(action_t)]

    grads = gradient(Flux.params(chain)) do

        preds = map(chain, state_seq)
        # println(size(preds[end]))
        
        q_tp1 = dropgrad(preds[end])
        q_t = preds[end-1]
        target = dropgrad(reward .+ γ*(1 .- terminal).*(maximum(q_tp1;dims=1)[1, :]))
        ℒ = sum((q_t[a_t] - target).^2)
        # ℒ = q_learning_loss(preds[end-1], action_t, reward, terminal, lu.γ, q_tp1)
        # ℒ = offpolicy_tdloss(ρ, preds[n-1], cumulants, discounts, v_tp1)
    end
    reset!(chain, h_init)
    for weights in Flux.params(chain)
        if !(grads[weights] === nothing)
            Flux.update!(opt, weights, grads[weights])
        end
    end
    ℒ
end
