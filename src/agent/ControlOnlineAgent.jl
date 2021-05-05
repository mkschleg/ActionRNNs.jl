import Flux

import Random
import DataStructures

mutable struct ControlOnlineAgent{LU<:LearningUpdate, O, C, F, H, Φ, Π} <: MinimalRLCore.AbstractAgent
    lu::LU # I.e. Q-learning or Double Q learning
    opt::O
    model::C
    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    s_t::Φ
    π::Π
    action::Int
    action_prob::Float64
end

function ControlOnlineAgent(model,
                            opt,
                            τ,
                            γ,
                            feature_creator,
                            feature_size,
                            acting_policy)

    state_list, init_state = begin
        if contains_rnntype(model, AbstractActionRNN)
            (DataStructures.CircularBuffer{Tuple{Int64, Array{Float32, 1}}}(τ+1), (0, zeros(Float32, 1)))
        else
            (DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1), zeros(Float32, 1))
        end
    end
    
    hidden_state_init = get_initial_hidden_state(model, 1)
    
    ControlOnlineAgent(QLearningSUM(γ),
                     opt,
                     model,
                     feature_creator,
                     state_list,
                     hidden_state_init,
                     init_state,
                     acting_policy,
                     1, 0.0)

end

# TODO: copied from src/agent/AbstractERAgent.jl and renamed
# so that two identical functions aren't included in agent.jl
# I think this function should eventually be moved to agent_util.jl
# since it should be used across all agents
function get_action_and_prob__(π, values, rng)
    action = 0
    action_prob = 0.0
    if π isa AbstractValuePolicy
        action = sample(rng, π, values)
        action_prob = get_prob(π, values, action)
    else
        action = sample(rng, π)
        action_prob = get_prob(π, action)
    end
    action, action_prob
end

# TODO: copied from src/agent/agent_util.jl, should just be added to agent_util.jl in the future
build_new_feat(agent::ControlOnlineAgent, state, action) = begin
    if eltype(agent.state_list) <: Tuple
        (action, agent.build_features(state, action))
    else
        agent.build_features(state, action)
    end
end

function MinimalRLCore.start!(agent::ControlOnlineAgent, s, rng; kwargs...)

    agent.action = 1

    s_t = build_new_feat(agent, s, agent.action)

    Flux.reset!(agent.model)
    values = agent.model(s_t)

    agent.action, agent.action_prob = get_action_and_prob__(agent.π, values, rng)

    empty!(agent.state_list)

    push!(agent.state_list, build_new_feat(agent, s, agent.action))
    agent.hidden_state_init = get_initial_hidden_state(agent.model, 1)
    agent.s_t = build_new_feat(agent, s, agent.action)
    return agent.action
end


function MinimalRLCore.step!(agent::ControlOnlineAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG; kwargs...)

    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))

    ####
    # Update model
    ####
    us = update!(agent.model,
                 agent.opt,
                 agent.lu,
                 agent.hidden_state_init,
                 agent.state_list,
                 agent.action,
                 r,
                 terminal)

    ####
    # Get predictions and manage hidden state
    ####
    reset!(agent.model, agent.hidden_state_init)
    values = agent.model.(agent.state_list)[end]
    
    cur_hidden_state = get_hidden_state(agent.model, 1)

    is_full = DataStructures.isfull(agent.state_list)
    if is_full
        agent.hidden_state_init =
            get_next_hidden_state(agent.model, agent.hidden_state_init, agent.state_list[1], 1)
    end

    ####
    # Manage small details needed for next step
    ####
    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    agent.action, agent.action_prob = get_action_and_prob__(agent.π, values, rng)

    return (preds=values, h=cur_hidden_state , action=agent.action, update_state=us)
end

