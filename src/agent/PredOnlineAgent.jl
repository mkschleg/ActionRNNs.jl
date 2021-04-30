import Flux

import Random
import DataStructures

mutable struct PredOnlineAgent{O, C, F, H, Φ, Π, G} <: MinimalRLCore.AbstractAgent
    lu::LearningUpdate
    opt::O
    model::C
    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    s_t::Φ
    π::Π
    action::Int64
    action_prob::Float64
    horde::Horde{G}
end


function PredOnlineAgent(out_horde,
                         model,
                         opt,
                         τ,
                         feature_creator,
                         acting_policy)

    # TODO: change this to handle device
    state_list, init_state = begin
        if contains_rnntype(model, ActionRNNs.AbstractActionRNN)
            (DataStructures.CircularBuffer{Tuple{Int64, Array{Float32, 1}}}(τ+1), (0, zeros(Float32, 1)))
        else
            (DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1), zeros(Float32, 1))
        end
    end
    
    hidden_state_init = get_initial_hidden_state(model, 1)
    
    PredOnlineAgent(TD(),
                    opt,
                    model,
                    feature_creator,
                    state_list,
                    hidden_state_init,
                    init_state,
                    acting_policy,
                    1, 0.0, out_horde)

end

# build_new_feat(agent::PredOnlineAgent{O, C, F, H, Φ, Π, G}, state, action) where {O, C, F, H, Φ, Π, G} =
#     agent.build_features(state, action)
#
# build_new_feat(agent::PredOnlineAgent{O, C, F, H, Φ, Π, G}, state, action) where {O, C, F, H, Φ<:Tuple, Π, G}=
#     (action, agent.build_features(state))


# TODO: copied from src/agent/AbstractERAgent.jl and renamed
# so that two identical functions aren't included in agent.jl
# I think this function should eventually be moved to agent_util.jl
# since it should be used across all agents
function get_action_and_prob_(π, values, rng)
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

# copied from src/agent/agent_util.jl
build_new_feat(agent::PredOnlineAgent, state, action) = begin
    if eltype(agent.state_list) <: Tuple
        (action, agent.build_features(state, action))
    else
        agent.build_features(state, action)
    end
end


function MinimalRLCore.start!(agent::PredOnlineAgent, s, rng; kwargs...)

    agent.action = 1

    s_t = build_new_feat(agent, s, agent.action)

    Flux.reset!(agent.model)
    values = agent.model(s_t)

    agent.action, agent.action_prob = get_action_and_prob_(agent.π, values, rng)

    empty!(agent.state_list)


#     agent.action, agent.action_prob = agent.π(env_s_tp1, rng)

#     fill!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))

    push!(agent.state_list, build_new_feat(agent, s, agent.action))
    agent.hidden_state_init = get_initial_hidden_state(agent.model, 1)
    agent.s_t = build_new_feat(agent, s, agent.action)
    return agent.action
end


# function MinimalRLCore.start!(agent::PredOnlineAgent, env_s_tp1, rng; kwargs...)
#
#     agent.action, agent.action_prob = agent.π(env_s_tp1, rng)
#
#     fill!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
#
#     push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
#     agent.hidden_state_init = get_initial_hidden_state(agent.model)
#     agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
#     return agent.action
# end


function MinimalRLCore.step!(agent::PredOnlineAgent, env_s_tp1, r, terminal, rng; kwargs...)

    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))

    ####
    # Update model
    ####
    update!(agent.model,
            agent.horde,
            agent.opt,
            agent.lu,
            agent.hidden_state_init,
            agent.state_list,
            env_s_tp1,
            agent.action,
            agent.action_prob)

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

    #TODO: review this
#     agent.hidden_state_init =
#         get_next_hidden_state(agent.model, agent.hidden_state_init, agent.state_list[1], 1)

    ####
    # Manage small details needed for next step
    ####
    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    agent.action, agent.action_prob = get_action_and_prob_(agent.π, values, rng)

    return (preds=values, h=cur_hidden_state, action=agent.action, loss=0.0f0)
end


# function MinimalRLCore.step!(agent::PredOnlineAgent, env_s_tp1, r, terminal, rng; kwargs...)
#
#     new_action, new_prob = agent.π(env_s_tp1, rng)
#     push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
#
#     RNN update function
#     update!(agent.model,
#             agent.horde,
#             agent.opt,
#             agent.lu,
#             agent.hidden_state_init,
#             agent.state_list,
#             env_s_tp1,
#             agent.action,
#             agent.action_prob)
#     End update function
#
#     reset!(agent.model, agent.hidden_state_init)
#     out_preds = agent.model.(agent.state_list)[end]
#
#     cur_hidden_state = get_hidden_state(agent.model)
#
#     agent.hidden_state_init =
#         get_next_hidden_state(agent.model, agent.hidden_state_init, agent.state_list[1])
#
#     agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
#     agent.action = copy(new_action)
#     agent.action_prob = new_prob
#
#     return (preds=out_preds, h=cur_hidden_state, action=agent.action, loss=0.0f0)
# end

# MinimalRLCore.get_action(agent::PredOnlineAgent, state) = agent.action
