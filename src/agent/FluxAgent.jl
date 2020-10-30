import Flux

import Random
import DataStructures

mutable struct FluxAgent{O, C, F, H, Φ, Π, G} <: MinimalRLCore.AbstractAgent
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

function FluxAgent(out_horde,
                   model,
                   opt,
                   τ,
                   feature_creator,
                   feature_size,
                   acting_policy)

    state_list, init_state = begin
        if contains_rnntype(model, ActionRNNs.AbstractActionRNN)
            (DataStructures.CircularBuffer{Tuple{Int64, Array{Float32, 1}}}(τ+1), (0, zeros(Float32, 1)))
        else
            (DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1), zeros(Float32, 1))
        end
    end
    
    hidden_state_init = get_initial_hidden_state(model)
    
    FluxAgent(TD(),
              opt,
              model,
              feature_creator,
              state_list,
              hidden_state_init,
              init_state,
              acting_policy,
              1, 0.0, out_horde)

end

build_new_feat(agent::FluxAgent{O, C, F, H, Φ, Π, G}, state, action) where {O, C, F, H, Φ, Π, G} = 
    agent.build_features(state, action)

build_new_feat(agent::FluxAgent{O, C, F, H, Φ, Π, G}, state, action) where {O, C, F, H, Φ<:Tuple, Π, G}= 
    (action, agent.build_features(state, nothing))




function MinimalRLCore.start!(agent::FluxAgent, env_s_tp1, rng; kwargs...)

    agent.action, agent.action_prob = agent.π(env_s_tp1, rng)

    fill!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))

    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    agent.hidden_state_init = get_initial_hidden_state(agent.model)
    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    return agent.action
end


function MinimalRLCore.step!(agent::FluxAgent, env_s_tp1, r, terminal, rng; kwargs...)

    # println("step:")
    # new_action = sample(rng, agent.π, env_s_tp1)
    new_action, new_prob = agent.π(env_s_tp1, rng)
    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    # println(agent.hidden_state_init)
    # RNN update function
    update!(agent.model,
            agent.horde,
            agent.opt,
            agent.lu,
            agent.hidden_state_init,
            agent.state_list,
            env_s_tp1,
            agent.action,
            agent.action_prob)
    # End update function
    # println(agent.hidden_state_init)
    # Flux.truncate!(agent.model)
    reset!(agent.model, agent.hidden_state_init)
    out_preds = agent.model.(agent.state_list)[end]

    cur_hidden_state = get_hidden_state(agent.model)

    agent.hidden_state_init =
        get_next_hidden_state(agent.model, agent.hidden_state_init, agent.state_list[1])

    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    agent.action = copy(new_action)
    agent.action_prob = new_prob

    return (preds=out_preds, h=cur_hidden_state, action=agent.action)
end

# MinimalRLCore.get_action(agent::FluxAgent, state) = agent.action
