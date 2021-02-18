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
end

function ControlOnlineAgent(model,
                            opt,
                            τ,
                            γ,
                            feature_creator,
                            feature_size,
                            acting_policy)

    # τ=parsed["truncation"]
    # opt = FluxUtils.get_optimizer(parsed)

    state_list, init_state = begin
        if contains_rnntype(model, AbstractActionRNN)
            (DataStructures.CircularBuffer{Tuple{Int64, Array{Float32, 1}}}(τ+1), (0, zeros(Float32, 1)))
        else
            (DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1), zeros(Float32, 1))
        end
    end
    
    hidden_state_init = get_initial_hidden_state(model)
    
    ControlOnlineAgent(QLearning(γ),
                     opt,
                     model,
                     feature_creator,
                     state_list,
                     hidden_state_init,
                     init_state,
                     acting_policy,
                     1)

end

function agent_settings!(as::Reproduce.ArgParseSettings,
                         env_type::Type{ControlOnlineAgent})
    FluxUtils.opt_settings!(as)
    FluxUtils.rnn_settings!(as)
end


build_new_feat(agent::ControlOnlineAgent, state, action) = 
    agent.build_features(state, action)

build_new_feat(agent::ControlOnlineAgent{LU, O, C, F, H, Φ, Π}, state, action) where {LU<:LearningUpdate, O, C, F, H, Φ<:Tuple, Π} = 
    (action, agent.build_features(state, nothing))





function MinimalRLCore.start!(agent::ControlOnlineAgent, env_s_tp1, rng=Random.GLOBAL_RNG; kwargs...)

    # agent.action, _ = agent.π(env_s_tp1, rng)
    agent.action = 1
    s_t = build_new_feat(agent, env_s_tp1, agent.action)
    values = agent.model(s_t)
    agent.action = sample(agent.π, values, rng)

    # fill!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    empty!(agent.state_list)

    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    agent.hidden_state_init = get_initial_hidden_state(agent.model)
    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    return agent.action
end


function MinimalRLCore.step!(agent::ControlOnlineAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG; kwargs...)


    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    # println(terminal)
    # RNN update function
    loss, l1_grads = update!(agent.model,
                             agent.opt,
                             agent.lu,
                             agent.hidden_state_init,
                             agent.state_list,
                             agent.action,
                             r,
                             terminal)
    # End update function

    # Flux.truncate!(agent.model)
    reset!(agent.model, agent.hidden_state_init)
    values = agent.model.(agent.state_list)[end]
    
    agent.action = sample(agent.π, values, rng)

    if DataStructures.isfull(agent.state_list)
        agent.hidden_state_init =
            get_next_hidden_state(agent.model, agent.hidden_state_init, agent.state_list[1])
    end
    agent.s_t = agent.state_list[end]

    return (action = agent.action, loss = loss, l1_grads = l1_grads, q=values)
end

# MinimalRLCore.get_action(agent::ControlOnlineAgent, state) = agent.action
