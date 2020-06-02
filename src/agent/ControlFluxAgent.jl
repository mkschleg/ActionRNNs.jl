import Flux

import Random
import DataStructures

mutable struct ControlFluxAgent{LU<:LearningUpdate, O, C, F, H, Φ, Π} <: MinimalRLCore.AbstractAgent
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

function ControlFluxAgent(model,
                          feature_creator,
                          feature_size,
                          acting_policy,
                          parsed;
                          rng=Random.GLOBAL_RNG)

    # num_gvfs = length(out_horde)

    τ=parsed["truncation"]
    opt = FluxUtils.get_optimizer(parsed)

    state_list, init_state = begin
        if contains_rnntype(model, ActionRNN.AbstractActionRNN)
            (DataStructures.CircularBuffer{Tuple{Int64, Array{Float32, 1}}}(τ+1), (0, zeros(Float32, 1)))
        else
            (DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1), zeros(Float32, 1))
        end
    end
    
    hidden_state_init = get_initial_hidden_state(model)
    
    ControlFluxAgent(QLearning(parsed["gamma"]),
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
                         env_type::Type{ControlFluxAgent})
    FluxUtils.opt_settings!(as)
    FluxUtils.rnn_settings!(as)
end


build_new_feat(agent::ControlFluxAgent, state, action) = 
    agent.build_features(state, action)

build_new_feat(agent::ControlFluxAgent{LU, O, C, F, H, Φ, Π}, state, action) where {LU<:LearningUpdate, O, C, F, H, Φ<:Tuple, Π} = 
    (action, agent.build_features(state, nothing))





function MinimalRLCore.start!(agent::ControlFluxAgent, env_s_tp1, rng=Random.GLOBAL_RNG; kwargs...)


    # agent.action, _ = agent.π(env_s_tp1, rng)
    agent.action = 1
    s_t = build_new_feat(agent, env_s_tp1, agent.action)
    values = Flux.data(agent.model(s_t))
    agent.action = sample(agent.π, values, rng)

    # fill!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    empty!(agent.state_list)

    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    agent.hidden_state_init = get_initial_hidden_state(agent.model)
    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    return agent.action
end


function MinimalRLCore.step!(agent::ControlFluxAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG; kwargs...)


    # new_action = sample(rng, agent.π, env_s_tp1)
    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    
    # reset!(agent.model, agent.hidden_state_init)
    # values = Flux.data(agent.model.(agent.state_list))


    
    # RNN update function
    update!(agent.model,
            agent.opt,
            agent.lu,
            agent.hidden_state_init,
            agent.state_list,
            agent.action,
            r,
            terminal)
    # End update function

    Flux.truncate!(agent.model)
    reset!(agent.model, agent.hidden_state_init)
    values = Flux.data(agent.model.(agent.state_list)[end])
    
    agent.action = sample(agent.π, values, rng)


    agent.hidden_state_init =
        get_next_hidden_state(agent.model, agent.hidden_state_init, agent.state_list[1])

    agent.s_t = agent.state_list[end]

    return agent.action
end

# MinimalRLCore.get_action(agent::ControlFluxAgent, state) = agent.action
