

import Flux

import Random
import DataStructures
import MinimalRLCore


mutable struct ControlERAgent{O, C, F, H, ER, Φ,  Π} <: MinimalRLCore.AbstractAgent
    lu::LearningUpdate
    opt::O
    model::C
    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    replay::ER
    warm_up::Int
    batch_size::Int
    τ::Int
    s_t::Φ
    π::Π
    γ::Float32
    action::Int
    am1::Int
    action_prob::Float64
end

function ControlERAgent(model,
                        opt,
                        τ,
                        γ,
                        feature_creator,
                        feature_size,
                        env_state_size,
                        replay_size,
                        warm_up,
                        batch_size,
                        acting_policy)

    
    state_list, init_state = begin
        if contains_rnntype(model, ActionRNNs.AbstractActionRNN)
            (DataStructures.CircularBuffer{Tuple{Int64, Array{Float32, 1}}}(2), (0, zeros(Float32, 1)))
        else
            (DataStructures.CircularBuffer{Array{Float32, 1}}(2), zeros(Float32, 1))
        end
    end

    println(typeof(state_list))

    hidden_state_init = get_initial_hidden_state(model[1])
    println(hidden_state_init)
    println(eltype(hidden_state_init))
    # TODO: Fix sequence Replay for episodes.
    replay = SequenceReplay(replay_size+τ-1,
                            (Int, Float32, Int, Float32, Float32, Float32, Float32, Bool, eltype(hidden_state_init)),
                            (1, feature_size, 1, feature_size, env_state_size, 1, 1, 1, length(hidden_state_init)),
                            (:am1, :s, :a, :sp, :esp, :ap, :r, :t, :hs))


    ControlERAgent(QLearning(γ),
                   opt,
                   model,
                   feature_creator,
                   state_list,
                   get_initial_hidden_state(model),
                   replay,
                   warm_up,
                   batch_size,
                   τ,
                   init_state,
                   acting_policy,
                   γ,
                   1, 1, 0.0)

end

build_new_feat(agent::ControlERAgent{O, C, F, H, ER, Φ, Π}, state, action) where {O, C, F, H, ER, Φ, Π} = 
    agent.build_features(state, action)

build_new_feat(agent::ControlERAgent{O, C, F, H, ER, Φ, Π}, state, action) where {O, C, F, H, ER, Φ<:Tuple, Π} = 
    (action, agent.build_features(state, nothing))

add_exp!(agent::ControlERAgent{O, C, F, H, ER, Φ, Π}, env_s_tp1, r, terminal, hs) where {O, C, F, H, ER, Φ, Π} = 
    push!(agent.replay,
          (agent.am1,
           agent.state_list[1],
           agent.action,
           agent.state_list[2],
           env_s_tp1,
           Float32(agent.action_prob),
           r,
           terminal,
           hs))

add_exp!(agent::ControlERAgent{O, C, F, H, ER, Φ, Π}, env_s_tp1, r, terminal, hs) where {O, C, F, H, ER, Φ<:Tuple, Π}= begin
    push!(agent.replay,
          (agent.am1,
           agent.state_list[1][2],
           agent.action,
           agent.state_list[2][2],
           env_s_tp1,
           Float32(agent.action_prob),
           r,
           terminal,
           hs))
end


function MinimalRLCore.start!(agent::ControlERAgent, env_s_tp1, rng; kwargs...)

    agent.action = 1
    s_t = build_new_feat(agent, env_s_tp1, agent.action)
    values = agent.model(s_t)
    agent.action = sample(agent.π, values, rng)

    # fill!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    empty!(agent.state_list)

    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    agent.hidden_state_init = get_initial_hidden_state(agent.model)
    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    
    # agent.action, agent.action_prob = agent.π(env_s_tp1, rng)

    # fill!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    # # push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    
    # agent.hidden_state_init = get_initial_hidden_state(agent.model)
    # agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    return agent.action
end


function MinimalRLCore.step!(agent::ControlERAgent, env_s_tp1, r, terminal, rng; kwargs...)

    # new_action, new_prob = agent.π(env_s_tp1, rng)
    
    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    is_full = DataStructures.isfull(agent.state_list)

    #Deal with ER buffer

    add_ret = add_exp!(agent,
                       Float32.(env_s_tp1),
                       r,
                       terminal,
                       (agent.hidden_state_init[k] for k in keys(agent.hidden_state_init))...)
    
    ℒ = 0.0f0
    if length(agent.replay) >= agent.warm_up

        bs = rand(1:(length(agent.replay) + 1 - agent.τ), agent.batch_size)
        exp = [view(agent.replay, bs .+ (i-1)) for i ∈ 1:agent.τ]

        state_list = if eltype(agent.state_list) <: Tuple
            sl = [(collect(exp[i].am1), collect(exp[i].s)) for i in 1:length(exp)]
            push!(sl, (collect(exp[end].a), collect(exp[end].sp)))
        else
            sl = [exp[i].s for i in 1:length(exp)]
            push!(sl, exp[end].sp)
        end

        sp1 = collect(exp[end].esp')

        action = exp[end].a


        hs = IdDict()
        hs[agent.model[1]] = collect(exp[1].hs)

        ℒ = update_batch!(agent.model,
                          agent.opt,
                          agent.lu,
                          hs,
                          state_list,
                          exp[end].r,
                          exp[end].t,
                          exp[end-1].a)
    end
    # End update function

    reset!(agent.model, agent.hidden_state_init)
    out_preds = agent.model.(agent.state_list)[end]

    cur_hidden_state = get_hidden_state(agent.model)

    if is_full
        agent.hidden_state_init =
            get_next_hidden_state(agent.model, agent.hidden_state_init, agent.state_list[1])
    end

    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    agent.am1 = copy(agent.action)
    agent.action = sample(agent.π, out_preds, rng)

    return (preds=out_preds, h=cur_hidden_state, action=agent.action, loss=ℒ)
end
