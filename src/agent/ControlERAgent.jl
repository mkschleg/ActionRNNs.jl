

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
    update_wait::Int
    τ::Int
    s_t::Φ
    π::Π
    γ::Float32
    action::Int
    am1::Int
    action_prob::Float64
    hs_learnable::Bool

    beg::Bool
    cur_step::Int
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
                        update_time,
                        acting_policy,
                        hs_learnable)

    
    state_list, init_state = begin
        if contains_rnntype(model, ActionRNNs.AbstractActionRNN)
            (DataStructures.CircularBuffer{Tuple{Int64, Array{Float32, 1}}}(2), (0, zeros(Float32, 1)))
        else
            (DataStructures.CircularBuffer{Array{Float32, 1}}(2), zeros(Float32, 1))
        end
    end

    hidden_state_init = get_initial_hidden_state(model)

    hs_type, hs_length, hs_symbol = ActionRNNs.get_hs_details_for_er(model)

    replay = EpisodicSequenceReplay(replay_size+τ-1,
                                    (Int, Float32, Int, Float32, Float32, Bool, Bool, hs_type...),
                                    (1, feature_size, 1, feature_size, 1, 1, 1, hs_length...),
                                    (:am1, :s, :a, :sp, :r, :t, :beg, hs_symbol...))
    
    ControlERAgent(QLearning(γ),
                   opt,
                   model,
                   feature_creator,
                   state_list,
                   get_initial_hidden_state(model),
                   replay,
                   warm_up,
                   batch_size,
                   update_time,
                   τ,
                   init_state,
                   acting_policy,
                   γ,
                   1, 1, 0.0, hs_learnable, true, 0)
end

build_new_feat(agent::ControlERAgent{O, C, F, H, ER, Φ, Π}, state, action) where {O, C, F, H, ER, Φ, Π} = 
    agent.build_features(state, action)

build_new_feat(agent::ControlERAgent{O, C, F, H, ER, Φ, Π}, state, action) where {O, C, F, H, ER, Φ<:Tuple, Π} = 
    (action, agent.build_features(state, nothing))

add_exp!(agent::ControlERAgent{O, C, F, H, ER, Φ, Π}, env_s_tp1, r, terminal, hs...) where {O, C, F, H, ER, Φ, Π} = 
    push!(agent.replay,
          (agent.am1,
           agent.state_list[1],
           agent.action,
           agent.state_list[2],
           r,
           terminal,
           agent.beg,
           hs...))

add_exp!(agent::ControlERAgent{O, C, F, H, ER, Φ, Π}, env_s_tp1, r, terminal, hs...) where {O, C, F, H, ER, Φ<:Tuple, Π}= begin
    push!(agent.replay,
          (agent.am1,
           agent.state_list[1][2],
           agent.action,
           agent.state_list[2][2],
           r,
           terminal,
           agent.beg,
           hs...))
end


function MinimalRLCore.start!(agent::ControlERAgent, env_s_tp1, rng; kwargs...)

    agent.action = 1
    agent.am1 = 1
    agent.beg = true
    s_t = build_new_feat(agent, env_s_tp1, agent.action)
    values = agent.model(s_t)
    agent.action = sample(agent.π, values, rng)

    empty!(agent.state_list)

    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    agent.hidden_state_init = get_initial_hidden_state(agent.model)
    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    
    return agent.action
end


function MinimalRLCore.step!(agent::ControlERAgent, env_s_tp1, r, terminal, rng; kwargs...)

    # new_action, new_prob = agent.π(env_s_tp1, rng)
    s = build_new_feat(agent, env_s_tp1, agent.action)
    push!(agent.state_list, s)
    is_full = DataStructures.isfull(agent.state_list)

    #Deal with ER buffer
    add_ret = add_exp!(agent,
                       Float32.(env_s_tp1),
                       r,
                       terminal,
                       (agent.hidden_state_init[k] for k in keys(agent.hidden_state_init))...)

    agent.beg = false
    
    us = nothing #ActionRNNs.UpdateState(0.0f0, nothing, nothing, nothing)
    if length(agent.replay) >= agent.warm_up && (agent.cur_step % agent.update_wait == 0)

        τ = agent.τ
        batch_size = agent.batch_size

        exp_idx, exp = sample(rng, agent.replay, batch_size, 2, τ)

        s = if eltype(agent.state_list) <: Tuple
            get_state(seq) = seq.s
            s_1 = Flux.batchseq([[get_state.(seq); [seq[end].sp]] for seq in exp], zero(exp[1][1].s))
            a_1 = [rpad([[seqi_j.am1[] for seqi_j ∈ seq]; [seq[end].a[]]], length(s_1), 1) for seq in exp]
            [([a_1[b][t] for b ∈ 1:length(a_1)], st) for (t, st) ∈ enumerate(s_1)]
        else
            Flux.batchseq([[getindex.(exp[i], :s); [exp[i][end].sp]] for i in 1:batch_size])
        end

        t = [exp[i][end].t[1] for i in 1:batch_size]
        r = [exp[i][end].r[1] for i in 1:batch_size]
        a = [exp[i][end].a[1] for i in 1:batch_size]

        actual_seq_lengths = [length(exp[i]) for i in 1:batch_size]

        hs = ActionRNNs.get_hs_from_experience(agent.model, exp)
        
        us = update_batch!(agent.model,
                           agent.opt,
                           agent.lu,
                           hs,
                           s,
                           r,
                           t,
                           a,
                           actual_seq_lengths)

        if agent.hs_learnable
            modify_hs_in_er!(agent.replay, agent.model, exp, exp_idx, hs)
        end
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
    agent.update_wait += 1
    return (preds=out_preds, h=cur_hidden_state, action=agent.action, update_state=us)
end
