
import Flux

import Random
import DataStructures
import MinimalRLCore

mutable struct ControlImageERAgent{O, C, H, ER, Φ, Φ2,  Π} <: MinimalRLCore.AbstractAgent
    lu::LearningUpdate
    opt::O
    model::C
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    replay::ER
    warm_up::Int
    batch_size::Int
    update_wait::Int
    τ::Int
    
    raw_s_t::Φ2
    π::Π
    γ::Float32
    action::Int
    am1::Int
    action_prob::Float64
    hs_learnable::Bool

    beg::Bool
    cur_step::Int
end

function ControlImageERAgent(model,
                             opt,
                             τ,
                             γ,
                             
                             # feature_creator,
                             # feature_size,
                             
                             env_state_shape,
                             env_state_type,
                             
                             replay_size,
                             warm_up,
                             batch_size,
                             update_time,
                             
                             acting_policy,
                             hs_learnable)

    
    state_list, init_state = begin
        if contains_rnntype(model, ActionRNNs.AbstractActionRNN)
            (DataStructures.CircularBuffer{Tuple{Int64, Array{Float32, 4}}}(2), (0, zeros(Float32, 1, 1)))
        else
            (DataStructures.CircularBuffer{Array{Float32, 4}}(2), zeros(Float32, 1, 1))
        end
    end

    hidden_state_init = get_initial_hidden_state(model)

    hs_type, hs_length, hs_symbol = ActionRNNs.get_hs_details_for_er(model)

    replay = EpisodicSequenceReplay(replay_size+τ-1,
                                    (Int, Int, Int, Int, Float32, Bool, Bool, hs_type...),
                                    (1, 1, 1, 1, 1, 1, 1, hs_length...),
                                    (:am1, :s, :a, :sp, :r, :t, :beg, hs_symbol...))

    sb = StateBuffer{env_state_type}(replay_size+τ*2, env_state_shape)
    
    image_replay = ImageReplay(replay, sb, identity, (img) -> Float32.(img .// 255))
    
    ControlImageERAgent(QLearning(γ),
                        opt,
                        model,
                        state_list,
                        get_initial_hidden_state(model),
                        image_replay,
                        warm_up,
                        batch_size,
                        update_time,
                        τ,
                        zeros(env_state_type, env_state_shape),
                        acting_policy,
                        γ,
                        1, 1, 0.0, hs_learnable, true, 0)
end

add_dim(x::Array{T, N}) where {T,N} = reshape(x, Val(N+1))

build_new_feat(x, agent::ControlImageERAgent, state, action) = 
    add_dim(proc_state(agent.replay, state))

build_new_feat(::Type{<:Tuple}, agent::ControlImageERAgent, state, action) = 
    (action, add_dim(proc_state(agent.replay, state)))

build_new_feat(agent, state, action) =
    build_new_feat(eltype(agent.state_list), agent, state, action)

function add_exp!(agent::ControlImageERAgent, env_s_tp1, r, terminal, hs...)
    _, _, hs_symbol = ActionRNNs.get_hs_details_for_er(agent.model)
    push!(agent.replay,
          (am1 = agent.am1,
           s = agent.raw_s_t,
           a = agent.action,
           sp = env_s_tp1,
           r = r,
           t = terminal,
           beg = agent.beg,
           zip(hs_symbol, hs)...))
end

function MinimalRLCore.start!(agent::ControlImageERAgent, env_s_tp1, rng; kwargs...)

    agent.action = 1
    agent.am1 = 1
    agent.beg = true
    s_t = build_new_feat(agent, env_s_tp1, agent.action)
    Flux.reset!(agent.model)
    values = agent.model(s_t)
    agent.action = sample(agent.π, values, rng)

    empty!(agent.state_list)

    push!(agent.state_list, build_new_feat(agent, env_s_tp1, agent.action))
    agent.hidden_state_init = get_initial_hidden_state(agent.model)
    agent.raw_s_t = env_s_tp1

    start_statebuffer!(agent.replay, env_s_tp1)
    
    return agent.action
end


function MinimalRLCore.step!(agent::ControlImageERAgent, env_s_tp1, r, terminal, rng; kwargs...)

    # new_action, new_prob = agent.π(env_s_tp1, rng)
    s = build_new_feat(agent, env_s_tp1, agent.action)
    push!(agent.state_list, s)
    is_full = DataStructures.isfull(agent.state_list)

    #Deal with ER buffer
    add_ret = add_exp!(agent,
                       env_s_tp1,
                       r,
                       terminal,
                       (agent.hidden_state_init[k] for k in keys(agent.hidden_state_init))...)

    agent.beg = false
    
    us = nothing #ActionRNNs.UpdateState(0.0f0, nothing, nothing, nothing)
    if length(agent.replay) >= agent.warm_up && (agent.cur_step % agent.update_wait == 0)

        τ = agent.τ
        batch_size = agent.batch_size

        exp_s_idx, (actual_seq_lengths, exp) = sample(rng, agent.replay, batch_size, τ)

        t = [exp.t[i][end] for i in 1:batch_size]
        r = [exp.r[i][end] for i in 1:batch_size]
        a = [exp.a[i][end] for i in 1:batch_size]

        s = if eltype(agent.state_list) <: Tuple
            [(exp.am1[i], exp.s[i]) for i in 1:length(exp.s)]
        else
            exp.s
        end

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
            modify_hs_in_er!(agent.replay, agent.model, exp, exp_s_idx, hs)
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

    agent.raw_s_t = env_s_tp1
    agent.am1 = copy(agent.action)
    agent.action = sample(agent.π, out_preds, rng)
    agent.update_wait += 1
    return (preds=out_preds, h=cur_hidden_state, action=agent.action, update_state=us)
end
