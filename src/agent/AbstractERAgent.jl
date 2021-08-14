


"""
    AbstractERAgent

example agent:
mutable struct DRQNAgent{ER, Φ,  Π, HS<:AbstractMatrix{Float32}} <: AbstractERAgent
    lu::LearningUpdate
    opt::O
    model::C
    target_network::CT

    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}

    hidden_state_init::Dict{Symbol, HS}
    
    replay::ER
    update_timer::UpdateTimer
    target_update_timer::UpdateTimer

    batch_size::Int
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

    hs_tr_init::Dict{Symbol, HS}
end

"""
abstract type AbstractERAgent{LU, ER, TN, DEV} <: AbstractAgent end

get_replay_buffer(agent::AbstractERAgent) = agent.replay
get_learning_update(agent::AbstractERAgent) = agent.lu
get_device(agent::AbstractERAgent) = agent.device



function get_action_and_prob(π, values, rng)
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


function MinimalRLCore.start!(agent::AbstractERAgent, s, rng; kwargs...)

    if true #agent.device isa GPU
        #=
        new probably more sensible behaviour. 
        
        The new behaviour uses a starting action of agent.action=1 
        in constructing the initial agent.s_t and adding to agent.state_list.
        =#
        agent.action = 1
        agent.am1 = 1
        agent.beg = true

        empty!(agent.state_list)

        if agent.replay isa ImageReplay
            start_statebuffer!(agent.replay, s)
        end

        agent.s_t = build_new_feat(agent, s, agent.action)
        push!(agent.state_list, agent.s_t)
        
        Flux.reset!(agent.model)
        values = [agent.model(s_t) for s_t in agent.state_list][end] 
        
        agent.action, agent.action_prob = get_action_and_prob(agent.π, values, rng)
        
        agent.hidden_state_init = get_initial_hidden_state(agent.model, 1)
        
        return agent.action
    else
        #=
        Old behaviour.

        The old behaviour uses the action sampled from the 
        inistial state to build the agent.state_list and 
        agent.s_t. This doesn't seem sensible, and a weird decision.
        Kept for backward compatibility.
        =#
        agent.action = 1
        agent.am1 = 1
        agent.beg = true

        s_t = build_new_feat(agent, s, agent.action)
        
        Flux.reset!(agent.model)
        values = agent.model(s_t)

        agent.action, agent.action_prob = get_action_and_prob(agent.π, values, rng)
        
        empty!(agent.state_list)

        if agent.replay isa ImageReplay
            start_statebuffer!(agent.replay, s)
        end

        push!(agent.state_list, build_new_feat(agent, s, agent.action))
        agent.hidden_state_init = get_initial_hidden_state(agent.model, 1)
        agent.s_t = build_new_feat(agent, s, agent.action)
        
        return agent.action
    end
end


function MinimalRLCore.step!(agent::AbstractERAgent, env_s_tp1, r, terminal, rng; kwargs...)

    push!(agent.state_list,
          build_new_feat(agent, env_s_tp1, agent.action))

    ####
    # Add new experience to replay buffer
    ####
    hs_sym_list = get_hs_symbol_list(agent.model)
    add_ret = add_exp!(agent,
                       env_s_tp1,
                       r,
                       terminal,
                       hs_sym_list,
                       (agent.hidden_state_init[k] for k in hs_sym_list))
    
    agent.beg = false

    ###
    # Update model
    ###
    us = if agent.update_timer(length(agent.replay))
        update!(agent, rng)
    end
    if agent.target_update_timer(length(agent.replay))
        update_target_network!(agent)
    end

    # progress update_timers
    step!(agent.update_timer), step!(agent.target_update_timer)


    ####
    # Get predictions and manage hidden state
    ####i
    reset!(agent.model, agent.hidden_state_init)
    values = agent.model.(agent.state_list)[end]

    cur_hidden_state = get_hidden_state(agent.model)

    is_full = DataStructures.isfull(agent.state_list)
    if is_full
        agent.hidden_state_init =
            get_next_hidden_state(agent.model, agent.hidden_state_init, agent.state_list[1], 1)
    end

    ####
    # Manage small details needed for next step
    ####
    agent.s_t = build_new_feat(agent, env_s_tp1, agent.action)
    agent.am1 = copy(agent.action)

    agent.action, agent.action_prob = get_action_and_prob(agent.π, values, rng)

    next!(agent.π)
    
    return (preds=values, h=cur_hidden_state, action=agent.action, update_state=us)
end


function update!(agent::AbstractERAgent{LU}, rng) where {LU<:ControlUpdate}

    τ = agent.τ
    batch_size = agent.batch_size

    exp_idx, exp = sample(rng, agent.replay, batch_size, τ)

    params = get_information_from_experience(agent, exp)
    
    if agent.replay isa ImageReplay
        ActionRNNs.get_hs_from_experience!(agent.model, exp[2], agent.hs_tr_init, get_device(agent))
    else
        ActionRNNs.get_hs_from_experience!(agent.model, exp, agent.hs_tr_init, get_device(agent))
    end

    us = update_batch!(agent.lu,
                       agent.model,
                       agent.target_network,
                       agent.opt,
                       agent.hs_tr_init,
                       params; device=get_device(agent))
    
    if agent.hs_learnable
        modify_hs_in_er!(agent.replay, agent.model, exp, exp_idx, agent.hs_tr_init, us.grads, agent.opt, get_device(agent))
    end
    
    us
end

function update!(agent::AbstractERAgent{LU}, rng) where {LU<:PredictionUpdate}

    τ = agent.τ
    batch_size = agent.batch_size
    
    exp_idx, exp = sample(rng, agent.replay, batch_size, τ)

    # Make batch.

    params = get_information_from_experience(agent, exp)

    hs = ActionRNNs.get_hs_from_experience(agent.model, exp, 1)
    for k ∈ keys(hs)
        h = get!(()->zero(hs[k]), agent.hs_tr_init, k)
        copyto!(h, hs[k])
    end
    
    us = update_batch!(agent.lu,
                       agent.model,
                       agent.horde,
                       agent.opt,
                       agent.hs_tr_init,
                       params)

    
    if agent.hs_learnable
        modify_hs_in_er!(agent.replay, agent.model, exp, exp_idx, agent.hs_tr_init, us.grads, agent.opt)
    end
    us 


end

update_target_network!(agent::AbstractERAgent) = begin
    update_target_network!(agent.model, agent.target_network)
end

update_target_network!(agent::AbstractERAgent{LU, ER, Nothing}) where {LU, ER} = nothing
