


"""
    AbstractERAgent

The abstract struct for building experience replay agents.

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

"""
    get_replay(agent::AbstractERAgent)

Get the replay buffer from the agent.
"""
get_replay(agent::AbstractERAgent) = agent.replay

"""
    get_learning_update(agent::AbstractERAgent)

Get the learning update from the agent.
"""
get_learning_update(agent::AbstractERAgent) = agent.lu

"""
    get_device(agent::AbstractERAgent)

Get the current device from the agent.
"""
get_device(agent::AbstractERAgent) = agent.device

"""
    get_hs_replay_strategy(agent::AbstractERAgent)

Get the replay strategy of the agent.
"""
get_hs_replay_strategy(agent::AbstractERAgent) = @error "Need to implement `get_hs_replay_strategy`"

"""
    get_model(agent::AbstractERAgent)

return the model from the agent.
"""
get_model(agent::AbstractERAgent) = agent.model


"""
    get_action_and_prob(π, values, rng)

Get action and the associated probability of taking the action.
"""
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

"""
        MinimalRLCore.start!(agent::AbstractERAgent, s, rng; kwargs...)

Start the agent for a new episode. 
"""
function MinimalRLCore.start!(agent::AbstractERAgent, s, rng=Random.GLOBAL_RNG; kwargs...)

    #=
    new probably more sensible behaviour. 
    
    The new behaviour uses a starting action of agent.action=1 
    in constructing the initial agent.s_t and adding to agent.state_list.
    =#
    agent.action = 1
    agent.am1 = 1
    agent.beg = true

    empty!(agent.state_list)

    replay = get_replay(agent)
    if replay isa ImageReplay
        start_statebuffer!(replay, s)
    end

    agent.s_t = build_new_feat(agent, s, agent.action)
    push!(agent.state_list, agent.s_t)
    
    Flux.reset!(agent.model)
    values = [agent.model(s_t) for s_t in agent.state_list][end] 
    
    agent.action, agent.action_prob = get_action_and_prob(agent.π, values, rng)
    
    agent.hidden_state_init = get_initial_hidden_state(agent.model)
    
    return agent.action

end

"""
    MinimalRLCore.step!(agent::AbstractERAgent, env_s_tp1, r, terminal, rng; kwargs...)

step! for an experience replay agent.
"""
function MinimalRLCore.step!(agent::AbstractERAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG; kwargs...)

    replay = get_replay(agent)
    state_list = agent.state_list
    model = agent.model


    push!(state_list,
          build_new_feat(agent, env_s_tp1, agent.action))


    
    ####
    # Add new experience to replay buffer
    ####
    hs_sym_list = get_hs_symbol_list(model)
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
    us = if agent.update_timer(length(replay))
        update!(agent, rng)
    end
    
    if agent.target_update_timer(length(replay))
        update_target_network!(agent)
    end

    # progress update_timers
    step!(agent.update_timer), step!(agent.target_update_timer)


    ####
    # Get predictions and manage hidden state
    ####i
    reset!(model, agent.hidden_state_init)
    values = [model(obs) for obs in state_list][end]

    cur_hidden_state = get_hidden_state(model)

    if DataStructures.isfull(state_list)
        agent.hidden_state_init =
            get_next_hidden_state!(
                model,
                agent.hidden_state_init,
                state_list[1])
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

"""
    update!(agent::AbstractERAgent{<:ControlUpdate}, rng)

Update the parameters of the model.
"""
function update!(agent::AbstractERAgent{<:ControlUpdate}, rng)

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

    # if get_hidden_state_replay_strategy(agent) #agent.hs_strategy
    modify_hs_in_er!(
        get_hs_replay_strategy(agent),
        agent.replay,
        agent.model,
        exp,
        exp_idx,
        agent.hs_tr_init,
        us.grads,
        agent.opt,
        get_device(agent))
    # end
    
    us
end

"""
    update!(agent::AbstractERAgent{<:PredictionUpdate}, rng)

Update the parameters of the model.
"""
function update!(agent::AbstractERAgent{<:PredictionUpdate}, rng)

    τ = agent.τ
    batch_size = agent.batch_size
    
    exp_idx, exp = sample(rng, agent.replay, batch_size, τ)

    # Make batch.

    params = get_information_from_experience(agent, exp)

    hs = ActionRNNs.get_hs_from_experience(agent.model, exp)
    
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

    
        modify_hs_in_er!(
            get_hs_replay_strategy(agent),
            agent.replay,
            agent.model,
            exp,
            exp_idx,
            agent.hs_tr_init,
            us.grads,
            agent.opt)
    
    us 

end

"""
    update_target_network!

Update the target network.
"""
update_target_network!(agent::AbstractERAgent) = begin
    update_target_network!(agent.model, agent.target_network)
end

update_target_network!(::AbstractERAgent{LU, ER, Nothing}) where {LU, ER} = nothing
