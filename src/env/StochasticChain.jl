
using Random
import MinimalRLCore

# import MinimalRLCore.reset!, MinimalRLCore.environment_step!, MinimalRLCore.get_reward

"""
 StochasticChain

   1 -> 0 -> 0 -> ... -> 0 -|
   ^------------------------|

chain_length: size of cycle
actions: Progress

"""

mutable struct StochasticChain <: AbstractEnvironment
    agent_state::Int64
    p1::Float64
    P2::Float64
    StochasticChain(p1, p2) = new(1, p1, p2)
end

function MinimalRLCore.reset!(env::StochasticChain, rng::AbstractRNG; kwargs...)
    env.agent_state = 1
end

MinimalRLCore.get_actions(env::StochasticChain) = Set(1:2)

function MinimalRLCore.environment_step!(env::StochasticChain, action::Int64, rng; kwargs...)
    @assert action ∈ MinimalRLCore.get_actions(env)

    p = rand(rng)
    
    if action == 1
        if p > env.p1
            env.agent_state = env.agent_state
        else
            env.agent_state = 2
        end
    else
        if p > env.p2
            env.agent_state = 2
        else
            env.agent_state = env.agent_state
        end
    end
end



MinimalRLCore.get_reward(env::StochasticChain) = env.agent_state == 2 ? 1 : 0 # -> get the reward of the environment

function MinimalRLCore.get_state(env::StochasticChain) # -> get state of agent
    env.agent_state == 1 ? [1.0f0] : [0.0f0]
end

MinimalRLCore.is_terminal(env::StochasticChain) = env.agent_state == 2
