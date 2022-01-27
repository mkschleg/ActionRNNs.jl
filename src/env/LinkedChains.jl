
using Random
import MinimalRLCore

# import MinimalRLCore.reset!, MinimalRLCore.environment_step!, MinimalRLCore.get_reward

"""
     LinkedChains

Many chains linked together with a special state (link=0, chain=0) and reward is positive only in this state.



"""

module LinkedChainsConst

const LINKING_STATE = (link=0, chain=0)
const STEP_REW = -1.0f0
const LS_REW = 0.0f0

end



mutable struct LinkedChainsV2{N, MODE} <: AbstractEnvironment
    chain_sizes::NTuple{N, Int}
    state::NamedTuple{(:link, :chain), Tuple{Int, Int}}
    time_to_fork::Int
    function LinkedChainsV2{MODE}(time_to_fork, sizes...) where MODE
        new{length(sizes), MODE}(
            sizes,
            (link=0, chain=0),
            time_to_fork)
    end
end

LinkedChains = LinkedChainsV2
LinkedChainsTERM{N} = LinkedChainsV2{N, :TERM}
LinkedChainsCONT{N} = LinkedChainsV2{N, :CONT}


MinimalRLCore.reset!(env::LinkedChains; kwargs...) =
    MinimalRLCore.reset!(env::LinkedChains, Random.GLOBAL_RNG; kwargs...)

function MinimalRLCore.reset!(env::LinkedChains, rng; kwargs...)
    env.state = (link=0, chain=0) # special "connecting" state
    # @show "reset"

end

MinimalRLCore.get_actions(env::LinkedChains{N}) where N = Tuple(1:N)

function MinimalRLCore.environment_step!(
    env::LinkedChains,
    action::Int64,
    # rng;
    args...; kwargs...)

    LCC = LinkedChainsConst
    
    # @show "step"
    state = env.state
    new_state = if state == LCC.LINKING_STATE
        (link=action, chain=1)
    elseif action == env.state.link
        (link=state.link, chain=state.chain+1)
    else
        state
    end

    if new_state.chain > env.chain_sizes[new_state.link]
        new_state = LCC.LINKING_STATE
    end

    env.state = new_state

end


MinimalRLCore.get_reward(env::LinkedChains) = begin
    LCC = LinkedChainsConst
    if env.state == LCC.LINKING_STATE
        LCC.LS_REW
    else
        LCC.STEP_REW
    end
end# = 0 # -> get the reward of the environment


obs_size(env::LinkedChains) = 2

function MinimalRLCore.get_state(env::LinkedChains) # -> get state of agent
    LCC = LinkedChainsConst
    if env.state == LCC.LINKING_STATE
        Float32.([0, 1])
    else
        Float32.([1, 0])
    end
end

function MinimalRLCore.is_terminal(env::LinkedChains{N, :CONT}) where N # -> determines if the agent_state is terminal
    return false
end

function MinimalRLCore.is_terminal(env::LinkedChains{N, :TERM}) where N # -> determines if the agent_state is terminal
    env.state == LinkedChainsConst.LINKING_STATE
end

function Base.show(io::IO, env::LinkedChains)
    padding = "             "
    print(io, "LinkedChains(",
          "chain_sizes: ", env.chain_sizes, ",\n",
          padding, "state: ", env.state, ")")
end
