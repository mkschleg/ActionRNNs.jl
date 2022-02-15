
using Random
import MinimalRLCore

# import MinimalRLCore.reset!, MinimalRLCore.environment_step!, MinimalRLCore.get_reward

"""
     LinkedChains

Many chains linked together with a special state (link=0, chain=0) and reward is positive only in this state.



"""

module LinkedChainsConst

const LINKING_STATE = (chain=0, link=0)
const STEP_REW = -1.0f0
const LS_REW = 0.0f0

end

"""
    LinkedChains

termmode:
- CONT: No termination
- TERM: Terminate after chain
dynmode: 
- STRAIGHT: high Negative reward on wrong actions, but still progress through chain
- JUMP: Jump to different chain on wrong action
- STUCK: Don't progress on wrong action
- JUMPSTUCK: Get "lost" with wrong actions, still being implemented.
"""
mutable struct LinkedChainsV2{N, TERMMODE, DYNMODE} <: AbstractEnvironment
    chain_sizes::NTuple{N, Int}
    state::NamedTuple{(:chain, :link), Tuple{Int, Int}}
    time_to_fork::Int
    start::Bool
    cur_rew::Float32
    function LinkedChainsV2(;time_to_fork, sizes, termmode=:CONT, dynmode=:STRAIGHT) where MODE
        if time_to_fork != 0 
            @error "time_to_fork not supported yet."
        end
        new{length(sizes), Val(termmode), Val(dynmode)}(
            ntuple((i) -> sizes[i], Val(length(sizes))),
            (chain=0, link=0),
            time_to_fork,
            true,
            0.0f0)
    end
end

LinkedChains = LinkedChainsV2
const LinkedChainsTERM{N, DM} = LinkedChains{N, Val{:TERM}, DM}
const LinkedChainsCONT{N, DM} = LinkedChains{N, Val{:CONT}, DM}

get_dyn_mode(::LinkedChains{N, TERMMODE, DYNMODE}) where {N, TERMMODE, DYNMODE} = DYNMODE
get_term_mode(::LinkedChains{N, TERMMODE, DYNMODE}) where {N, TERMMODE, DYNMODE} = TERMMODE

MinimalRLCore.reset!(env::LinkedChains; kwargs...) =
    MinimalRLCore.reset!(env::LinkedChains, Random.GLOBAL_RNG; kwargs...)

function MinimalRLCore.reset!(env::LinkedChains, rng; kwargs...)
    env.state = (chain=0, link=0) # special "connecting" state
    env.start = true
end

MinimalRLCore.get_actions(env::LinkedChains{N}) where N = Tuple(1:N)

function MinimalRLCore.environment_step!(
    env::LinkedChains,
    action::Int,
    # rng;
    args...; kwargs...)

    LCC = LinkedChainsConst

    new_state, new_rew = lc_get_next_state(env, action)

    if new_state.link > env.chain_sizes[new_state.chain]
        new_state = LCC.LINKING_STATE
        new_rew = LCC.LS_REW
    end

    env.state = new_state
    env.cur_rew = new_rew
    env.start = false
end

lc_get_next_state(env::LinkedChains, action) =
    lc_get_next_state(get_dyn_mode(env), env.state, action)

function lc_get_next_state(::Val{:STRAIGHT}, state, action)
    LCC = LinkedChainsConst
    new_state = if state == LCC.LINKING_STATE
        (chain=action, link=1), LCC.STEP_REW
    elseif action == state.chain
        (chain=state.chain, link=state.link+1), LCC.STEP_REW
    else
        (chain=state.chain, link=state.link+1), LCC.STEP_REW * 3
    end
end

function lc_get_next_state(::Val{:JUMP}, state, action)
    LCC = LinkedChainsConst
    new_state = if state == LCC.LINKING_STATE
        (chain=action, link=1), LCC.STEP_REW
    elseif action == state.chain
        (chain=state.chain, link=state.link+1), LCC.STEP_REW
    else
        (chain=action, link=state.link), LCC.STEP_REW
    end
end

#= JUMPSTUCK. Still working through...
function lc_get_next_state(::Val{:JUMPSTUCK}, state, action)
    LCC = LinkedChainsConst
    new_state = if state == LCC.LINKING_STATE
        (chain=action, link=1), LCC.STEP_REW
    elseif action == env.state.chain
        if state.chain == -1
            (chain=state.chain, link=state.link), LCC.STEP_REW
        elseif state.chain < -1
            (chain=state.chain + 1, link=state.link), LCC.STEP_REW
        else
            (chain=state.chain, link=state.link + 1), LCC.STEP_REW
        end
    else
        if state.chain 
        (chain=-1, link=state.link), LCC.STEP_REW
    end
end
=#
    
function lc_get_next_state(::Val{:STUCK}, state, action)
    LCC = LinkedChainsConst
    new_state = if state == LCC.LINKING_STATE
        (chain=action, link=1), LCC.STEP_REW
    elseif action == env.state.chain
        (chain=state.chain, link=state.link+1), LCC.STEP_REW
    else
        state, LCC.STEP_REW
    end
end


MinimalRLCore.get_reward(env::LinkedChains) = begin
    LCC = LinkedChainsConst
    if env.state == LCC.LINKING_STATE
        LCC.LS_REW
    else
        LCC.STEP_REW
    end
end # = 0 # -> get the reward of the environment


obs_size(env::LinkedChains) = 2

function MinimalRLCore.get_state(env::LinkedChains) # -> get state of agent
    LCC = LinkedChainsConst
    if env.state == LCC.LINKING_STATE
        Float32.([0, 1])
    else
        Float32.([1, 0])
    end
end

MinimalRLCore.is_terminal(env::LinkedChains) = lc_is_term(get_term_mode(env), env)

function lc_is_term(::Val{:CONT}, env::LinkedChains) # -> determines if the agent_state is terminal
    return false
end

function lc_is_term(::Val{:TERM}, env::LinkedChains) # -> determines if the agent_state is terminal
    env.state == LinkedChainsConst.LINKING_STATE && !env.start
end

function Base.show(io::IO, env::LinkedChains)
    padding = "             "
    print(io, "LinkedChains(",
          "chain_sizes: ", env.chain_sizes, ",\n",
          padding, "state: ", env.state, ")")
end
