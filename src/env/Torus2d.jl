
using Random
import MinimalRLCore: MinimalRLCore, AbstractEnvironment

module Torus2dConst

const NORTH = 1
const EAST = 2
const SOUTH = 3
const WEST = 4

end

mutable struct Torus2d <: AbstractEnvironment
    size::Tuple{Int, Int}
    state::NamedTuple{(:x, :y), Tuple{Int64, Int64}}
    po::Bool
end

function Torus2d(width, height; po=true)
    @assert isodd(width) && isodd(height)
    sze = (width, height)
    Torus2d(sze,
            (x = (width ÷ 2) + 1, y = (height ÷ 2) + 1),
            po)
end

find_middle(env::Torus2d) = (x = (width ÷ 2) + 1, y = (height ÷ 2) + 1)

Base.size(env::Torus2d) = env.size


function MinimalRLCore.reset!(env::Torus2d)
    env.state = (x = rand(1:env.size[1]), y = rand(1:env.size[2]))
end

function MinimalRLCore.reset!(env::Torus2d, state::Tuple{Int, Int})
    env.state = (x = state[1], y = state[2])
end

function MinimalRLCore.reset!(env::Torus2d, x::Int, y::Int)
    env.state = (x = x, y = y)
end

MinimalRLCore.get_actions(env::Torus2d) = [Torus2dConst.NORTH, Torus2dConst.EAST, Torus2dConst.SOUTH, Torus2dConst.WEST]

function MinimalRLCore.environment_step!(env::Torus2d, action::Int)

    T2DC = Torus2dConst
    
    state = env.state
    width, height = size(env)

    # go north
    new_state = if action == T2DC.NORTH
        if state.y == height
            (state.x, 1)
        else
            (state.x, state.y + 1)
        end
    # go south
    elseif action == T2DC.SOUTH
        if state.y == 1
            (state.x, 1)
        else
            (state.x, state.y - 1)
        end
    # go east
    elseif action == T2DC.EAST
        if state.x == width
            (1, state.y)
        else
            (state.x + 1, state.y)
        end
    # go west
    elseif action == T2DC.WEST
        if state.x == 1
            (1, state.y)
        else
            (state.x - 1, state.y)
        end
    else
        throw("Action invalid")
    end

    env.state = new_state
    
end

function MinimalRLCore.get_reward(env::Torus2d) # -> get the reward of the environment
    return 0.0f0
end

function MinimalRLCore.get_state(env::Torus2d) # -> get state of agent
    if env.po
        return partially_observable_state(env)
    else
        return fully_observable_state(env)
    end
end

function fully_observable_state(env::Torus2d)
    collect(env.agent_state)
end

function partially_observable_state(env::Torus2d)
    if env.state == find_middle(env)
        [1.0f0]
    else
        [0.0f0]
    end
    return state
end

function MinimalRLCore.is_terminal(env::RingWorld) # -> determines if the agent_state is terminal
    return false
end

