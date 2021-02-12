
using Random
import MinimalRLCore

import Reproduce

module TMazeConst
# Actions

const LEFT  = 1
const RIGHT = 2
const UP = 3
const DOWN = 4

const GOOD_REW = 4.0f0
const BAD_REW = -1.0f0
end

const TMC = TMazeConst

"""
 TMaze           _
     _ _ _ _ _ _|_|
    |_|_|_|_|_|_|_|
                |_|
"""


mutable struct TMaze <: AbstractEnvironment
    size::Int
    goal_dir::Int
    state::NamedTuple{(:x, :y), Tuple{Int, Int}}
end

TMaze(size) = TMaze(size, TMC.UP, (x=1, y=0))
TMaze(parsed::Dict) = TMaze(parsed["size"])

Base.size(env::TMaze) = env.size

function MinimalRLCore.reset!(env::TMaze, rng::AbstractRNG=Random.GLOBAL_RNG)
    env.goal_dir = rand(rng, [TMC.UP, TMC.DOWN])
    env.state = (x=1, y=0)
end

MinimalRLCore.get_actions(env::TMaze) = [TMC.LEFT, TMC.RIGHT, TMC.UP, TMC.DOWN]
get_num_features(env::TMaze) = 3

function MinimalRLCore.environment_step!(env::TMaze, action::Int, rng=Random.GLOBAL_RNG)
    
    if env.state.x == size(env)
        if action == TMC.UP
            env.state = (x=env.state.x, y=env.state.y + 1)
        elseif action == TMC.DOWN
            env.state = (x=env.state.x, y=env.state.y - 1)
        elseif action == TMC.RIGHT
            env.state = (x=clamp(env.state.x + 1, 1, size(env)), y=env.state.y)
        elseif action == TMC.LEFT
            env.state = (x=clamp(env.state.x - 1, 1, size(env)), y=env.state.y)
        end
    else
        if action == TMC.RIGHT
            env.state = (x=clamp(env.state.x + 1, 1, size(env)), y=env.state.y)
        elseif action == TMC.LEFT
            env.state = (x=clamp(env.state.x - 1, 1, size(env)), y=env.state.y)
        end        
    end
end

MinimalRLCore.get_reward(env::TMaze) = begin
    if env.state.x == env.size
        if env.state.y == 0
            -0.1f0
        elseif env.state.y == 1
            env.goal_dir == TMC.UP ? TMC.GOOD_REW : TMC.BAD_REW
        elseif env.state.y == -1
            env.goal_dir == TMC.DOWN ? TMC.GOOD_REW : TMC.BAD_REW
        end
    else
        -0.1f0
    end
end
    


function MinimalRLCore.get_state(env::TMaze) # -> get state of agent
    state = env.state
    env_size = env.size
    if state.x == 1
        env.goal_dir == TMC.UP ? [1, 1, 0] : [0, 1, 1] 
    elseif state.x == env_size
        [0, 1, 0]
    else
        [1, 0, 1]
    end
end


MinimalRLCore.is_terminal(env::TMaze) = env.state.y != 0


function to_string(env::TMaze)
    char_strg = fill(' ', 3, env.size)
    char_strg[:, 1] .= Char(0x25A1)
    char_strg[2, :] .= Char(0x25A1)
    char_strg[env.state.x+1, env.state.y+1] = 'a'
    str = ""
    for i ∈ 1:3
        str *= *(char_strg[i, :]...)
        str *= "\n"
    end
    str
end

function Base.show(io::IO, env::TMaze)
    println(to_string(env))
end
