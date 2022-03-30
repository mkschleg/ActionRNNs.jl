
using Random
import MinimalRLCore

# module TMazeConst
# # Actions

# const LEFT  = 1
# const RIGHT = 2
# const UP = 3
# const DOWN = 4

# const GOOD_REW = 4.0f0
# const BAD_REW = -1.0f0
# end

const TMC = TMazeConst

"""
 TMaze           _
     _ _ _ _ _ _|_|
    |_|_|_|_|_|_|_|
                |_|
"""


mutable struct VariableTMaze{D} <: AbstractEnvironment
    size::Int
    goal_dir::Int
    size_range::D
    state::NamedTuple{(:x, :y), Tuple{Int, Int}}
end

VariableTMaze(size) = VariableTMaze(size, TMC.UP, 2:size, (x=1, y=0))
VariableTMaze(parsed::Dict) = VariableTMaze(parsed["size"])

Base.size(env::VariableTMaze) = env.size

function MinimalRLCore.reset!(env::VariableTMaze, rng::AbstractRNG=Random.GLOBAL_RNG)
    env.goal_dir = rand(rng, [TMC.UP, TMC.DOWN])
    env.size = rand(rng, env.size_range)
    env.state = (x=1, y=0)
end

MinimalRLCore.get_actions(env::VariableTMaze) = [TMC.LEFT, TMC.RIGHT, TMC.UP, TMC.DOWN]
get_num_features(env::VariableTMaze) = 3

function MinimalRLCore.environment_step!(env::VariableTMaze, action::Int, rng=Random.GLOBAL_RNG)
    
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

MinimalRLCore.get_reward(env::VariableTMaze) = begin
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
    

function MinimalRLCore.get_state(env::VariableTMaze) # -> get state of agent
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


MinimalRLCore.is_terminal(env::VariableTMaze) = env.state.y != 0

function to_string(env::VariableTMaze)
    char_strg = fill(' ', 3, env.size)
    char_strg[:, end] .= Char(0x25A2)
    char_strg[2, :] .= Char(0x25A2)
    if env.goal_dir == TMC.UP
        char_strg[1, end] = Char(0x25C8)
    else
        char_strg[end, end] = Char(0x25C8)
    end
    char_strg[(-env.state.y)+2, env.state.x] = Char(0x25A3)
    str = ""
    for i ∈ 1:3
        str *= join(char_strg[i, :], " ")
        str *= "\n"
    end
    str
end

function Base.show(io::IO, env::VariableTMaze)
    print(to_string(env))
    println("(size: $(env.size), goal: $(env.goal_dir), state: $(env.state)")
end

using RecipesBase

@recipe function f(env::VariableTMaze)
    ticks := nothing
    foreground_color_border := nothing
    grid := false
    legend := false
    aspect_ratio := 1
    xaxis := false
    yaxis := false

    PP = PlotParams

    screen = fill(PP.WALL, 5, env.size+2)
    screen[2:4, end-1] .= PP.BG
    screen[2+1, 2:end-1] .= PP.BG
    if env.goal_dir == TMC.UP
        screen[1+1, end-1] = PP.GOAL
    else
        screen[end-1, end-1] = PP.GOAL
    end
    screen[(-env.state.y)+2+1, env.state.x+1] = PP.AC

    screen
end
