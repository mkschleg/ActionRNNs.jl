
using Random
import MinimalRLCore

import Reproduce

module DirectionalTMazeConst
# Actions

const LEFT = 1
const FORWARD = 2
const RIGHT = 3


const NORTH = 1
const EAST = 2
const SOUTH = 3
const WEST = 4

const GOOD_REW = 4.0f0
const BAD_REW = -1.0f0
const STEP_REW = -0.1f0

const GOAL_OBS = [1, 1, 0]
const WALL_OBS = [0, 1, 0]
const EMPTY_OBS = [0, 0, 1]

end

const DTMC = DirectionalTMazeConst

"""
    DirectionalTMaze

Similar to [`ActionRNNs.TMaze`](@ref) but with a directional componenet overlayed ontop. This also changes to 
observation structure, where the agent must know what direction it is facing to get information
about which goal is the good goal.
"""
mutable struct DirectionalTMaze <: AbstractEnvironment
    size::Int
    goal_dir::Int
    state::NamedTuple{(:x, :y, :dir), Tuple{Int, Int, Int}}
end

DirectionalTMaze(size) = DirectionalTMaze(size, DTMC.NORTH, (x=1, y=0, dir=1))
DirectionalTMaze(parsed::Dict) = DirectionalTMaze(parsed["size"])

Base.size(env::DirectionalTMaze) = env.size

function MinimalRLCore.reset!(env::DirectionalTMaze, rng::AbstractRNG=Random.GLOBAL_RNG)
    env.goal_dir = rand(rng, [DTMC.NORTH, DTMC.SOUTH])
    env.state = (x=1, y=0, dir=rand(rng, 1:4))
    
    @data Env state=env.state
    @data Env goal_dir=env.goal_dir
    @data Env reset=true
end

MinimalRLCore.get_actions(env::DirectionalTMaze) = [DTMC.LEFT, DTMC.FORWARD, DTMC.RIGHT]
get_num_features(env::DirectionalTMaze) = 3

function MinimalRLCore.environment_step!(env::DirectionalTMaze, action::Int, rng=Random.GLOBAL_RNG)

    @assert action ∈ get_actions(env)
    
    if DTMC.FORWARD == action
        if DTMC.NORTH == env.state.dir
            if env.state.x == size(env)
                env.state = (x=env.state.x, y=env.state.y + 1, dir=env.state.dir)
            end
        elseif DTMC.SOUTH == env.state.dir
            if env.state.x == size(env)
                env.state = (x=env.state.x, y=env.state.y - 1, dir=env.state.dir)
            end
        elseif DTMC.EAST == env.state.dir
            if env.state.x != size(env)
                env.state = (x=env.state.x + 1, y=env.state.y, dir=env.state.dir)
            end
        elseif DTMC.WEST == env.state.dir
            if env.state.x != 1
                env.state = (x=env.state.x - 1, y=env.state.y, dir=env.state.dir)
            end
        end
    elseif DTMC.LEFT == action
        new_dir = env.state.dir - 1
        if new_dir == 0
            new_dir = 4
        end
        env.state = (x=env.state.x, y=env.state.y, dir=new_dir)
    else # Turn RIGHT Action
        new_dir = env.state.dir + 1
        if new_dir == 5
            new_dir = 1
        end
        env.state = (x=env.state.x, y=env.state.y, dir=new_dir)
    end

    @data Env state=env.state
    @data Env reset=false
end

MinimalRLCore.get_reward(env::DirectionalTMaze) = begin
    if env.state.x == env.size
        if env.state.y == 0
            DTMC.STEP_REW
        elseif env.state.y == 1
            env.goal_dir == DTMC.NORTH ? DTMC.GOOD_REW : DTMC.BAD_REW
        elseif env.state.y == -1
            env.goal_dir == DTMC.SOUTH ? DTMC.GOOD_REW : DTMC.BAD_REW
        end
    else
        DTMC.STEP_REW
    end
end
    

function MinimalRLCore.get_state(env::DirectionalTMaze) # -> get state of agent

    state = env.state
    env_size = env.size
    if state.x == 1
        if (env.state.dir == DTMC.NORTH && env.goal_dir == DTMC.NORTH) ||
            (env.state.dir == DTMC.SOUTH && env.goal_dir == DTMC.SOUTH)
            DTMC.GOAL_OBS
        elseif env.state.dir == DTMC.EAST
            DTMC.EMPTY_OBS
        else
            DTMC.WALL_OBS
        end
    elseif state.x == env_size
        if env.state.dir == DTMC.EAST
            DTMC.WALL_OBS
        else
            DTMC.EMPTY_OBS
        end
    else
        if env.state.dir ∈ [DTMC.NORTH, DTMC.SOUTH]
            DTMC.WALL_OBS
        else
            DTMC.EMPTY_OBS
        end
    end
end


MinimalRLCore.is_terminal(env::DirectionalTMaze) = env.state.y != 0

function to_string(env::DirectionalTMaze)
    dir_char_dict = Dict(DTMC.NORTH => Char(0x25B2),
                         DTMC.EAST => Char(0x25B6),
                         DTMC.SOUTH => Char(0x25BC),
                         DTMC.WEST => Char(0x25C0))
    char_strg = fill(' ', 3, env.size)
    char_strg[:, end] .= Char(0x25A2)
    char_strg[2, :] .= Char(0x25A2)
    if env.goal_dir == DTMC.NORTH
        char_strg[1, end] = Char(0x25C8)
    else
        char_strg[end, end] = Char(0x25C8)
    end
    char_strg[(-env.state.y)+2, env.state.x] = dir_char_dict[env.state.dir]
    str = ""
    for i ∈ 1:3
        str *= join(char_strg[i, :], " ")
        str *= "\n"
    end
    str
end

function Base.show(io::IO, env::DirectionalTMaze)
    print(to_string(env))
    println("(size: $(env.size), goal: $(env.goal_dir), state: $(env.state)")
end

using RecipesBase

module DirectionalTMazePlotParams

using Colors

const SIZE = 10
const BG = Colors.RGB(1.0, 1.0, 1.0)
const WALL = Colors.RGB(0.3, 0.3, 0.3)
const BORDER = Colors.RGB(0.0, 0.0, 0.0)
const AC = Colors.RGB(0.69921875, 0.10546875, 0.10546875)
const GOAL = Colors.RGB(0.796875, 0.984375, 0.76953125)
const AGENT = [BORDER BORDER BORDER BORDER BORDER BORDER BORDER BORDER BORDER BORDER;
               BORDER BG BG BG BG BG BG BG BG BORDER;
               BORDER BG BG AC AC BG BG BG BG BORDER;
               BORDER BG BG AC AC AC AC BG BG BORDER;
               BORDER BG BG AC AC AC AC AC BG BORDER;
               BORDER BG BG AC AC AC AC AC BG BORDER;
               BORDER BG BG AC AC AC AC BG BG BORDER;
               BORDER BG BG AC AC BG BG BG BG BORDER;
               BORDER BG BG BG BG BG BG BG BG BORDER;
               BORDER BORDER BORDER BORDER BORDER BORDER BORDER BORDER BORDER BORDER]

end

@recipe function f(env::DirectionalTMaze)
    ticks := nothing
    foreground_color_border := nothing
    grid := false
    legend := false
    aspect_ratio := 1
    xaxis := false
    yaxis := false

    PP = DirectionalTMazePlotParams

    cell = fill(PP.BG, PP.SIZE, PP.SIZE)
    cell[1:end, 1] .= PP.BORDER
    cell[1:end, end] .= PP.BORDER
    cell[1, 1:end] .= PP.BORDER
    cell[end, 1:end] .= PP.BORDER 

    screen = fill(PP.WALL, PP.SIZE*5, PP.SIZE*(env.size+2))

    for i ∈ 1:size(env)
        screen[(PP.SIZE*(2)+1):PP.SIZE*3, ((i-1+1)*PP.SIZE+1:(i+1)*PP.SIZE)] .= cell
    end

    screen[PP.SIZE*(3)+1:PP.SIZE*4, (end-2*PP.SIZE+1):(end-PP.SIZE)] .= cell
    screen[PP.SIZE*(1)+1:PP.SIZE*2, (end-2*PP.SIZE+1):(end-PP.SIZE)] .= cell
    
    if env.goal_dir == DTMC.NORTH
        v = @view screen[PP.SIZE*(1)+1:PP.SIZE*2, (end-2*PP.SIZE+1):(end-PP.SIZE)]
        v[v .== PP.BG] .= PP.GOAL
    else
        v = @view screen[PP.SIZE*(3)+1:PP.SIZE*4, (end-2*PP.SIZE+1):(end-PP.SIZE)]
        v[v .== PP.BG] .= PP.GOAL
    end

    sqr_i = ((-env.state.y+2)*PP.SIZE + 1):((-env.state.y+2+1)*PP.SIZE)
    sqr_j = ((env.state.x)*PP.SIZE + 1):((env.state.x+1)*PP.SIZE)
    screen[sqr_i, sqr_j] .= if env.state.dir == DTMC.NORTH
        (PP.AGENT[:, end:-1:1])'
    elseif env.state.dir == DTMC.EAST
        PP.AGENT
    elseif env.state.dir == DTMC.SOUTH
        PP.AGENT'
    elseif env.state.dir == DTMC.WEST
        (PP.AGENT[:, end:-1:1])
    end
    
    screen
end
