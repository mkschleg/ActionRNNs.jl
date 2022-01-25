
using Random
import MinimalRLCore: MinimalRLCore, AbstractEnvironment

module Torus2dConst

const NORTH = 1
const EAST = 2
const SOUTH = 3
const WEST = 4

const STEP_REW = -0.f0
const GOOD_REW = 4f0
const BAD_REW = -100f0

end



mutable struct Torus2dv3 <: AbstractEnvironment
    size::Tuple{Int, Int}
    state::NamedTuple{(:x, :y), Tuple{Int64, Int64}}
    goal_st::NamedTuple{(:x, :y), Tuple{Int64, Int64}}
    goal_idx::Int
    anchors::Matrix{Int}
    fix_goal::Bool
    po::Bool
    non_euclidean::Bool
end

Torus2d = Torus2dv3

function Torus2dv3(width,
                   height,
                   anchors,
                   rng=Random.GLOBAL_RNG;
                   po=true,
                   non_euclidean=false,
                   fix_goal=false,
                   goal_idx=1)
    
    sze = (width, height)
    goal_state = get_goal_state(sze, goal_idx)
    anchor_matrix = create_anchors(sze, anchors, rng)
    env = Torus2dv3(sze,
                  (x=1, y=1), # initial state for un-reset agent.
                  goal_state, # goal_state
                  goal_idx, # goal_idx
                  anchor_matrix,
                  fix_goal,
                  po,
                  non_euclidean)
end


function create_anchors(sze, anchors::Int, rng)
    idxs = randperm(rng, *(sze...))[1:anchors]
    anchor_matrix = zeros(Int, sze...)
    anchor_matrix[idxs] .= 1:anchors
    anchor_matrix
end

function create_anchors(sze, anchors::Vector{T}, args...) where T <: Union{Tuple{Int, Int}, Vector{Int}}
    anchor_matrix = zeros(Int, sze...)
    for (idx, a) in enumerate(anchors)
        anchor_matrix[a...] = idx
    end
    anchor_matrix
end

function create_anchors(sze, anchors::Vector{CartesianIndex{2}}, args...)
    anchor_matrix = zeros(Int, sze...)
    anchor_matrix[anchors] .= 1:length(anchors)
    anchor_matrix
end

function get_anchor(env::Torus2d, state=env.state)
    env.anchors[state...]
end

get_goal_state(env::Torus2d, goal_idx) = get_goal_state(size(env), goal_idx)
function get_goal_state((width, height), goal_idx)
    if goal_idx == 1
        (x=2, y=height-1)
    elseif goal_idx == 2
        (x=width-1, y=height-1)
    elseif goal_idx == 3
        (x=width-1, y=2)
    else
        (x=2, y=2)
    end
end

obs_size(env::Torus2d) = maximum(env.anchors)

Base.size(env::Torus2d) = env.size


function MinimalRLCore.reset!(env::Torus2d, rng::Random.AbstractRNG=Random.GLOBAL_RNG)
    env.state = (x = rand(rng, 1:env.size[1]), y = rand(rng, 1:env.size[2]))
    if !env.fix_goal
        env.goal_idx = rand(rng, 1:4)
        env.goal_st = get_goal_state(env, env.goal_idx)
    end
end

function MinimalRLCore.reset!(env::Torus2d, agent::Tuple{Int, Int}, goal::Int)
    env.state = (x = agent[1], y = agent[2])
    env.goal_st = get_goal_state(env, goal)
    env.goal_idx = goal
end

function MinimalRLCore.reset!(env::Torus2d, ax::Int, ay::Int, g_idx::Int)
    env.state = (x = ax, y = ay)
    env.goal_st = get_goal_state(env, goal)
    env.goal_idx = g_idx
end

MinimalRLCore.get_actions(env::Torus2d) = [Torus2dConst.NORTH, Torus2dConst.EAST, Torus2dConst.SOUTH, Torus2dConst.WEST]

function MinimalRLCore.environment_step!(env::Torus2d, action::Int)

    @assert action ∈ get_actions(env)
    
    T2DC = Torus2dConst
    
    state = env.state
    width, height = size(env)
    odd_zone_width = width ÷ 3
    odd_zone_height = height ÷ 3

    # go north
    new_state = if action == T2DC.NORTH
        if state.y == height
            if env.non_euclidean && state.x <= odd_zone_width
                (width - (state.x - 1), state.y)
            else
                (state.x, state.y)
            end
        else
            (state.x, state.y + 1)
        end
    # go south
    elseif action == T2DC.SOUTH
        if state.y == 1
            if env.non_euclidean && state.x > width - odd_zone_width
                (width - state.x + 1, 1)
            elseif env.non_euclidean && !(state.x <= odd_zone_width)
                (state.x, height)
            else
                (state.x, state.y)
            end
        else
            (state.x, state.y - 1)
        end
    # go east
    elseif action == T2DC.EAST
        if state.x == width
            if env.non_euclidean && state.y > height - odd_zone_height
                (width, height - state.y + 1)
            elseif env.non_euclidean && !(state.y <= odd_zone_height)
                (1, state.y)
            else
                (state.x, state.y)
            end
        else
            (state.x + 1, state.y)
        end
    # go west
    elseif action == T2DC.WEST
        if state.x == 1
            if env.non_euclidean && state.y <= odd_zone_height
                (1, height - state.y + 1)
            else 
                (state.x, state.y)
            end
        else
            (state.x - 1, state.y)
        end
    end

    env.state = (x=new_state[1], y=new_state[2])
    
end

function MinimalRLCore.get_reward(env::Torus2d) # -> get the reward of the environment
    if env.state == env.goal_st
        Torus2dConst.GOOD_REW
    else
        Torus2dConst.STEP_REW
    end
end

function MinimalRLCore.is_terminal(env::Torus2d) # -> determines if the agent_state is terminal
    env.state == env.goal_st
end

function MinimalRLCore.get_state(env::Torus2d) # -> get state of agent
    if env.po
        return partially_observable_state(env)
    else
        return fully_observable_state(env)
    end
end

function fully_observable_state(env::Torus2d)
    st = zeros(Float32, 3)
    st[1] = env.state.x
    st[2] = env.state.y
    st[3] = env.goal_idx
    st
end

function partially_observable_state(env::Torus2d)
    obs = zeros(Float32, obs_size(env)+1+4)
    obs_idx = 1 + get_anchor(env)
    obs[obs_idx] = 1
    obs[end-env.goal_idx] = 1
    return obs
end



function Base.show(io::IO, env::Torus2d)
    padding = "        "
    print(io,
          "Torus2d(",
          "size: ", size(env), ",\n",
          padding*"agent_state: ", env.state, ",\n",
          padding*"goal_idx: ", env.goal_idx, ",\n",
          padding*"num_anchors: ", maximum(env.anchors),)
          print(io, ")")
end

import RecipesBase
using Colors
RecipesBase.@recipe function f(env::Torus2d)

    ticks := nothing
    foreground_color_border := nothing
    grid := false
    legend := false
    aspect_ratio := 1
    xaxis := false
    yaxis := false
    yflip := false

    CELL_SIZE=20#size(env)
    BG = Colors.RGB(1.0, 1.0, 1.0)
    BORDER = Colors.RGB(0.0, 0.0, 0.0)
    WALL = Colors.RGB(0.3, 0.3, 0.3)
    AC = Colors.RGB(0.69921875, 0.10546875, 0.10546875)
    GOAL = Colors.RGB(0.796875, 0.984375, 0.76953125)
    ANCHOR_GOAL = Colors.RGB(31/255, 103/255, 38/255)
    ANCHOR_REGRET = Colors.RGB(115/255, 19/255, 67/255)

    cell = fill(BG, CELL_SIZE, CELL_SIZE)
    cell[1, :] .= BORDER
    cell[end, :] .= BORDER
    cell[:, 1] .= BORDER
    cell[:, end] .= BORDER

    s_x = size(env)[1] # width
    s_y = size(env)[2] # height

    anchor_goal = copy(cell)

    for i in 3:4
        anchor_goal[i, i:end-i+1] .= ANCHOR_GOAL
        anchor_goal[end-i+1, i:end-i+1] .= ANCHOR_GOAL
        anchor_goal[i:end-i+1, i] .= ANCHOR_GOAL
        anchor_goal[i:end-i+1, end-i+1] .= ANCHOR_GOAL
    end

    anchor_regret = copy(cell)

    for i in 3:4
        anchor_regret[i, i:end-i+1] .= ANCHOR_REGRET
        anchor_regret[end-i+1, i:end-i+1] .= ANCHOR_REGRET
        anchor_regret[i:end-i+1, i] .= ANCHOR_REGRET
        anchor_regret[i:end-i+1, end-i+1] .= ANCHOR_REGRET
    end

    screen = fill(BG, (s_y + 2)*CELL_SIZE, (s_x + 2)*CELL_SIZE)

    screen[:, 1:CELL_SIZE] .= WALL
    screen[1:CELL_SIZE, :] .= WALL
    screen[end-(CELL_SIZE-1):end, :] .= WALL
    screen[:, end-(CELL_SIZE-1):end] .= WALL

    state = env.state

    goal = fill(GOAL, CELL_SIZE, CELL_SIZE)
    goal[1, :] .= BORDER
    goal[end, :] .= BORDER
    goal[:, 1] .= BORDER
    goal[:, end] .= BORDER
    

    for i ∈ 1:s_x
        for j ∈ 1:s_y
            sqr_i = ((i)*CELL_SIZE + 1):((i+1)*CELL_SIZE)
            sqr_j = ((j)*CELL_SIZE + 1):((j+1)*CELL_SIZE)

            screen[sqr_j, sqr_i] .= (env.goal_st.x == i && env.goal_st.y == j) ? goal : cell
            
            if get_anchor(env, (i, j)) != 0
                # screen[sqr_j, sqr_i] .= anchor_regret #get_anchor(env, (i, j)) % 2 == env.anchor_rew ? anchor_goal : anchor_regret
                ac = @view screen[sqr_j, sqr_i]
                for i in 3:4
                    ac[i, i:end-i+1] .= ANCHOR_REGRET
                    ac[end-i+1, i:end-i+1] .= ANCHOR_REGRET
                    ac[i:end-i+1, i] .= ANCHOR_REGRET
                    ac[i:end-i+1, end-i+1] .= ANCHOR_REGRET
                end
                
            end
            
            if state.x == i && state.y == j
                v = @view screen[sqr_j, sqr_i]
                # v .= cell
                v[Int(CELL_SIZE/2)-4:Int(CELL_SIZE/2)+5, Int(CELL_SIZE/2)-4:Int(CELL_SIZE/2)+5] .= AC
            end
        end
    end
    # screen[end:-1:1,:]
    screen
    
end
