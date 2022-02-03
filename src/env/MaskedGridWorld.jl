

import Random: Random, randperm
import MinimalRLCore: MinimalRLCore, AbstractEnvironment

module MaskedGridWorldConst

const NORTH = 1
const EAST = 2
const SOUTH = 3
const WEST = 4

const STEP_REW = -0.1f0
const GOOD_REW = 4f0
const NEUTRAL_REW = 0.0f0
const BAD_REW = -100f0

end

"""
    MaskedGridWorldHelpers

Helper functions for the Masked grid world environment.
"""
module MaskedGridWorldHelpers

import Random: Random, randperm
import ..MaskedGridWorldConst: STEP_REW, GOOD_REW, NEUTRAL_REW, BAD_REW

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

function create_goals(sze, num_goals::Int, anchor_matrix, rng)
    @assert *(sze...) - sum(anchor_matrix .!= 0) > num_goals
    possible_states = CartesianIndices(sze)[anchor_matrix .== 0]
    perm = randperm(rng, length(possible_states))[1:num_goals]
    [(x=possible_states[i][1], y=possible_states[i][2], r=GOOD_REW) for i in perm]
end

function create_goals(sze, rews::Vector{<:AbstractFloat}, anchor_matrix, rng)
    @assert *(sze...) - sum(anchor_matrix .!= 0) > num_goals
    possible_states = CartesianIndices(sze)[anchor_matrix .== 0]
    perm = randperm(rng, length(possible_states))[1:num_goals]
    [(x=possible_states[i][1], y=possible_states[i][2], r=rews[idx]) for (idx, i) in enumerate(perm)]
end

function create_goals(sze, goals::Vector{NamedTuple{(:x, :y, :r), Tuple{Int, Int, Float32}}}, args...)
    goals
end

end


mutable struct MaskedGridWorld <: AbstractEnvironment
    size::Tuple{Int, Int}
    state::NamedTuple{(:x, :y), Tuple{Int64, Int64}}
    goals::Vector{NamedTuple{(:x, :y, :r), Tuple{Int, Int, Float32}}}
    anchors::Matrix{Int}
    obs_strategy::Symbol
    pacman_wrapping::Bool
end


function MaskedGridWorld(width,
                         height,
                         anchors,
                         goals_or_rews,
                         rng=Random.GLOBAL_RNG;
                         obs_strategy=:seperate,
                         pacman_wrapping=true)
    
    sze = (width, height)
    anchor_matrix = MaskedGridWorldHelpers.create_anchors(sze, anchors, rng)
    goals = MaskedGridWorldHelpers.create_goals(sze, goals_or_rews, anchor_matrix, rng)
    env = MaskedGridWorld(sze,
                          (x=1, y=1), # initial state for un-reset agent.
                          goals, # goal_state
                          anchor_matrix,
                          Symbol(obs_strategy),
                          pacman_wrapping)
end




function get_anchor(env::MaskedGridWorld, state=env.state)
    env.anchors[state...]
end

obs_size(env::MaskedGridWorld) = if env.obs_strategy == :full
    2
elseif env.obs_strategy == :seperate
    maximum(env.anchors)+1
elseif env.obs_strategy == :aliased
    2
end


Base.size(env::MaskedGridWorld) = env.size

function MinimalRLCore.reset!(env::MaskedGridWorld, rng::Random.AbstractRNG=Random.GLOBAL_RNG)
    state = (x = rand(rng, 1:env.size[1]), y = rand(rng, 1:env.size[2]))
    check_in_goal = (g) -> state.x == g.x && state.y==g.y
    while any(check_in_goal.(env.goals))
        state = (x = rand(rng, 1:env.size[1]), y = rand(rng, 1:env.size[2]))
    end
    env.state = state
end

function MinimalRLCore.reset!(env::MaskedGridWorld, agent::Tuple{Int, Int})
    env.state = (x = agent[1], y = agent[2])
end

function MinimalRLCore.reset!(env::MaskedGridWorld, ax::Int, ay::Int)
    env.state = (x = ax, y = ay)
end

MinimalRLCore.get_actions(env::MaskedGridWorld) = [MaskedGridWorldConst.NORTH, MaskedGridWorldConst.EAST, MaskedGridWorldConst.SOUTH, MaskedGridWorldConst.WEST]

function MinimalRLCore.environment_step!(env::MaskedGridWorld, action::Int, rng=Random.GLOBAL_RNG)

    @assert action ∈ get_actions(env)

    new_state = env_step_pacman_wrapping(size(env), env.state, action)
    env.state = (x=new_state[1], y=new_state[2])
end

function env_step_pacman_wrapping((width, height), state, action)
    MGWC = MaskedGridWorldConst
    if action == MGWC.NORTH
        if state.y == height
            (state.x, 1)
        else
            (state.x, state.y + 1)
        end
    # go south
    elseif action == MGWC.SOUTH
        if state.y == 1
            (state.x, height)
        else
            (state.x, state.y - 1)
        end
    # go east
    elseif action == MGWC.EAST
        if state.x == width
            (1, state.y)
        else
            (state.x + 1, state.y)
        end
    # go west
    elseif action == MGWC.WEST
        if state.x == 1
            (width, state.y)
        else
            (state.x - 1, state.y)
        end
    end
end

function env_step_no_wrapping(env::MaskedGridWorld, action::Int)
    MGWC = MaskedGridWorldConst
    if action == MGWC.NORTH
        if state.y == height
            (state.x, state.y)
        else
            (state.x, state.y + 1)
        end
    # go south
    elseif action == MGWC.SOUTH
        if state.y == 1
            (state.x, state.y)
        else
            (state.x, state.y - 1)
        end
    # go east
    elseif action == MGWC.EAST
        if state.x == width
            (state.x, state.y)
        else
            (state.x + 1, state.y)
        end
    # go west
    elseif action == MGWC.WEST
        if state.x == 1
            (state.x, state.y)
        else
            (state.x - 1, state.y)
        end
    end
end


function MinimalRLCore.get_reward(env::MaskedGridWorld) # -> get the reward of the environment
    check_in_goal = (g) -> env.state.x == g.x && env.state.y==g.y
    goal_bit_mask = check_in_goal.(env.goals)
    if any(goal_bit_mask)
        env.goals[goal_bit_mask][1].r
    else
        MaskedGridWorldConst.STEP_REW
    end
end

function MinimalRLCore.is_terminal(env::MaskedGridWorld) # -> determines if the agent_state is terminal
    check_in_goal = (g) -> env.state.x == g.x && env.state.y==g.y
    any(check_in_goal.(env.goals))
end

function MinimalRLCore.get_state(env::MaskedGridWorld) # -> get state of agent
    if env.obs_strategy == :full
        fully_observable_state(env)
    elseif env.obs_strategy == :seperate
        partial_obs_seperate(env)
    elseif env.obs_strategy == :aliased
        partial_obs_aliased(env)
    else
        @error "$(env.obs_strategy) not a supported observation strategy for MaskedGridWorld"
    end
end

function fully_observable_state(env::MaskedGridWorld)
    st = zeros(Float32, obs_size(env))
    st[1] = env.state.x
    st[2] = env.state.y
    st
end

function partial_obs_seperate(env::MaskedGridWorld)
    obs = zeros(Float32, obs_size(env))
    obs_idx = 1 + get_anchor(env)
    obs[obs_idx] = 1
    return obs
end

function partial_obs_aliased(env::MaskedGridWorld)
    obs = zeros(Float32, obs_size(env))
    if get_anchor(env) > 0
        obs[2] = 1
    else
        obs[1] = 1
    end
    return obs
end



function Base.show(io::IO, env::MaskedGridWorld)
    padding = "                "
    print(io,
              "MaskedGridWorld(",
          "size: ", size(env), ",\n",
          padding*"agent_state: ", env.state, ",\n",
          padding*"num_goals: ", length(env.goals), ",\n",
          padding*"num_anchors: ", maximum(env.anchors),)
          print(io, ")")
end

import RecipesBase
using Colors
RecipesBase.@recipe function f(env::MaskedGridWorld)

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
    WALL_1 = Colors.RGB(67/255, 36/255, 195/255)
    WALL_2 = Colors.RGB(31/255, 103/255, 36/255)
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


    screen[1:CELL_SIZE, :] .= WALL_1
    screen[end-(CELL_SIZE-1):end, :] .= WALL_1
    screen[:, 1:CELL_SIZE] .= WALL_2
    screen[:, end-(CELL_SIZE-1):end] .= WALL_2

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
            check_in_goal = (g) -> i == g.x && j ==g.y
            screen[sqr_j, sqr_i] .= any(check_in_goal.(env.goals)) ? goal : cell
            
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

