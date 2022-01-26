using Colors
import MLDatasets

mutable struct ImageDirTMaze{TM, L} <: AbstractEnvironment
    env::TM
    r_state::Array{UInt8, 3}
    data::Vector{Array{UInt8, 3}}
    labels::L
    l_c_idx::Int
    l_ic_idx::Int
end

# @forward ImageDirTMaze.env MinimalRLCore.get_reward
MinimalRLCore.get_reward(env::ImageDirTMaze) = begin
    if env.env.state.x == env.env.size
        if env.env.state.y == 0
            DTMC.BAD_REW
        elseif env.env.state.y == 1
            env.env.goal_dir == DTMC.NORTH ? DTMC.GOOD_REW : -DTMC.GOOD_REW#DTMC.BAD_REW
        elseif env.env.state.y == -1
            env.env.goal_dir == DTMC.SOUTH ? DTMC.GOOD_REW : -DTMC.GOOD_REW
        end
    else
        DTMC.BAD_REW
    end
end
@forward ImageDirTMaze.env MinimalRLCore.is_terminal
@forward ImageDirTMaze.env MinimalRLCore.get_actions

@forward ImageDirTMaze.env Base.size
@forward ImageDirTMaze.env Base.show

function ImageDirTMaze(size)
    d, l = MLDatasets.MNIST.traindata();
    proc = (img) -> collect(reinterpret(UInt8, reshape(img', 28, 28, 1)))
    d_r = proc.(eachslice(d, dims=3))
    ImageDirTMaze(
        DirectionalTMaze(size),
        zero(d_r[1]),
        d_r, l, 0, 0
    )
end
   
function MinimalRLCore.reset!(env::ImageDirTMaze, rng::AbstractRNG=Random.GLOBAL_RNG)
    MinimalRLCore.reset!(env.env, rng)
    env.r_state = rand(rng, UInt8, 28, 28, 1)
    while true
        env.l_c_idx = rand(rng, keys(env.labels))
        if  env.labels[env.l_c_idx] % 2 == 0
            break
        end
    end

    while true
        env.l_ic_idx = rand(rng, keys(env.labels))
        if  env.labels[env.l_ic_idx] % 2 == 1
            break
        end
    end
end

function MinimalRLCore.environment_step!(env::ImageDirTMaze, action::Int, rng=Random.GLOBAL_RNG)
    MinimalRLCore.environment_step!(env.env, action, rng)
end


obs_dims(env::ImageDirTMaze) = (28, 28, 1)

module ImageDirTMazeConst

const GOAL_OBS = fill(0xFF, (28, 28, 1))
const WALL_OBS = begin
    ret = fill(0xFF, (28, 28, 1))
    ret[1:14, 1:14] .= 0x00
    ret
end

const HALL_OBS = fill(0x00, (28, 28, 1))

end

function MinimalRLCore.get_state(env::ImageDirTMaze) # -> get state of agent
    state = env.env.state
    sze = size(env.env)
    IMDTMC = ImageDirTMazeConst
    if state.x == 1
        if state.dir == env.env.goal_dir
            env.data[env.l_c_idx]
            # fill(0xFF, obs_dims(env))
            # IMDTMC.GOAL_OBS
        elseif state.dir == DirectionalTMazeConst.EAST 
            # fill(0x33, obs_dims(env))
            IMDTMC.HALL_OBS
        elseif env.env.state.dir == DirectionalTMazeConst.WEST
            # fill(0xff, obs_dims(env)...)
            env.r_state
            IMDTMC.WALL_OBS
        else
            # fill(0xF0, obs_dims(env))
            # env.r_state
            # IMDTMC.WALL_OBS
            env.data[env.l_ic_idx]
        end

    elseif state.x == sze && state.dir == DirectionalTMazeConst.EAST
        # fill(0xff, obs_dims(env)...)
        env.r_state
        IMDTMC.WALL_OBS
    elseif state.dir == DirectionalTMazeConst.NORTH || state.dir == DirectionalTMazeConst.SOUTH
        env.r_state
        IMDTMC.WALL_OBS
    else
        IMDTMC.HALL_OBS
        # fill(0x33, obs_dims(env))
    end
end
