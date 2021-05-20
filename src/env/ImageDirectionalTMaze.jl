using Colors

mutable struct ImageDirTMaze{TM, L} <: AbstractEnvironment
    env::TM
    r_state::Array{UInt8, 3}
    data::Vector{Array{UInt8, 3}}
    labels::L
    l_c_idx::Int
    l_ic_idx::Int
end

@forward ImageDirTMaze.env MinimalRLCore.get_reward
@forward ImageDirTMaze.env MinimalRLCore.is_terminal
@forward ImageDirTMaze.env MinimalRLCore.get_actions

@forward ImageDirTMaze.env Base.size
@forward ImageDirTMaze.env Base.show

function ImageDirTMaze(size)
    d = Flux.Data.MNIST.images()
    d_r = [collect(reshape(reinterpret(UInt8, d[i]), 28, 28, 1)) for i ∈ 1:length(d)]
    l = Flux.Data.MNIST.labels()
    ImageDirTMaze(
        DirectionalTMaze(size),
        zero(d_r[1]),
        d_r, l, 0, 0
    )
end
   
function MinimalRLCore.reset!(env::ImageDirTMaze, rng::AbstractRNG=Random.GLOBAL_RNG)
    MinimalRLCore.reset!(env.env, rng)
    
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
    env.r_state .= rand(Random.GLOBAL_RNG, UInt8, 28, 28, 1)
end


obs_dims(env::ImageDirTMaze) = (28, 28, 1)

function MinimalRLCore.get_state(env::ImageDirTMaze) # -> get state of agent
    state = env.env.state
    sze = size(env.env)
    if state.x == 1
        if state.dir == env.env.goal_dir
            env.data[env.l_c_idx]
            fill(0x0F, obs_dims(env))
        elseif state.dir == DirectionalTMazeConst.EAST 
            fill(0x00, obs_dims(env))
        elseif env.env.state.dir == DirectionalTMazeConst.WEST
            # fill(0xff, obs_dims(env)...)
            env.r_state
        else
            fill(0xF0, obs_dims(env))
            # env.data[env.l_ic_idx]
        end

    elseif state.x == sze && state.dir == DirectionalTMazeConst.EAST
        # fill(0xff, obs_dims(env)...)
        env.r_state
    elseif state.dir == DirectionalTMazeConst.NORTH || state.dir == DirectionalTMazeConst.SOUTH
        env.r_state
    else
        fill(0x00, obs_dims(env))
    end
end
