using Colors

mutable struct ImageTMaze{TM, RS, D, L} <: AbstractEnvironment
    env::TM
    r_state::RS
    data::D
    labels::L
    l_idx::Int
end

@forward ImageTMaze.env MinimalRLCore.get_reward
@forward ImageTMaze.env MinimalRLCore.is_terminal
@forward ImageTMaze.env MinimalRLCore.get_actions

@forward ImageTMaze.env Base.size
@forward ImageTMaze.env Base.show

function ImageTMaze(size)
    d = Flux.Data.MNIST.images()
    l = Flux.Data.MNIST.labels()
    ImageTMaze(
        TMaze(size),
        zero(d[1]),
        d, l, 0
    )
end

function ImageVariableTMaze(size)
    d = Flux.Data.MNIST.images()
    l = Flux.Data.MNIST.labels()
    ImageTMaze(
        VariableTMaze(size),
        zero(d[1]),
        d, l, 0
    )
end
   
function MinimalRLCore.reset!(env::ImageTMaze, rng::AbstractRNG=Random.GLOBAL_RNG)
    MinimalRLCore.reset!(env.env, rng)
    while true
        env.l_idx = rand(rng, keys(env.labels))
        if env.env.goal_dir == TMazeConst.UP && env.labels[env.l_idx] % 2 == 0
            break
        elseif env.env.goal_dir == TMazeConst.DOWN && env.labels[env.l_idx] % 2 == 1
            break
        end
    end
end

function MinimalRLCore.environment_step!(env::ImageTMaze, action::Int, rng=Random.GLOBAL_RNG)
    MinimalRLCore.environment_step!(env.env, action, rng)
    env.r_state .= Gray{Colors.N0f8}.(reinterpret.(Colors.N0f8, (rand(Random.GLOBAL_RNG, UInt8, 28, 28))))
end


obs_dims(env::ImageTMaze) = (28, 28)

function MinimalRLCore.get_state(env::ImageTMaze) # -> get state of agent
    state = env.env.state
    sze = size(env.env)
    if state.x == 1
        env.data[env.l_idx]
    elseif state.x == sze
        ones(Colors.Gray{Colors.N0f8}, obs_dims(env)...)
    else
        env.r_state
    end
end


