


import OpenAIGym
import OpenAIGym: PyCall
import .PyCall: PyObject, PyNULL, PyAny, pycall!, pycall, pyimport



# --------------------------------------------------------------

"A simple wrapper around the OpenAI gym environments to add to the Reinforce framework"
mutable struct MiniGridEnv <: OpenAIGym.AbstractGymEnv
    name::Symbol
    ver::Symbol
    pyenv::PyObject   # the python "env" object
    pystep::PyObject  # the python env.step function
    pyreset::PyObject # the python env.reset function
    pystepres::PyObject # used to make stepping the env slightly more efficient
    info::Dict{Any, Any}    # store it as a PyObject for speed, since often unused
    state::Dict{String, Any}
    reward::Float64
    actions::AbstractSet
    done::Bool
    function MiniGridEnv(name, ver, pyenv, pystate, state, seed)
        env = new(name, ver, pyenv,
                  pyenv."step", pyenv."reset",
                  pystate, PyNULL(), PyNULL(), state)
        env.pyenv.seed(seed)
        MinimalRLCore.reset!(env)
        env
    end
    # function MiniGridEnv{T}(name, ver, pyenv, pystate, state, seed) where T
    #     env = new{T}(name, ver, pyenv, pyenv."step", pyenv."reset",
    #                  pystate, PyNULL(), PyNULL(), state)
    #     env.pyenv.seed(seed)
    #     MinimalRLCore.reset!(env)
    #     env
    # end
end


# state_type(envname::Symbol) = state_type(Val(envname))
# state_type(::Val{:Blackjack}) = Tuple{Int, Int, Bool}
# state_type(generic) = Vector{Float32}


# function MiniGridEnv(name::Symbol, ver::Symbol = :v0;
#                 stateT = ifelse(use_pyarray_state(name), PyArray, PyAny),
#                 seed=0)
#     if PyCall.ispynull(pysoccer) && name ∈ (:Soccer, :SoccerEmptyGoal)
#         copy!(pysoccer, pyimport("gym_soccer"))
#     end

#     MiniGridEnv(name, ver, pygym.make("$name-$ver"), stateT, seed)
# end

# MiniGridEnv(name::AbstractString; kwargs...) =
#     MiniGridEnv(Symbol.(split(name, '-', limit = 2))...; kwargs...)

# function MiniGridEnv(name::Symbol, ver::Symbol, pyenv, stateT, seed)
#     pystate = pycall(pyenv."reset", PyObject)
#     state = pyenv.reset()
#     T = typeof(state)
#     MiniGridEnv{T}(name, ver, pyenv, pystate, state, seed)
# end

# function Base.show(io::IO, env::MiniGridEnv)
#   println(io, "MiniGridEnv $(env.name)-$(env.ver)")
#   if hasproperty(env.pyenv, :class_name)
#     println(io, "  $(env.pyenv.class_name())")
#   end
#   println(io, "  r  = $(env.reward)")
#   print(  io, "  ∑r = $(env.total_reward)")
# end

# --------------------------------------------------------------

function MinimalRLCore.get_actions(env::AbstractMiniGridEnv, s′)
    OpenAIGym.actionset(env.pyenv.action_space)
end

pyaction(a::Vector) = Any[pyaction(ai) for ai=a]
pyaction(a) = a

"""
`reset!(env::MiniGridEnv)` reset the environment
"""
function MinimalRLCore.reset!(env::MiniGridEnv, args...)
    pycall!(env.pystate, env.pyreset, PyObject)
    convert_state!(env)
    env.reward = 0.0
    env.total_reward = 0.0
    env.actions = MinimalRLCore.get_actions(env, nothing)
    env.done = false
    return env.state
end

"""
    step!(env::MiniGridEnv, a)

take a step in the enviroment
"""
function MinimalRLCore.environment_step!(env::MiniGridEnv, a, args...)


    # pyact = pyaction(a)
    # pycall!(env.pystepres, env.pystep, PyObject, pyact)

    # env.pystate, env.reward, env.done, env.info =
    #     convert(Tuple{PyObject, Float64, Bool, PyObject}, env.pystepres)

    # convert_state!(env)
    # env.total_reward += env.reward

    env.state, env.reward, env.done, env.info = env.pystep(a)
    
    # return (env.reward, env.state)
end

convert_state!(env::MiniGridEnv{T}) where T =
    env.state = convert(T, env.pystate)

convert_state!(env::MiniGridEnv{<:PyArray}) =
    env.state = PyArray(env.pystate)

# Reinforce.finished(env::MiniGridEnv)     = env.done
# Reinforce.finished(env::MiniGridEnv, s′) = env.done
MinimalRLCore.is_terminal(env::MiniGridEnv) = env.done
MinimalRLCore.get_reward(env::MiniGridEnv) = env.reward
MinimalRLCore.get_state(env::MiniGridEnv) = env.state

# --------------------------------------------------------------

# function __init__()
#     # the copy! puts the gym module into `pygym`, handling python ref-counting
#     copy!(pymg, pyimport("gym_minigrid"))
# end

