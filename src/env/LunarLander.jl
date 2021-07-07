

using Random
import MinimalRLCore

import Reproduce
import OpenAIGym


"""
    LunarLander
"""
mutable struct LunarLander <: AbstractEnvironment
    gym::OpenAIGym.GymEnv
    cont::Bool
    omit_states::Vector
    state_conditions::Vector
    function LunarLander(seed, continuous=false, omit_states=[], state_conditions=[])
        if continuous
            new(OpenAIGym.GymEnv(:LunarLanderContinuous, :v2; seed=seed), continuous, omit_states, state_conditions)
        else
            new(OpenAIGym.GymEnv(:LunarLander, :v2; seed=seed), continuous, omit_states, state_conditions)
        end
    end
end

Base.size(env::LunarLander) = env.size

MinimalRLCore.reset!(env::LunarLander, rng::AbstractRNG=Random.GLOBAL_RNG) =
    MinimalRLCore.reset!(env.gym)

MinimalRLCore.get_actions(env::LunarLander) = begin
    if !env.cont
        1:4
    else
        throw("not implemented")
    end
end
# get_num_features(env::LunarLander) = 8
MinimalRLCore.get_reward(env::LunarLander) =
    Float32(MinimalRLCore.get_reward(env.gym))
MinimalRLCore.is_terminal(env::LunarLander) = MinimalRLCore.is_terminal(env.gym)

function MinimalRLCore.get_state(env::LunarLander) # -> get state of agent

    pystate = env.gym.state
    observation = [pystate[i] for i in 1:length(pystate) if !(i in env.omit_states)]
    if 1 in env.state_conditions
        @assert !(1 in env.omit_states)
        observation[1] = if -0.5 < observation[1] < 0.5
            1.0f0
        else
            0.0f0
        end
    end
    if 2 in env.state_conditions
        @assert !(5 in env.omit_states)
        observation[5] = if -0.2356 < observation[5] < 0.2356
            1.0f0
        else
            0.0f0
        end
    end
    observation

end

MinimalRLCore.environment_step!(env::LunarLander, action, rng=Random.GLOBAL_RNG) =
    MinimalRLCore.environment_step!(env.gym, action-1)

function Base.show(io::IO, env::LunarLander)
    s = env.gym.state
    print(io, "LunarLander(x: $(s[1]), y: $(s[2]), vx: $(s[3]), vy: $(s[4]), θ: $(s[5]), dθ: $(s[6]))")
end




