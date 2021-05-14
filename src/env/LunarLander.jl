

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
    function LunarLander(seed, continuous=false)
        if continuous
            new(OpenAIGym.GymEnv(:LunarLanderContinuous, :v2; seed=seed), continuous)
        else
            new(OpenAIGym.GymEnv(:LunarLander, :v2; seed=seed), continuous)
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
get_num_features(env::LunarLander) = 8
MinimalRLCore.get_reward(env::LunarLander) =
    Float32(MinimalRLCore.get_reward(env.gym))
MinimalRLCore.is_terminal(env::LunarLander) = MinimalRLCore.is_terminal(env.gym)

function MinimalRLCore.get_state(env::LunarLander) # -> get state of agent

    pystate = env.gym.state
#     x_in = if -0.5 < pystate[1] 0.5
#         1.0f0
#     else
#         0.0f0
#     end
#     y = pystate[2]
#
#     ang = if -0.05 < pystate[5] < 0.05
#         1.0f0
#     else
#         0.0f0
#     end
#
#     left_bumper = pystate[end-1]
#     right_bumper = pystate[end]
#
#     [x_in, y, ang, left_bumper, right_bumper]
end

MinimalRLCore.environment_step!(env::LunarLander, action, rng=Random.GLOBAL_RNG) =
    MinimalRLCore.environment_step!(env.gym, action-1)

function Base.show(io::IO, env::LunarLander)
    s = env.gym.state
    print(io, "LunarLander(x: $(s[1]), y: $(s[2]), vx: $(s[3]), vy: $(s[4]), θ: $(s[5]), dθ: $(s[6]))")
end




