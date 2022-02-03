
import MinimalRLCore: AbstractAgent
import Random

struct RandomAgent{T} <: AbstractAgent
    actions::T
end
MinimalRLCore.start!(agent::RandomAgent, env_s_tp1, rng=Random.GLOBAL_RNG) = rand(rng, agent.actions)
MinimalRLCore.step!(agent::RandomAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG) = rand(rng, agent.actions)

