
import StatsBase
import StatsBase: Weights
import Random
# using CuArrays

abstract type AbstractPolicy end

mutable struct RandomActingPolicy{T<:AbstractFloat} <: AbstractPolicy
    probabilities::Array{T,1}
    weight_vec::Weights{T, T, Array{T, 1}}
    RandomActingPolicy(probabilities::Array{T,1}) where {T<:AbstractFloat} =
        new{T}(probabilities, Weights(probabilities))
end

get_prob(π::RandomActingPolicy, action_t) =
    π.probabilities[action_t]

sample(π::RandomActingPolicy) =
    sample(Random.GLOBAL_RNG, π)

sample(rng::Random.AbstractRNG, π::RandomActingPolicy) =
    StatsBase.sample(rng, π.weight_vec)

function (π::RandomActingPolicy)(state_t, rng::Random.AbstractRNG=Random.GLOBAL_RNG)
    action = sample(rng, π)
    return action, get_prob(π, action)
end


abstract type AbstractValuePolicy <: AbstractPolicy end

action_set(ap::AbstractValuePolicy) = nothing
Base.eltype(ap::AbstractValuePolicy) = eltype(action_set(ap))

_get_max_action(ap::AbstractValuePolicy, values) =
    action_set(ap)[findmax(values)[2]]
# _get_max_action(ap::AbstractValuePolicy, values::CuArray) = 
#     action_set(ap)[findmax(cpu(values))[2]]

"""
    ϵGreedy(ϵ, action_set)
    ϵGreedy(ϵ, num_actions)

Simple ϵGreedy value policy.
"""
Base.@kwdef struct ϵGreedy{AS} <: AbstractValuePolicy
    ϵ::Float64
    action_set::AS
end

ϵGreedy(ϵ::Float64, num_actions::Int) = ϵGreedy(ϵ, 1:num_actions)

action_set(ap::ϵGreedy) = ap.action_set

"""
    sample(ap::ϵGreedy, values, rng)

Select an action according to the values.
"""
sample(ap::ϵGreedy, values) = StatsBase.sample(Random.GLOBAL_RNG, ap, values)
function sample(rng::Random.AbstractRNG, ap::ϵGreedy, values)
    if rand(rng) > ap.ϵ
        return ap.action_set[findmax(values)[2]]
    else
        return rand(rng, ap.action_set)
    end
end

"""
    get_prob(ap::ϵGreedy, values, action)

Get probabiliyt of action according to values.
"""
function get_prob(ap::ϵGreedy, values, action)
    if action == findmax(values)[2]
        return 1 - ap.ϵ + (ap.ϵ / length(ap.action_set))
    else
        return ap.ϵ / length(ap.action_set)
    end
end


"""
    ϵGreedyDecay{AS}(ϵ_range, decay_period, warmup_steps, action_set::AS)
    ϵGreedyDecay(ϵ_range, end_step, num_actions)

This is an acting policy which decays exploration linearly over time. This api will possibly change overtime once I figure out a better way to specify decaying epsilon.

# Arguments
`ϵ_range::Tuple{Float64, Float64}`: (max epsilon, min epsilon)
`decay_period::Int`: period epsilon decays
`warmup_steps::Int`: number of steps before decay starts
"""
Base.@kwdef mutable struct ϵGreedyDecay{AS} <: AbstractValuePolicy
    ϵ_range::Tuple{Float64, Float64}
    decay_period::Int
    warmup_steps::Int
    cur_step::Int = 0
    action_set::AS
    ϵGreedyDecay(ϵ_range, decay_period, warmup_steps, action_set::AS) where {AS} =
        new{AS}(ϵ_range, decay_period, warmup_steps, 0, action_set)
end

ϵGreedyDecay(ϵ_range, end_step, num_actions) = ϵGreedyDecay(ϵ_range, end_step, 1:num_actions)

action_set(ap::ϵGreedyDecay) = ap.action_set

function _get_eps_for_step(ap::ϵGreedyDecay, step=ap.cur_step)
    ϵ_min = ap.ϵ_range[2]
    ϵ_max = ap.ϵ_range[1]
    
    steps_left = ap.decay_period + ap.warmup_steps - ap.cur_step
    bonus = (ϵ_max - ϵ_min) * steps_left / ap.decay_period
    bonus = clamp(bonus, 0.0, ϵ_max - ϵ_min)
    ϵ_min + bonus
end

sample(ap::ϵGreedyDecay, values) = StatsBase.sample(Random.GLOBAL_RNG, ap, values)
function sample(rng::Random.AbstractRNG, ap::ϵGreedyDecay, values, step=ap.cur_step)
    ϵ = _get_eps_for_step(ap::ϵGreedyDecay, step)
    if rand(rng) > ϵ
        return _get_max_action(ap, values)
    else
        return rand(rng, ap.action_set)
    end
end

function get_prob(ap::ϵGreedyDecay, values, action, step=ap.cur_step)
    ϵ = _get_eps_for_step(ap, step)
    if ap.action_set[action] == _get_max_action(ap, values)
        return 1 - ϵ + (ϵ / length(ap.action_set))
    else
        return ϵ / length(ap.action_set)
    end
end

