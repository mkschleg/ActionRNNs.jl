abstract type ValueFunctionLearner <: Learner end

update(l::ValueFunctionLearner) = l.update

"""
   QLearner(model, num_actions, num_demons)
"""
mutable struct QLearner{F, LU<:LearningUpdate} <: ValueFunctionLearner
    model::F
    update::LU
    num_actions::Int
end

predict(l::QLearner, ϕ::AbstractArray{F}) where {F<:Number} = l.model(ϕ)
predict(l::QLearner, hs::Dict, ϕ) = begin
    reset!(l.model, hs)
    predict(l, ϕ)
end
predict(l::QLearner, ϕ::AbstractArray{F}) where {F<:AbstractArray} = l.model.(ϕ)

(l::QLearner)(ϕ) = predict(l, ϕ)

predict(l::QLearner, ϕ::AbstractVector{<:Number}, a) = predict(l, ϕ)[a]

(l::QLearner)(ϕ, a) = predict(l, ϕ, a)


"""
   VLearner(model, num_actions, num_demons)
"""
mutable struct VLearner{F, LU<:LearningUpdate, H} <: ValueFunctionLearner
    model::F
    update::LU
    horde::H
end

predict(l::VLearner, ϕ::AbstractArray{F}) where {F<:Number} = l.model(ϕ)
predict(l::VLearner, hs::Dict, ϕ) = begin
    reset!(l.model, hs)
    predict(l, ϕ)
end
predict(l::VLearner, ϕ::AbstractArray{F}) where {F<:AbstractArray} = l.model.(ϕ)

(l::VLearner)(ϕ) = predict(l, ϕ)

predict(l::VLearner, ϕ::AbstractVector{<:Number}, a) = predict(l, ϕ)[a]

(l::VLearner)(ϕ, a) = predict(l, ϕ, a)


