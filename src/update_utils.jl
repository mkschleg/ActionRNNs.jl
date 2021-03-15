

using LinearAlgebra

# Learning Phases to make evaluation easier

abstract type AbstractLearningPhase end
struct TrainingPhase <: AbstractLearningPhase end
struct EvaluationPhase <: AbstractLearningPhase end


mutable struct UpdateState{LT, GT, PST, OPT}
    loss::LT
    grads::GT
    params::PST
    opt::OPT
end

UpdateState() = UpdateState(nothing, nothing, nothing, nothing)

# Update introspections.

abstract type AbstractIntrospection end

apply(tpl::Tuple{Vararg{T} where {T<:AbstractIntrospection}}, us::UpdateState) =
    [apply(interspect, us) for interspect ∈ tpl]
struct L1GradIntrospection <: AbstractIntrospection end
struct L2GradIntrospection <: AbstractIntrospection end
struct LossIntrospection <: AbstractIntrospection end

apply(::AbstractIntrospection, ::UpdateState) = nothing
apply(::L1GradIntrospection, us::UpdateState) = begin
    l1_grad = 0.0f0
    for (k, v) ∈ us.grads
        l1_grad += sum(abs.(v))
    end
    l1_grad
end
apply(::L2GradIntrospection, us::UpdateState) = begin
    l2_grad = 0.0f0
    for (k, v) ∈ us.grads
        l2_grad += dot(v, v)
    end
    l2_grad
end
apply(::LossIntrospection, us::UpdateState) = us.loss


get_params(model, h_init, hs_learnable=false) = if hs_learnable
        Flux.params(model, [h_v for (h_k, h_v) ∈ h_init]...)
    else
        Flux.params(model)
    end

