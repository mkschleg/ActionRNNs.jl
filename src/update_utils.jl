

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

# abstract type AbstractIntrospection end

# apply(tpl::Tuple{Vararg{T} where {T<:AbstractIntrospection}}, us::UpdateState) =
#     [apply(interspect, us) for interspect ∈ tpl]
# struct L1GradIntrospection{F} <: AbstractIntrospection where F end
# struct L2GradIntrospection{F} <: AbstractIntrospection where F end
# struct LossIntrospection{F} <: AbstractIntrospection where F end

# # apply(::AbstractIntrospection, ::UpdateState) = nothing
# apply(::L1GradIntrospection{F<:AbstractFloat}, us::UpdateState) = begin
#     l1_grad = zero(F)
#     for (k, v) ∈ us.grads
#         if !(v isa Nothing)
#             l1_grad += sum(abs, v)
#         end
#     end
#     l1_grad
# end
# apply(::L2GradIntrospection{F<:AbstractFloat}, us::UpdateState) = begin
#     l2_grad = zero(F)
#     for (k, v) ∈ us.grads
#         if !(v isa Nothing)
#             l2_grad += sum(abs2, v)
#         end
#     end
#     l2_grad
# end
# apply(::LossIntrospection, us::UpdateState) = us.loss


get_params(model, h_init, hs_learnable=false) = if hs_learnable
        Flux.params(model, [h_v for (h_k, h_v) ∈ h_init]...)
    else
        Flux.params(model)
    end


get_batch_idx(x::AbstractArray{Float32, 1}, i) = x[i]
get_batch_idx(x::AbstractArray{Float32, 2}, i) = x[:, i]
get_batch_idx(x::AbstractArray{Float32, 3}, i) = x[:, :, i]
get_batch_idx(x::AbstractArray{Float32, 4}, i) = x[:, :, :, i]


set_batch_idx!(x::Array{F, 1}, y::F, i) where F = x[i] .= y
set_batch_idx!(x::Array{F, 2}, y::AbstractArray{F, 1}, i) where F = x[:, i] .= y
set_batch_idx!(x::Array{F, 3}, y::AbstractArray{F, 2}, i) where F = x[:, :, i] .= y
set_batch_idx!(x::Array{F, 4}, y::AbstractArray{F, 3}, i) where F = x[:, :, :, i] .= y

set_batch_idx!(x::CuArray{F, 1}, y::F, i) where F = x[i] = y
set_batch_idx!(x::CuArray{F, 2}, y::AbstractArray{F, 1}, i) where F = copyto!(view(x, :, i), y)
set_batch_idx!(x::CuArray{F, 3}, y::AbstractArray{F, 2}, i) where F = copyto!(view(x, :, :, i), y)
set_batch_idx!(x::CuArray{F, 4}, y::AbstractArray{F, 3}, i) where F = copyto!(view(x, :, :, :, i), y)


