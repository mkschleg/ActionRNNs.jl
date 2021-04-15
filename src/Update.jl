
using Statistics
using LinearAlgebra: Diagonal
# import Flux.Tracker.update!
using Flux.Optimise: apply!
using Flux
using Flux.Zygote: dropgrad, ignore, Buffer

abstract type LearningUpdate end

abstract type PredictionUpdate <: LearningUpdate end
abstract type ControlUpdate <: LearningUpdate end
abstract type ComboUpdate <: LearningUpdate end

include("update_utils.jl")

# Control
include("update/QLearning.jl")

# Predictions
include("update/TDLearning.jl")



