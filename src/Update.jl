
using Statistics
using LinearAlgebra: Diagonal
# import Flux.Tracker.update!
using Flux.Optimise: apply!
using Flux
using Zygote: dropgrad, ignore, Buffer

abstract type LearningUpdate end

# Control
include("update/QLearning.jl")

# Predictions
include("update/TDLearning.jl")



