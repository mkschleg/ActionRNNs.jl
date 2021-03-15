

abstract type AbstractLearner end

mutable struct PredictionLearner{M, ER, LU} <: AbstractLearner
    model::M
    replay::ER
    lu::LU

    
end

