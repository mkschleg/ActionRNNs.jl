
# abstract type AbstractRecurType end

# struct NoRecur end
# struct Recur <: AbstractRecurType end
# struct ActionRecur <: AbstractRecurType end

# function RecurType(m)
#     if contains(m, Union{Flux.LSTMCell, Flux.GRUCell, Flux.RNNCell, RNNCell})
#         Recur()
#     elseif contains(m, AbstractActionRNN)
#         ActionRecur()
#     else
#         NoRecur()
#     end
# end

abstract type AbstractLearner end

model(l::AbstractLearner) = l.model

function (l::AbstractLearner)(ϕ)
    map(model(l), ϕ)
end
(l::AbstractLearner)(ϕ) = model(l)(ϕ)

function (l::AbstractLearner)(ϕ, hs::IdDict)
    reset!(l.model, hs)
    l(ϕ)
end

abstract type AbstractValueLearner <: AbstractLearner end

mutable struct VLearner{M, O, LU, TN} <: AbstractValueLearner
    model::M
    opt::O
    lu::LU
    tn::TN
end

function VLearner(model, opt, lu; target_network = true)
    recur_type = RecurType(model)
    VLearner{recur_type}(model, opt, lu, deepcopy(model))
end


function learner_start!(learner::VLearner)
    # Does anything need here?
end

function update!(learner::VLearner, horde, hidden_state_init, state_list, env_s_tp1, action, action_prob)
    update!(learner.model,
            horde,
            learner.opt,
            learner.lu,
            hidden_state_init,
            state_list,
            env_s_tp1,
            action,
            action_prob)
end

mutable struct QLearner{RECUR_TYPE, M, O, LU, TN} <: AbstractValueLearner
    model::M
    opt::O
    lu::LU
    tn::TN
end

(l::QLearner)(ϕ::Array{<:Number}) = model(l)(ϕ)
(l::QLearner)(ϕ::Array{<:Number}, a) = model(l)(ϕ)[a]
(l::QLearner{ActionRecur})(ϕ::Tuple) = model(l)(ϕ)
(l::QLearner{ActionRecur})(ϕ::Tuple, a) = model(l)(ϕ)[a]

function (l::QLearner)(ϕ, hs::IdDict)
    reset!(l.model, hs)
    l(ϕ)
end

function QLearner(model, opt, lu; target_network = true)
    recur_type = RecurType(model)
    QLearner{recur_type}(model, opt, lu, deepcopy(model))
end


function learner_start!(learner::QLearner)
    # Does anything need here?
end

function update!(learner::QLearner, stuff...)

end



