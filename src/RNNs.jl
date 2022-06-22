
using Flux # Neural Networks



#=
# Abstraction to help deal w/ whether to pass in tuple or not.
=#

"""
    AbstractActionRNN

An abstract struct which will take the current hidden state and 
a tuple of observations and actions and returns the next hidden state.
"""
abstract type AbstractActionRNN end

"""
    _needs_action_input

If true, this means the cell or layer needs a tuple as input.
"""
_needs_action_input(m) = false
_needs_action_input(m::Flux.Recur{T}) where {T} = _needs_action_input(m.cell)
_needs_action_input(m::AbstractActionRNN) = true

# Compat for output size....
function (m::Flux.Recur{<:AbstractActionRNN})(x::Vector{Flux.NilNumber.Nil})
    fill(Flux.NilNumber.nil, size(m.state, 1))
end

function (m::Flux.Recur{<:AbstractActionRNN})(x::Matrix{Flux.NilNumber.Nil})
    fill(Flux.NilNumber.nil, size(m.state, 1), size(x, 2))
end

#=
https://github.com/mkschleg/HelpfulKernelFuncs.jl

to reduce precompilation we remove the kernels. HelpfulKernelFuncs are
just what is in kernels.jl.
=#
import HelpfulKernelFuncs: contract_WA, get_waa, contract_Wga, contract_Wgax

include("RNNUtil.jl")

include("rnns/RNN.jl")
include("rnns/GRU.jl")
include("rnns/LSTM.jl")

include("rnns/ActionGated.jl")
include("rnns/RNNCombo.jl")
include("rnns/GRUCombo.jl")
include("rnns/mixture_rnns.jl")
include("rnns/mixture_grus.jl")
include("rnns/active_gating.jl")

include("rnns/combo_cells_softmax.jl")



