using Flux
# using OMEinsum
using KernelAbstractions, LoopVectorization
using OMEinsum
using Tullio

####
# Abstraction to help deal w/ whether to pass in tuple or not.
####
abstract type AbstractActionRNN end
_needs_action_input(m) = false
_needs_action_input(m::Flux.Recur{T}) where {T} = _needs_action_input(m.cell)
_needs_action_input(m::AbstractActionRNN) = true


####
# Dealing with tensor operations
####
contract_WA(W, a::Int, x) =
    W[a, :, :]*x

contract_WA(W, a::Vector{Int}, x) = begin
    @tullio ret[i, k] := W[a[k], i, j] * x[j, k]
end

contract_WA(W, a::Vector{Int}, x::AbstractVector{<:Number}) = begin
    @tullio ret[i, k] := W[a[k], i, 1] * x[k]
end

contract_WA(W, a::AbstractVector{<:Number}, x) =
    @tullio ret[i] := W[k, i, j] * a[k] * x[j]

contract_WA(W, a::AbstractMatrix{<:Number}, x) =
    @tullio ret[i, k] := W[l, i, j] * a[l, k] * x[j, k]

get_waa(Wa, a::Int) = Wa[:, a]
get_waa(Wa, a::Vector{Int}) = Wa[:, a]
get_waa(Wa, a::AbstractArray{<:AbstractFloat}) = Wa*a


include("RNNUtil.jl")


# export ARNN, ALSTM, AGRU, reset!, get
export AARNN, MARNN, AAGRU, MAGRU, FacARNN, reset!, get
include("rnns/RNN.jl")
include("rnns/GRU.jl")
include("rnns/LSTM.jl")

# Keep list of all the non-specialized RNN types in ActionRNNs
rnn_types() = ["AARNN", "MARNN", "AAGRU", "MAGRU", "AALSTM", "MALSTM"]
fac_rnn_types() = ["FacMARNN", "FacMAGRU"]
