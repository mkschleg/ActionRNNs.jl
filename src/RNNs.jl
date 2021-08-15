
using Flux # Neural Networks
using KernelAbstractions, LoopVectorization # Loop vectorization and kernel abstractions
using CUDA, CUDAKernels # GPUs
import TensorCore: ⊡ # boxdot Generalised matrix multiplication
using Tullio


#=
# Abstraction to help deal w/ whether to pass in tuple or not.
=#

"""
    AbstractActionRNNCell

An abstract struct which will take the current hidden state and 
a tuple of observations and actions and returns the next hidden state.
"""
abstract type AbstractActionRNN end
_needs_action_input(m) = false
_needs_action_input(m::Flux.Recur{T}) where {T} = _needs_action_input(m.cell)
_needs_action_input(m::AbstractActionRNN) = true


"""
    contract_WA(W, a::Int, x)
    contract_WA(W, a::AbstractVector{Int}, x)
    contract_WA(W::CuArray, a::Vector{Int}, x)

This contraction operator will take the weights `W`, action (or action vector for batches) `a`, and features.
The weight matrix is assumed to be in `nactions × out × in`.

    contract_WA(W, )
"""
contract_WA(W, a::Int, x) =
    W[a, :, :]*x

# Maybe fixed by new version of tullio.
contract_WA(W, a::AbstractVector{Int}, x) = begin
    #=
    ⊡ Generalised matrix multiplication: Contracts the last dimension of `A` with
    the first dimension of `B`, for any `ndims(A)` & `ndims(B)`.
    If both are vectors, then it returns a scalar `== sum(A .* B)`.
    =#
    mid = W ⊡ x
    @tullio ret[i, k] := mid[a[k], i, k]
end

# Maybe fixed by new version of tullio.
contract_WA(W::CuArray, a::Vector{Int}, x) = begin
    Wa = W[a, :, :]
    @tullio ret[i, k] := Wa[k, i, j] * x[j, k]
end

contract_WA(W, a::AbstractVector{Int}, x::AbstractVector) = begin
    @tullio ret[i, k] := W[a[k], i, 1] * x[k]
end

contract_WA(W, a::AbstractVector{<:AbstractFloat}, x) =
    @tullio ret[i] := W[k, i, j] * a[k] * x[j]

contract_WA(W, a::AbstractMatrix{<:AbstractFloat}, x) =
    @tullio ret[i, k] := W[l, i, j] * a[l, k] * x[j, k]

get_waa(Wa, a::Int) = Wa[:, a]
get_waa(Wa, a::Vector{Int}) = Wa[:, a]
get_waa(Wa, a::AbstractArray{<:AbstractFloat}) = Wa*a

contract_Wga(Wg, Wa::AbstractVector{<:Number}) =
    @tullio ret[q, r] := Wg[p, q, r] * Wa[p]

contract_Wgax(Wg, Wa::AbstractMatrix{<:Number}, Wx::AbstractMatrix{<:Number}) =
    @tullio ret[q, k] := Wg[p, q, r] * Wa[p, k] * Wx[r, k]


include("RNNUtil.jl")

include("rnns/RNN.jl")
include("rnns/GRU.jl")
include("rnns/LSTM.jl")

include("rnns/ActionGated.jl")

# Keep list of all the non-specialized RNN types in ActionRNNs
rnn_types() = ["AARNN", "MARNN", "AAGRU", "MAGRU", "AALSTM", "MALSTM"]
fac_rnn_types() = ["FacMARNN", "FacMAGRU"]
gated_rnn_types() = ["ActionGatedRNN"]
fac_tuc_rnn_types() = ["FacTucMARNN", "FacTucMAGRU"]
