

import KernelAbstractions, LoopVectorization # Loop vectorization and kernel abstractions
import CUDA, CUDAKernels # GPUs
import TensorCore: ⊡ # boxdot Generalised matrix multiplication
using Tullio


"""
    contract_WA(W, a::Int, x)
    contract_WA(W, a::AbstractVector{Int}, x)
    contract_WA(W, a::AbstractVector{<:AbstractFloat}, x)
    contract_WA(W::CuArray, a::AbstractVector{Int}, x)

This contraction operator will take the weights `W`, action (or action vector for batches) `a`, and features.
The weight matrix is assumed to be in `nactions × out × in`.
"""
contract_WA(W, a::Int, x) = W[a, :, :]*x

function contract_WA(W, a::AbstractVector{Int}, x::AbstractMatrix)
    # original tullio. Removed for faster blas operation
    # @tullio ret[i, k] := W[a[k], i, j] * x[j, k]
    mid = W ⊡ x
    @tullio ret[i, k] := mid[a[k], i, k]
end

function contract_WA(W, a::AbstractVector{Int}, x::AbstractVector)
    @tullio ret[i, k] := W[a[k], i, 1] * x[k] # if state is scalar?
end

function contract_WA(W, a::AbstractVector{<:AbstractFloat}, x)
    @tullio ret[i] := W[k, i, j] * a[k] * x[j]
end

function contract_WA(W, a::AbstractMatrix{<:AbstractFloat}, x)
    @tullio ret[i, k] := W[l, i, j] * a[l, k] * x[j, k]
end

# GPU Versions
# Maybe fixed by new version of tullio. Need to test.
function contract_WA(W::CuArray, a::AbstractVector{Int}, x)
    #=
    ⊡ Generalised matrix multiplication: Contracts the last dimension of `A` with
    the first dimension of `B`, for any `ndims(A)` & `ndims(B)`.
    If both are vectors, then it returns a scalar `== sum(A .* B)`.
    
    View is needed to prevent tullio reverting to scalar operations on gpu. I wonder if we can solve this another way...
    =#
    mid = @view (W ⊡ x)[a, :, :]
    @tullio ret[i, k] := mid[k, i, k]
end


# Precompile commands to reduce latency.
precompile(contract_WA, (Matrix{Float32}, Vector{Int}, Matrix{Float32}))
precompile(contract_WA, (Matrix{Float32}, Int, Vector{Float32}))
precompile(contract_WA, (Matrix{Float32}, Int, Matrix{Float32}))


"""
    get_waa(Wa, a)

Different ways of handeling geting action value from a set of weights. This operation
can be seen as `Wa*a` where `Wa` is the weight matrix, and a is the action representation.
This is to be used with various cells to incorporate this operation more reliably.

"""
get_waa(Wa, a::Int) = Wa[:, a]
get_waa(Wa, a::Vector{Int}) = Wa[:, a]
get_waa(Wa, a::AbstractArray{<:AbstractFloat}) = Wa*a


function contract_Wga(Wg, Wa::AbstractVector{<:Number})
    @tullio ret[q, r] := Wg[p, q, r] * Wa[p]
end

function contract_Wgax(Wg, Wa::AbstractMatrix{<:Number}, Wx::AbstractMatrix{<:Number})
    @tullio ret[q, k] := Wg[p, q, r] * Wa[p, k] * Wx[r, k]
end


