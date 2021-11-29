

# test contract_WA

# import ActionRNNs
using Flux
import TensorCore: ⊡
using KernelAbstractions, LoopVectorization # Loop vectorization and kernel abstractions
using CUDA, CUDAKernels # GPUs
using Tullio
using Random

using BenchmarkTools
import LinearAlgebra: dot

function contract_WA_tensorcore(W, a::AbstractVector{Int}, x)
    #=
    ⊡ Generalised matrix multiplication: Contracts the last dimension of `A` with
    the first dimension of `B`, for any `ndims(A)` & `ndims(B)`.
    If both are vectors, then it returns a scalar `== sum(A .* B)`.
    =#
    mid = W ⊡ x
    @tullio ret[i, k] := mid[a[k], i, k]
end

function contract_WA_tullio(W, a::AbstractVector{Int}, x)
    #=
    ⊡ Generalised matrix multiplication: Contracts the last dimension of `A` with
    the first dimension of `B`, for any `ndims(A)` & `ndims(B)`.
    If both are vectors, then it returns a scalar `== sum(A .* B)`.
    =#
    # mid = W ⊡ x
    @tullio ret[i, k] := W[a[k], i, j] * x[j, k]
end

function contract_WA_custom(W, a::AbstractVector{Int}, x)
    #=
    ⊡ Generalised matrix multiplication: Contracts the last dimension of `A` with
    the first dimension of `B`, for any `ndims(A)` & `ndims(B)`.
    If both are vectors, then it returns a scalar `== sum(A .* B)`.
    =#
    # mid = W ⊡ x
    # @tullio ret[i, k] := W[a[k], i, j] * x[j, k]
    ret = zeros(eltype(W), size(W, 2), size(x, 2))
    for i in 1:size(W, 2), k in 1:size(x, 2)
        ret[i, k] = dot(W[a[k], i, :], x[:, k])
    end
    ret
end



function test_contract_WA(f, g, nb=128, na=18, ni=1024, no=1024; seed=1)
    Random.seed!(seed)
    a = rand(1:na, nb)
    x = rand(ni, nb)
    W = rand(na, no, ni)

    all((f(W, a, x) .≈ g(W, a, x)))
end

function benchmark_contract_WA(f, nb=128, na=18, ni=1024, no=1024; seed=1)

    Random.seed!(seed)
    a = rand(1:na, nb)
    x = rand(ni, nb)
    W = rand(no, na, ni)

    @benchmark $f($W, $a, $x)
    
end


function take_grad(f, W, a, x)
    gradient(Flux.params(W)) do
        sum(f(W, a, x))
    end
end

function benchmark_grad_contract_WA(f, nb=128, na=18, ni=1024, no=1024; seed=1)
    Random.seed!(seed)
    a = rand(1:na, nb)
    x = rand(ni, nb)
    W = rand(no, na, ni)

    @benchmark take_grad($f, $W, $a, $x)
    
end


