using Flux, Adapt
using Flux.CUDA: CuArray


abstract type Device end

(device::Device)(x, args...) = to_device(device, x, args...)

struct CPU <: Device end
struct GPU <: Device
    memory::Dict{Symbol, CuArray{Float32}}
    GPU() = new(Dict{Symbol, CuArray{Float32}}())
end

function get_mem!(f, device::GPU, symbol)
    get!(f, device.memory, symbol)
end

is_gpu(x) = false
is_gpu(x::Flux.CUDA.CuArray) = true
is_gpu(x::Dense) = is_gpu(x.W)
is_gpu(x::Conv) = is_gpu(x.weight)
is_gpu(x::Tuple) = is_gpu(x[1])
is_gpu(x::Flux.Recur) = is_gpu(x.state)

is_gpu(model::Flux.Chain) = begin
    ig = false
    foreach(x -> ig = ig || is_gpu(x),
            Flux.functor(model)[1])
    ig
end

function Device(model)
    is_gpu(model) ? GPU() : CPU()
end

to_device(device::CPU, x::CuArray, args...) = to_host(x)
to_device(device::CPU, x, args...) = x

to_device(device::GPU, x) = to_gpu(x)
to_device(device::GPU, x::CuArray) = begin
    @warn "Extra call to to_device."
    x
end

function to_device(device::GPU, x, sym::Symbol)
    if sym ∈ keys(device.memory)
        gpu_x = device.memory[sym]
        copyto!(gpu_x, x)
        gpu_x
    else
        device.memory[sym] = to_gpu(x)
        device.memory[sym]
    end
end

to_device(use_cuda::Val{:cpu}, x) = to_host(x)
to_device(use_cuda::Val{:gpu}, x) = to_gpu(x)



to_host(m) = fmap(x -> adapt(Array, x), m)
to_host(x::Array) = x

to_gpu(x) = x |> gpu
# gpu_if_avail(x, use_cuda::Val{true}) = fmap(CuArrays.cu, x)
