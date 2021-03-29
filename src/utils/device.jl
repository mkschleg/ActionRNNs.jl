using Flux, Adapt


abstract type Device end

(device::Device)(x) = to_device(device, x)

struct CPU <: Device end
struct GPU <: Device end

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


to_device(device::CPU, x) = to_host(x)
to_device(device::GPU, x) = to_gpu(x)

to_device(use_cuda::Val{:cpu}, x) = to_host(x)
to_device(use_cuda::Val{:gpu}, x) = to_gpu(x)


to_host(m) = fmap(x -> adapt(Array, x), m)
to_host(x::Array) = x

to_gpu(x) = x |> gpu
# gpu_if_avail(x, use_cuda::Val{true}) = fmap(CuArrays.cu, x)
