

using Flux
import Random

glorot_uniform(args...; kwargs...) = glorot_uniform(Random.GLOBAL_RNG, args...; kwargs...)
glorot_normal(args...; kwargs...) = glorot_normal(Random.GLOBAL_RNG, args...; kwargs...)

function glorot_uniform(rng::Random.AbstractRNG, dims...; ignore_dims=0)
    if ignore_dims == 0
        (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
    else
        (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/(sum(dims) - sum(dims[ignore_dims])))
    end
end

function glorot_normal(rng::Random.AbstractRNG, dims...; ignore_dims=0)
    if ignore_dims == 0
        randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))
    else
        (randn(rng, Float32, dims...)) .* sqrt(2.0f0/(sum(dims) - sum(dims[ignore_dims])))
    end
end


# Need to deal with some of the issues w/ using Flux layers + ActionRNN layers.
(l::Flux.Dense)(x::Tuple) = (x[1], l(x[2]))
(l::Flux.Conv)(x::Tuple) = (x[1], l(x[2]))
Flux.flatten(x::Tuple) = (x[1], Flux.flatten(x[2]))

function update_target_network!(model, target_network)
    for ps ∈ zip(params(model),
                 params(target_network))
        copyto!(ps[2], ps[1])
    end
end

include("utils/device.jl")
include("utils/replay.jl")
include("utils/state_buffer.jl")

export ImageReplay
include("utils/image_replay.jl")
