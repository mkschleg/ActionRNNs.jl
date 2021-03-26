

using Flux
import Random

glorot_uniform(rng::Random.AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))

# Need to deal with some of the issues w/ using Flux layers + ActionRNN layers.
(l::Flux.Dense)(x::Tuple) = (x[1], l(x[2]))
(l::Flux.Conv)(x::Tuple) = (x[1], l(x[2]))
Flux.flatten(x::Tuple) = (x[1], Flux.flatten(x[2]))

include("utils/device.jl")
include("utils/replay.jl")
include("utils/state_buffer.jl")
