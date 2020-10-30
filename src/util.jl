

using Flux
# using Flux.Tracker
import Reproduce: ArgParseSettings, @add_arg_table

glorot_uniform(rng::Random.AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))


# Should we export the namespaces? I think not...
include("utils/compassworld.jl")
include("utils/cycleworld.jl")
include("utils/ringworld.jl")
include("utils/tmaze.jl")

include("utils/flux.jl")
include("utils/experiment.jl")
include("utils/replay.jl")

