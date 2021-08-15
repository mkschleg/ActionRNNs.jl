module ActionRNNs

using GVFHordes
import Reexport: @reexport
@reexport using MinimalRLCore

export glorot_uniform, glorot_normal, ExperienceReplay
include("Utils.jl")

export
    SingleLayer,
    Linear,
    deriv,
    sigmoid,
    sigmoid′

include("Layers.jl")
include("models/viz_backbone.jl")


export AARNN, MARNN, FacMARNN, AAGRU, MAGRU, FacMAGRU, reset!, get


include("RNNs.jl")

export QLearning, TD, update!
include("Update.jl")

include("ActingPolicy.jl")

include("Environments.jl")
include("Agent.jl")

include("exp_util.jl")


end # module
