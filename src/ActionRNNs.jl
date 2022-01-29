module ActionRNNs

using GVFHordes
import Reexport: @reexport
@reexport using MinimalRLCore

export glorot_uniform, glorot_normal, ExperienceReplay
include("Utils.jl")

export AARNN, MARNN, FacMARNN, AAGRU, MAGRU, FacMAGRU, reset!, get
include("RNNs.jl")

include("Layers.jl")
include("models/viz_backbone.jl")
include("action_dense_layer.jl")

export QLearning, TD, update!
include("Update.jl")

include("ActingPolicy.jl")

include("Environments.jl")
include("Agent.jl")

include("exp_util.jl")
include("construct.jl")

end # module
