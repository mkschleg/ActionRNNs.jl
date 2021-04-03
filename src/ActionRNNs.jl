module ActionRNNs

using GVFHordes
import Reexport: @reexport
@reexport using MinimalRLCore

export glorot_uniform, glorot_normal, ExperienceReplay
include("util.jl")

export
    SingleLayer,
    Linear,
    deriv,
    sigmoid,
    sigmoid′

include("Layers.jl")
include("models/viz_backbone.jl")


export AARNN, MARNN, FacARNN, reset!, get
include("rnns.jl")
# include("RNN.jl")
# include("LSTM.jl")
# include("GRU.jl")

export QLearning, TD, update!
include("Loss.jl")
include("Update.jl")

include("ActingPolicy.jl")

include("Environments.jl")
include("Agent.jl")


include("exp_util.jl")


end # module
