module ActionRNNs

using Reexport

@reexport using MinimalRLCore

using GVFHordes

export
    SingleLayer,
    Linear,
    deriv,
    sigmoid,
    sigmoid′

include("Layers.jl")

include("RNNUtil.jl")

export ARNN, ALSTM, AGRU, reset!, get
include("RNN.jl")
include("LSTM.jl")
include("GRU.jl")

# export RTD, RTD_jacobian, TDLambda, TD, update!
export TD, update!
include("Loss.jl")
include("Update.jl")

include("ActingPolicy.jl")


include("Environments.jl")

export glorot_uniform, glorot_normal, ExperienceReplay
include("util.jl")

include("Agent.jl")


end # module
