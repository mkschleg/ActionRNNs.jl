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

export ARNN, ALSTM, reset!, get
include("RNN.jl")
include("LSTM.jl")

# export RTD, RTD_jacobian, TDLambda, TD, update!
export TD, update!
include("Loss.jl")
include("Update.jl")

include("ActingPolicy.jl")


include("Environments.jl")

export jacobian, glorot_uniform, glorot_normal
include("util.jl")

include("Agent.jl")

end # module
