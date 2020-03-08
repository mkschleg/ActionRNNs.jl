module ActionRNN

using MinimalRLCore

export
    SingleLayer,
    Linear,
    deriv,
    sigmoid,
    sigmoid′

include("Layers.jl")

export ARNN, reset!, get
include("RNN.jl")

# export RTD, RTD_jacobian, TDLambda, TD, update!
export TD, update!
include("Loss.jl")
include("Update.jl")

include("ActingPolicy.jl")

import Reproduce

export env_settings!, agent_settings!
env_settings!(as::Reproduce.ArgParseSettings, env_type) = throw("Settings not implemented for environment $(env_type)")
agent_settings!(as::Reproduce.ArgParseSettings, agent_type) = throw("Settings not implemented for agent $(agent_type)")

export step!, start!
include("Environments.jl")

export jacobian, glorot_uniform, glorot_normal
include("util.jl")

include("Agent.jl")

end # module
