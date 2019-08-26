module ActionRNN

export
    SingleLayer,
    Linear,
    deriv,
    sigmoid,
    sigmoid′

include("Layers.jl")

export
    GVF,
    # get, get!,
    cumulant,
    discount,
    policy,
    Horde,
    NullPolicy,
    PersistentPolicy,
    ConstantDiscount,
    StateTerminationDiscount,
    FeatureCumulant,
    PredictionCumulant,
    ScaledCumulant

include("GVF.jl")

export RNNActionLayer, reset!, get
include("RNN.jl")

export RTD, RTD_jacobian, TDLambda, TD, update!
include("Loss.jl")
include("Update.jl")

include("ActingPolicy.jl")

export step!, start!
include("Environments.jl")

export jacobian, glorot_uniform, glorot_normal, StopGradient
include("util.jl")

include("Agent.jl")

end # module
