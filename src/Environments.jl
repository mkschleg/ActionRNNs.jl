

import MinimalRLCore
# using RLCore: AbstractEnvironment, step!, start!

export CompassWorld, get_num_features
include("env/CompassWorld.jl")

export CycleWorld
include("env/CycleWorld.jl")

export RingWorld, RingWorldConst
include("env/RingWorld.jl")

export TMaze
include("env/TMaze.jl")

export VariableTMaze
include("env/VariableTMaze.jl")

export ImageTMaze
include("env/ImageTMaze.jl")

export DirectionalTMaze
include("env/DirectionalTMaze.jl")

export ImageDirTMaze
include("env/ImageDirectionalTMaze.jl")

export LunarLander
include("env/LunarLander.jl")

export Torus2d
include("env/Torus2d.jl")

export MaskedGridWorld
include("env/MaskedGridWorld.jl")

export LinkedChains
include("env/LinkedChains.jl")
