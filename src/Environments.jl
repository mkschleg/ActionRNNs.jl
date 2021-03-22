

import MinimalRLCore
# using RLCore: AbstractEnvironment, step!, start!

export CompassWorld, get_num_features
include("env/CompassWorld.jl")

export CycleWorld
include("env/CycleWorld.jl")

export RingWorld
include("env/RingWorld.jl")

export TMaze
include("env/TMaze.jl")

export VariableTMaze
include("env/VariableTMaze.jl")

# export ContFourRooms
# include("env/ContFourRooms.jl")
