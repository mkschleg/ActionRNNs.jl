
module ExpUtils

using ..ActionRNNs

# Specific Utilityies for environments
# include("utils/compassworld.jl")
include("exp_util/cycleworld.jl")
include("exp_util/ringworld.jl")
include("exp_util/tmaze.jl")
include("exp_util/lunar_lander.jl")

# Experiment utilities
include("exp_util/experiment.jl")
include("exp_util/simple_logger.jl")
include("exp_util/flux.jl")

include("exp_util/macros.jl")

end
