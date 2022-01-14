

using JLD2
import ProgressMeter
# using Reproduce
using FileIO
import Reproduce

experiment_wrapper(exp_func, parsed; hash_exclude_save_dir=false, kwargs...) = Reproduce.experiment_wrapper(exp_func, parsed; hash_exclude_save_dir=hash_exclude_save_dir, kwargs...)
experiment_wrapper(exp_func::Function, parsed, working::Bool; kwargs...) = begin
    @warn "Working in experiment wrapper is deprecated." maxlog=1
    experiment_wrapper(exp_func, parsed; kwargs...)
end
