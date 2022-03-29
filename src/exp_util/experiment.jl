

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


# custom config parser...

# function Reproduce.get_arg_iter(::Val{:iterV2}, dict)

#     # static_args_dict = get_static_args(dict)
#     cdict = dict["config"]
    
#     arg_order = get(cdict, "arg_list_order", nothing)

#     @assert arg_order isa Nothing || all(sort(arg_order) .== sort(collect(keys(dict["sweep_args"]))))
    
#     sweep_args_dict = dict["sweep_args"]
    
#     for key ∈ keys(sweep_args_dict)
#         if sweep_args_dict[key] isa String
#             sweep_args_dict[key] = eval(Meta.parse(sweep_args_dict[key]))
#         end
#     end

#     ArgIterator(sweep_args_dict,
#                 static_args_dict,
#                 arg_order=arg_order)
# end
