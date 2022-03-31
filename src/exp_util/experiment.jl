

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
function get_static_args(::Val{:iterV2}, dict)
    filter(dict) do kv
        kv.first ∉ ["sweep_args", "config"]
    end
end

function Reproduce.get_arg_iter(::Val{:iterV2}, dict)

    static_args_dict = get_static_args(Val(:iterV2), dict)
    cdict = dict["config"]
    
    arg_order = get(cdict, "arg_list_order", nothing)

    @assert arg_order isa Nothing || all(sort(arg_order) .== sort(collect(keys(dict["sweep_args"]))))
    
    sweep_args_dict = dict["sweep_args"]
    
    for key ∈ keys(sweep_args_dict)
        if sweep_args_dict[key] isa String
            sweep_args_dict[key] = eval(Meta.parse(sweep_args_dict[key]))
        end
    end

    ArgIteratorV2(sweep_args_dict,
                  static_args_dict,
                  arg_order=arg_order)
end


struct ArgIteratorV2 <: Reproduce.AbstractArgIter
    sweep_args::Dict
    static_args::Dict{String, Any}
    arg_order::Vector{String}
    done::Bool
end

function ArgIteratorV2(sweep_args, static_arg::Dict=Dict(); arg_order=nothing)
    ArgIteratorV2(sweep_args,
                  Dict{String, Any}(static_arg),
                  arg_order==nothing ? collect(keys(sweep_args)) : arg_order,
                  false)
end

set_save_dir!(iter::ArgIteratorV2, path) = iter.static_args["save_dir"] = path


function set_argument!(d, args::Vector, v)
    if length(args) == 1
        set_argument!(d, args[1], v)
    elseif occursin("+", args[1])
        ks = split(args[1], "+")
        for (i, k) ∈ enumerate(ks)
            set_argument!(d[k], args[2:end], v[i])
        end
    else
        set_argument!(d[args[1]], args[2:end], v)
    end
end

function set_argument!(d, arg::AbstractString, v)
    if startswith(arg, "[") && endswith(arg, "]") && occursin(r"\[.*\]", arg) && !occursin(".", arg)
        str_idxs = findall(r"\[[0-9_a-z_A-Z_\__+]*\]")
        idxs = [isnothing(tryparse(Int, idx)) ? idx : parse(Int, idx)  for idx in str_idxs]
        set_argument!(d, idxs, v)
    elseif occursin(r"\[.*\]", arg)
        idx = findfirst("[", arg)[1]
        set_argument!(d[arg[1:idx-1]], arg[idx:end], v)
    elseif occursin(".", arg)
        sarg = split(arg, ".")
        set_argument!(d, sarg, v)
    elseif occursin("+", arg)
        ks = split(arg, "+")
        for (i, k) ∈ enumerate(ks)
            d[k] = v[i]
        end
    else
        d[arg] = v
    end
end

function make_arguments(iter::ArgIteratorV2, state)
    d = Dict{String, Any}()
    d = deepcopy(iter.static_args)
    for (arg_idx, arg) in enumerate(iter.arg_order)
        set_argument!(d, arg, iter.sweep_args[arg][state[2][arg_idx]])
    end
    d
end

function Base.iterate(iter::ArgIteratorV2)
    state = (1, ones(Int64, length(iter.arg_order)))
    arg_list = make_arguments(iter, state)
    return (state[1], arg_list), next_state(iter, state)
end

function next_state(iter::ArgIteratorV2, _state)
    state = _state[2]
    n_state = _state[1]

    state[end] += 1

    for (arg_idx, arg) in Iterators.reverse(enumerate(iter.arg_order))
        if arg_idx == 1
            return (n_state+1, state)
        end

        if state[arg_idx] > length(iter.sweep_args[arg])
            state[arg_idx] = 1
            state[arg_idx - 1] += 1
        end
    end
end

function Base.iterate(iter::ArgIteratorV2, state)
    if state[2][1] > length(iter.sweep_args[iter.arg_order[1]])
        return nothing
    end
    arg_list = make_arguments(iter, state)
    return (state[1], arg_list), next_state(iter, state)
end

function Base.length(iter::ArgIteratorV2)
    return *([length(iter.sweep_args[key]) for key in iter.arg_order]...)
end
