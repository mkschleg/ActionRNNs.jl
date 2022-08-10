module ItemColToDataFrame

using Reproduce
using FileIO
using DataFrames
using ProgressLogging


get_parg(key, arg) = [key=>arg]
get_parg(key, arg::AbstractVector) = collect(Iterators.flatten([get_parg(key*"_$(i)", arg[i]) for i in eachindex(arg)]))
get_parg(key, arg::AbstractDict) = collect(Iterators.flatten([get_parg(key*"_$(k)", arg[k]) for k in keys(arg)]))

function create_pargs(args, fkeys = keys(args))

    pargs = Pair{String}[]
    for k in fkeys
        append!(pargs, get_parg(k, args[k]))
    end
    pargs
end


function proc_to_data_frame(proc::Function, dir_or_ic)
    item_col = if dir_or_ic isa ItemCollection
        dir_or_ic
    else
        ItemCollection(dir)
    end


    rows = []
    @progress for j in 1:length(item_col)
        i = item_col[j]
        args = i.parsed_args
        if "_HASH" ∉ keys(args)
            args["_HASH"] = parse(UInt64, split(i.folder_str, "_")[end])
        end
        if "_GIT_INFO" ∉ keys(args)
            args["_GIT_INFO"] = parse(UInt64, split(i.folder_str, "_")[end-1])
        end
        fkeys = filter(
            (a)->a ∉ ["_HASH", "_GIT_INFO", "_SAVE"],
            sort(collect(keys(args))))
        fkeys = [fkeys; ["_HASH", "_GIT_INFO"]]

        try
            res = FileIO.load(joinpath(i.folder_str, "results.jld2"))
            data = proc(res)
            pargs = create_pargs(args, fkeys)
            push!(rows, [pargs; data])
        catch
        end
    end

    df = DataFrame(rows[1])
    for r in rows[2:end]
        append!(df, r)
    end
    
    df
end


end
