module DataFrameUtils

using DataFrames, Query

collapse_group(grp_df) = collapse_group_agg(grp_df) do c
    [collect(c)]
end

function collapse_group_agg(agg::Function, grp_df)
    values = []
    nms = names(grp_df)
    for n in nms
        vs = unique(grp_df[!, Symbol(n)])
        if length(vs) != 1
            push!(values, agg(skipmissing(grp_df[!, n])))
        else
            push!(values, vs[1])
        end
    end
    DataFrame(;(Symbol(n)=>v for (n, v) in zip(nms, values))...)
end

function simplify_dataframe(agg::Function, df; special_keys=["_HASH", "_GIT_INFO", "seed"])
    nms = filter((d)->d ∉ special_keys, names(df, Between(1, :_GIT_INFO)))
    grps = groupby(df, Symbol.(nms))
    reduce(vcat, [collapse_group_agg(agg, grp) for grp in grps])
end

function get_diff_dict(df)
    Dict(filter((d)->length(d.second) != 1, [n=>unique(df[!, n]) for n in names(df, Between(1, :_HASH))[1:end-1]]))
end


function best_from_sweep_param(o, df, sweep_params, avg_params=["seed"])
    nms = filter((d)->d ∉ avg_params && d ∉ sweep_params, names(df, Between(1, :_HASH))[1:end-1])
    grps = groupby(df, Symbol.(nms))
    reduce(vcat, [DataFrame(sort(grp, o)[1, :]) for grp in grps])
end

end
