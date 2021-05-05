
using Reproduce, Plots, RollingFunctions, Statistics, FileIO, PlutoUI


function mean_uneven(d::Vector{Array{F, 1}}) where {F}
    ret = zeros(F, maximum(length.(d)))
    n = zeros(Int, maximum(length.(d)))
    for v ∈ d
        ret[1:length(v)] .+= v
        n[1:length(v)] .+= 1
    end
    ret ./ n
end

function std_uneven(d::Vector{Array{F, 1}}) where {F}
    
    m = mean_uneven(d)
    
    ret = zeros(F, maximum(length.(d)))
    n = zeros(Int, maximum(length.(d)))
    for v ∈ d
        ret[1:length(v)] .+= (v .- m[1:length(v)]).^2
        n[1:length(v)] .+= 1
    end
    sqrt.(ret ./ n)
end

function new_search(f, ic::ItemCollection)

    found_items = Array{Reproduce.Item, 1}()
    for (item_idx, item) in enumerate(ic.items)
        if f(item)
            push!(found_items, item)
        end
    end
    return ItemCollection(found_items)

end

function get_agg(agg, ddict, key)
    agg(ddict["results"][key])
end

get_mean(ddict, key) = get_agg(mean, ddict, key)
get_AUC(ddict, key) = get_agg(sum, ddict, key)
get_AUE(ddict, key, perc=0.1) = get_agg(ddict, key) do x
    sum(x[end-max(1, Int(floor(length(x)*perc))):end])
end

function get_rolling_mean_line(ddict, key, n)
    if n > length(ddict["results"][key])
        n = length(ddict["results"][key])
    end
    rollmean(ddict["results"][key], n)
end

function get_runs(ic, get_data)
    tmp = get_data(FileIO.load(joinpath(ic[1].folder_str, "results.jld2")))
    d = typeof(tmp)[]
    for (idx, item) ∈ enumerate(ic)
        push!(d, get_data(FileIO.load(joinpath(item.folder_str, "results.jld2"))))
    end
    d
end

function internal_func(ic, 
	               param_keys;
		       comp=findmax,
		       get_comp_data, 
		       get_data=get_comp_data)
    
    ic_diff = diff(ic)
    params = if param_keys isa String
        ic_diff[param_keys]
    else
        collect(Iterators.product([ic_diff[k] for k ∈ param_keys]...))
    end
    
    s = zeros(length(params))
    for (p_idx, p) ∈ enumerate(params)
        sub_ic = if param_keys isa String
            search(ic, Dict(param_keys=>p))
        else
            search(ic, Dict(param_keys[i]=>p[i] for i ∈ 1:length(p)))
        end
	
        s[p_idx] = mean(get_runs(sub_ic, get_comp_data))
    end
    
    v, idx = comp(s)
    
    best_ic = if param_keys isa String
	search(
	    ic, 
	    Dict(param_keys=>params[idx]))
    else
	search(
	    ic, 
	    Dict(param_keys[i]=>params[idx][i] for i ∈ 1:length(params[idx])))

    end

    data = get_runs(best_ic, get_data)
    @show params[idx]
    data, v, params[idx]
end


struct LineData{LP, SP, D, C}
    line_params::LP
    swept_params::SP
    data::D
    c::C
end

Base.show(io::IO, ld::LineData) = print(io, "LineData(", ld.line_params, ", ", ld.swept_params, ", ", ld.c, ")")

function get_line_data_for(
    ic::ItemCollection, 
    line_keys, 
    param_keys; 
    comp=findmax,
    get_comp_data,
    get_data)
    ic_diff = diff(ic)
    params = if line_keys isa String
        ic_diff[line_keys]
    else
        collect(Iterators.product([ic_diff[k] for k ∈ line_keys]...))
    end
    
    strg = LineData[]
    
    Threads.@threads for p_idx ∈ 1:length(params)
	p = params[p_idx]

	sub_ic = if line_keys isa String
            search(ic, Dict(line_keys=>p))
        else
            search(ic, Dict(line_keys[i]=>p[i] for i ∈ 1:length(p)))
        end
        if !isempty(sub_ic)
	    d, c, ps = internal_func(
	        sub_ic, 
	        param_keys;
	        get_comp_data=get_comp_data, 
	        get_data=get_data)
	    push!(strg, LineData(params[p_idx], ps, d, c))
        end
    end
    strg
end

plot_line_from_data_with_params(data_col, params; pkwargs...) =
    plot_line_from_data_with_params!(nothing, data_col, params; pkwargs...)

function plot_line_from_data_with_params!(plt, data_col, params; pkwargs...)
    idx = findfirst(data_col) do (line_params, sweep_params, datum, comp_val)
	all([line_params[i] == params[i] for i ∈ 1:length(line_params)])
    end
    d = data_col[idx]
    label = if :label ∈ keys(pkwargs)
	"$(pkwargs[:label]), $(d[2])"
    else
	d[2]
    end
    if plt isa Nothing
	plt = plot(mean_uneven(d[3]), ribbon=std_uneven(d[3]); pkwargs..., label=label)
    else
	plot!(plt, mean_uneven(d[3]), ribbon=std_uneven(d[3]); pkwargs..., label=label)
    end
    plt
end

function plot_line_from_data_with_params!(plt, data_col::Vector{LineData}, params; pkwargs...)
    idx = findfirst(data_col) do (ld)
        line_params = ld.line_params
	all([line_params[i] == params[i] for i ∈ 1:length(line_params)])
    end
    d = data_col[idx]
    label = if :label ∈ keys(pkwargs)
	"$(pkwargs[:label]), $(d[2])"
    else
	d[2]
    end
    if plt isa Nothing
	plt = plot(mean_uneven(d[3]), ribbon=std_uneven(d[3]); pkwargs..., label=label)
    else
	plot!(plt, mean_uneven(d[3]), ribbon=std_uneven(d[3]); pkwargs..., label=label)
    end
    plt
end
