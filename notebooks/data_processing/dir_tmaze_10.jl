### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 278b9e82-43da-4670-ac5e-9cec326689ca
begin
	using Reproduce
	using ReproducePlotUtils
	using DataFrames
	using StatsPlots
	using FileIO
	using Statistics
	using InvertedIndices # Not(1)
	using PlutoUI
end

# ╔═╡ 9bb2ff09-3364-4f31-9d07-29ffa47d716a
const RPU = ReproducePlotUtils

# ╔═╡ 8aa844f8-ec57-47d3-9745-86fc8482c826
md"""
# Data Analysis Notebook.

Hopefully this can be used as a better template for making plots in the future.
"""

# ╔═╡ 3175d098-1e6c-4ad5-872c-ce1f6e49409a
md"""
Choose location of data:
"""

# ╔═╡ 71cff3f1-3b58-42d9-a1c9-acbfba9a5e2d
local_data = "../local_data"

# ╔═╡ e2db63ec-9ba7-472c-bca0-bf8892f134c1
exp_folders, deeper_folders = let
	folders = readdir(local_data)
	filter!(folders) do f
		splitext(f)[end] != ".gz" && 
		splitext(f)[end] != ".tar" && 
		splitext(f)[end] != ".pdf" && 
		splitext(f)[end] != "tar.gz"
	end
	exp_folders = folders[isdir.(joinpath.((local_data,), folders, ("data",)))]
	deeper_folders = folders[Not(isdir.(joinpath.((local_data,), folders, ("data",))))]
	push!(deeper_folders, ".")
	exp_folders, sort!(deeper_folders)
end;

# ╔═╡ 6d62de45-71f0-4783-bea3-9224ac2499e0
md"""
view experiment folders: 
$(@bind view_folder PlutoUI.Select(deeper_folders))

"""

# ╔═╡ a54d5cf6-7725-497e-b6c5-530fbc49cfbf
let
	if view_folder == "."
		exp_folders
	else
		deeper_local_data = joinpath(local_data, view_folder)
		folders = readdir(deeper_local_data)
		filter!(folders) do f
			splitext(f)[end] != ".gz" && 
			splitext(f)[end] != ".tar" && 
			splitext(f)[end] != ".pdf" && 
			splitext(f)[end] != "tar.gz"
		end
		folders[isdir.(joinpath.((deeper_local_data,), folders, ("data",)))]
	end
end

# ╔═╡ 3c3aa346-3911-4a73-b1e4-11ad8c9cccc4
md"view processed data folder"

# ╔═╡ 36f540ea-b945-4b2f-825f-f97a7ce2fb9b
readdir("../processed_data/")

# ╔═╡ f999a721-f26e-4cc5-8cc1-1732bb69b134
begin
	load_at = local_data
	save_at = "../processed_data/dir_tmaze_10.jld2"
	results_folders = ["dir_tmaze_er/dir_tmaze_er_rnn_rmsprop_10/", 
				   	   "dir_tmaze_er/dir_tmaze_er_rnn_rmsprop_10_p2/"]
	settings_filter = ["_GIT_INFO", "_SAVE", "save_dir"]
end;

# ╔═╡ 67d58fe0-6b1f-43a9-8c48-321bd9c5c518
begin
	line_params = ["numhidden", "truncation", "cell"]
	sweep_params = ["eta"]
end;

# ╔═╡ 36bc2210-9b37-48f5-b0b1-7089576e1a64
if isfile(save_at)
	cur_save = FileIO.load(save_at)
	md"""
	The current data at **$(save_at)** uses:
	
	**Comparitor**: $(cur_save["comparedirection"]), 
	**Get Data**: $(cur_save["compare_get"]),
	**Key**: $(cur_save["compare_key"]),
	**Custom?**: $(cur_save["custom_compare"]) \
	
	**CompareFunc**: $(cur_save["cust_compare_code"])
	
	**LineParams**: $(string(cur_save["line_params"])) \
	**SweepParams**: $(string(cur_save["sweep_params"]))
	"""
else
	"No Save file at current location"
end

# ╔═╡ 9769195c-5075-4ad7-af6c-572dc433ca3a
cust_compare_code = quote
	error("Not Implemented")
end

# ╔═╡ a211294b-852d-447f-b538-66b50c531374
md"""
### Processing the best settings:

Implement the below function to process the best settings as found by the compare.
"""

# ╔═╡ 759eca80-b113-4d88-887e-6c762016ad72
function get_data(x)
	(all=RPU.get_rolling_mean_line(x, :successes, 300), 
	 mue=RPU.get_MUE(x, :successes))
end

# ╔═╡ 607a92d2-5d70-4ff5-89a9-9a0fb638a3db
begin
	results_folders, save_at, load_at
	md"""
	#### Saving data
	save? $(@bind save_results_df CheckBox()) force new save? $(@bind force_save CheckBox(false))

	"""
end	

# ╔═╡ c24aa3ab-d9de-47c9-b4b0-6b642e2333f0
md"""
#### Inspect

Which data frame? $(@bind inspect_df PlutoUI.Select(["settings", "processed"]))
"""

# ╔═╡ 65a3b91e-5a3f-4db9-a89e-017b893a7a38
md"#### Loading stuff"

# ╔═╡ 31bf6622-1a98-41fe-b7fc-5d66be759f65
md"""
# Appendix 

Eventually we will want to move these out of this notebook and into RPU
"""

# ╔═╡ f54091b2-795c-417c-8baa-2e05d0edbde6
@eval function custom_compare(x)
	$cust_compare_code
end

# ╔═╡ e5794ea6-5a5f-49ea-98c3-7d4b28c48dd3
function ic_to_df(ic::Reproduce.ItemCollection; filter=["_GIT_INFO", "_SAVE", "save_dir"])
	
	dlist = [merge(item.parsed_args, Dict("folder_str"=>item.folder_str)) for item in ic]
	doa = Dict()
	for k in keys(dlist[1])
		if filter isa Nothing || !(k in filter)
			doa[k] = [datum[k] for datum in dlist]
		end
	end
	DataFrame(doa)
end

# ╔═╡ 6893793b-20be-4907-8b3c-3e8143ee40f8
function files_to_ic_to_df(files::Vector{String}; kwargs...)
	ic_arr = [ItemCollection(at(file)) for file in files]
	ic_to_df(ItemCollection(vcat(getfield.(ic_arr, :items)...)); kwargs...)
end

# ╔═╡ e68631ea-1488-4afe-9996-ccbb8e16ded9
function load_runs(df::DataFrame, get_data)
	
    tmp = get_data(FileIO.load(joinpath(df[1, :folder_str], "results.jld2")))
    d = typeof(tmp)[]
	
	diff_arr = [k=>unique(sort(col)) for (k, col) in zip(names(df), eachcol(df))]
	diff_dict = Dict(diff_arr[[(length(a[2]) > 1) for a in diff_arr]])
	
	pms = @static if VERSION <= v"1.5"
		(; Symbol(k)=>df[:, k] for k ∈ keys(diff_dict))
	else
		NamedTuple(Symbol(k)=>df[:, k] for k ∈ keys(diff_dict))
	end
	
    for (idx, row) ∈ enumerate(eachrow(df))
        push!(d, get_data(FileIO.load(joinpath(row.folder_str, "results.jld2"))))
    end
    d, pms
end

# ╔═╡ d635d06a-8f39-4306-bcf3-de77664394c9
function find_best_params(df::DataFrame,
	                  	  param_keys,
		          		  comp,
		          		  get_comp_data,
		          		  get_data=RPU.get_comp_data)

    comp_func = if comp isa Symbol
        RPU.get_comp_function(comp)
	else
		comp
    end

	params = if param_keys isa String
		unique(sort!(df[:, param_keys]))
	else
		collect(Iterators.product([unique(sort!(df[:, k])) for k in param_keys]...))
	end

    values = zeros(length(params))
    for (p_idx, p) ∈ enumerate(params)
		df_view = filter(df; view=false) do row
			all([row[param_keys[i]] == p[i] for i ∈ 1:length(p)])
		end
        values[p_idx] = mean(load_runs(df_view, get_comp_data)[1])
    end

    v, idx = comp_func(values)
	
	df_view = filter(df; view=false) do row
		all([row[pk] == params[idx][i] for (i, pk) ∈ enumerate(param_keys)])
	end
	
    data, data_pms = load_runs(df_view, get_data)

	data, data_pms, v, params[idx], values, params
	
end

# ╔═╡ 51851161-5cde-4011-856f-2da11e398e21
function get_data_for(
		get_data::Function,
    	df::DataFrame,
    	line_keys,
    	param_keys;
    	comp,
    	get_comp_data)

	ks = union(line_keys, param_keys)
	nsks = filter!((k)->k ∉ ks, names(df))

	params_outer, params_inner, kouter, kinner = if line_keys isa String
		unique(sort!(df[:, line_keys])), nothing
	else
		ps = [unique(sort!(df[:, k])) for k in line_keys]
		_, idx = findmax(length.(ps))
		iter = Iterators.product(ps[Not(idx)]...)
		ps[idx], iter, line_keys[idx], line_keys[Not(idx)]
	end
	
	df_ret_th = Vector{DataFrame}(undef, Threads.nthreads())
    
    Threads.@threads for po ∈ params_outer
		tid = Threads.threadid()
			
		for pi in params_inner
			df_view = filter(df; view=false) do row
				row[kouter] == po && all(row[k] == pi[i] for (i, k) ∈ enumerate(kinner))
			end
			
			if !isempty(df_view)
				fbp_ret = find_best_params(df_view,
				  					   	   param_keys,
										   comp,
										   get_comp_data,
										   get_data)
				data, data_params, v_best, vb_param, values, value_pms = fbp_ret
				df_viewier = filter(df_view) do row
					all(row[k] == vb_param[i] for (i, k) in enumerate(param_keys))
				end

				d_kinner = Dict(k=>pi[i] for (i, k) in enumerate(kinner))
				d_kinner[kouter] = po
				d_sk_b = Dict(param_keys[i]=>vb_param[i] 
					for i in 1:length(param_keys))
				d_sk = Dict("sweep_"*param_keys[i]=>getindex.(value_pms, i)
					for i in 1:length(param_keys))
				d_nsk = Dict(k => let
						if Symbol(k) ∉ keys(data_params)
							ret_u = unique(sort!(df_viewier[:, k]))
							if length(ret_u) == 1
								ret_u[1]
							else
								[ret_u]
							end
						else
							data_params[Symbol(k)]
						end
					end for k in nsks)
				d = merge(d_kinner, d_sk, d_sk_b, d_nsk)
				
				if eltype(data) <: AbstractDict || eltype(data) <: NamedTuple
					for k in keys(data[1])
						d["data_" * string(k)] = getindex.(data, (k, ))
					end
				else
					d["data"] = data
				end
				d["sweep_value_best"] = v_best
				d["sweep_values"] = values

				if !isassigned(df_ret_th,tid)
					df_ret_th[tid] = DataFrame((k=>Vector{typeof(d[k])}() for k in keys(d))...)
				end
				push!(df_ret_th[tid], d)
			end
		end
	end
	as_df_ret_th = df_ret_th[isassigned.([df_ret_th], 1:length(df_ret_th))]
	if length(as_df_ret_th) != 1
		for _df in as_df_ret_th[2:end]
			append!(as_df_ret_th[1], _df)
		end
	end
	as_df_ret_th[1]
end

# ╔═╡ cfeb98f2-a4e0-4218-bb95-947f76e9d09b
function datacol_to_df(dc::RPU.DataCollection, sp::Vector{String})
	dict = Dict()
	for k in keys(dc[1].line_params)
		dict[k] = [data.line_params[k] for data in dc.data]
	end
	for (i, k) in enumerate(sp)
		dict[k] = [data.swept_params[i] for data in dc.data]
	end
	dict["z_data"] = [data.data for data in dc.data]
	dict["z_data_pms"] = [data.data_pms for data in dc.data]
	dict["z_c"] = [data.c for data in dc.data]
	
	DataFrame(dict)
end

# ╔═╡ 74fce7a9-bbd2-40b1-976e-3cc1e8c7a72b
function build_dataframe(folders::Vector{String}; kwargs...)
	dfs = [build_dataframe(folder; kwargs...) for folder in folders]
	for i in 2:length(dfs)
		append!(dfs[1], dfs[i])
	end
	dfs[1]
end

# ╔═╡ 685dd309-710c-431c-9591-80afd57e9ac2
function build_dataframe(folder::String; 
						 filter=nothing, 
						 force=false, 
						 settings_file="settings.jld2")
    
	dir = splitpath(folder)[end] == "data" ? folder : joinpath(folder, "data")
    dir_list = readdir(dir)

    cache_loc = joinpath(dir, "data_frame.jld2")
    id = hash(string(dir_list))
    if isfile(cache_loc)
        data = FileIO.load(cache_loc)
        if id == data["id"] && !force
            return data["data"]
        end
    end

    # df = DataFrame()
    doa = Dict()
    item = Reproduce.Item(joinpath(dir, dir_list[1], settings_file))
    dlist = merge(item.parsed_args, Dict("folder_str"=>item.folder_str))
    
    for (k,v) in dlist
        if isnothing(filter) || k ∉ filter
            doa[k] = [v]
        end
    end
    
    for p in dir_list
		if basename(p) ∈ ["item_col.jld2", "data_frame.jld2"]
			continue
		end
        item = Reproduce.Item(joinpath(dir, p, settings_file))
        dlist = merge(item.parsed_args, Dict("folder_str"=>item.folder_str))
        for k in keys(doa)
           push!(doa[k], dlist[k])
        end
    end

    df = DataFrame(doa)
    FileIO.save(cache_loc, "data", df, "id", id)
    
    return df
end

# ╔═╡ faa9e0f2-ba96-42cf-b8be-3b2387cbdc3c
settings_df = build_dataframe(joinpath.((load_at,), results_folders); 
							  filter=settings_filter);

# ╔═╡ e4f91cd2-f9d1-4434-af07-614e6d7fc27c
begin

	results_keys = collect(keys(FileIO.load(joinpath(settings_df[1, :folder_str], "results.jld2"))["results"]))	
	
	md"""
	#### Comparing Data
	**A(M)UC**: Area (mean) under the curve, **A(M)UE**: Area (mean) under end
	
	Compare:
	$(@bind comparedirection PlutoUI.Select(["max", "min"], default="max"))
	Get data:
	$(@bind compare_get PlutoUI.Select(["AUC", "MUC", "AUE", "MUE"], default="MUE"))
	Compare key:
	$(@bind compare_key PlutoUI.Select(string.(results_keys)))
	custom compare?: $(@bind cust_compare PlutoUI.CheckBox(false))
	"""
	
end

# ╔═╡ 0cb55aa5-0b84-432a-9235-4d9db6b86bf2
if cust_compare
	md"Please implement `custom_compare`"
elseif compare_get == "AUE" || compare_get == "MUE"
	md"""
	Please specify the percentage of data: $(@bind comp_perc PlutoUI.NumberField(0.00: 0.1 : 1.00, default=0.1))
	"""
end

# ╔═╡ 77b723ce-fa57-4050-9616-20433de02ca2
function get_comp_data(x)
	key = eltype(results_keys)(compare_key)
	if cust_compare
		return custom_compare(x)
	end
	if compare_get == "AUC"
		RPU.get_AUC(x, key)
	elseif compare_get == "MUC"
		RPU.get_MEAN(x, key)
	elseif compare_get == "AUE"
		RPU.get_MUE(x, key, comp_perc)
	elseif compare_get == "MUE"
		RPU.get_MUE(x, key, comp_perc)
	end
end

# ╔═╡ 5a03146e-5264-423b-ad14-b0dae69b0fff
data_df = get_data_for(settings_df,
					line_params,
					sweep_params;
					comp=Symbol(comparedirection),
					get_comp_data=get_comp_data) do x
	get_data(x)
end;

# ╔═╡ a6fcd030-9486-409b-89df-1c0e1e43c76b
if save_results_df
	if isfile(save_at) && !force_save
		md"File $(save_at) already exists. Either force or use a different file name"
	else
		FileIO.save(save_at, 
					"data", data_df, 
					"comparedirection", comparedirection, 
					"compare_get", compare_get, 
					"compare_key", compare_key, 
					"custom_compare", cust_compare, 
					"cust_compare_code", cust_compare_code,
					"line_params", line_params,
					"sweep_params", sweep_params,
					"results_folders", results_folders)
		md"Saved Successfully ✅"
	end
else
	md"Saving turned off"
end

# ╔═╡ 420a1872-873e-4270-adc2-d01f32dc2003
if inspect_df == "settings"
	settings_df
else
	data_df
end

# ╔═╡ Cell order:
# ╠═278b9e82-43da-4670-ac5e-9cec326689ca
# ╠═9bb2ff09-3364-4f31-9d07-29ffa47d716a
# ╟─8aa844f8-ec57-47d3-9745-86fc8482c826
# ╟─3175d098-1e6c-4ad5-872c-ce1f6e49409a
# ╠═71cff3f1-3b58-42d9-a1c9-acbfba9a5e2d
# ╟─e2db63ec-9ba7-472c-bca0-bf8892f134c1
# ╟─6d62de45-71f0-4783-bea3-9224ac2499e0
# ╟─a54d5cf6-7725-497e-b6c5-530fbc49cfbf
# ╟─3c3aa346-3911-4a73-b1e4-11ad8c9cccc4
# ╟─36f540ea-b945-4b2f-825f-f97a7ce2fb9b
# ╠═f999a721-f26e-4cc5-8cc1-1732bb69b134
# ╟─e4f91cd2-f9d1-4434-af07-614e6d7fc27c
# ╟─0cb55aa5-0b84-432a-9235-4d9db6b86bf2
# ╠═67d58fe0-6b1f-43a9-8c48-321bd9c5c518
# ╟─36bc2210-9b37-48f5-b0b1-7089576e1a64
# ╠═9769195c-5075-4ad7-af6c-572dc433ca3a
# ╟─a211294b-852d-447f-b538-66b50c531374
# ╟─759eca80-b113-4d88-887e-6c762016ad72
# ╟─607a92d2-5d70-4ff5-89a9-9a0fb638a3db
# ╠═a6fcd030-9486-409b-89df-1c0e1e43c76b
# ╟─c24aa3ab-d9de-47c9-b4b0-6b642e2333f0
# ╟─420a1872-873e-4270-adc2-d01f32dc2003
# ╟─65a3b91e-5a3f-4db9-a89e-017b893a7a38
# ╠═faa9e0f2-ba96-42cf-b8be-3b2387cbdc3c
# ╠═5a03146e-5264-423b-ad14-b0dae69b0fff
# ╟─31bf6622-1a98-41fe-b7fc-5d66be759f65
# ╠═f54091b2-795c-417c-8baa-2e05d0edbde6
# ╠═77b723ce-fa57-4050-9616-20433de02ca2
# ╠═e5794ea6-5a5f-49ea-98c3-7d4b28c48dd3
# ╠═6893793b-20be-4907-8b3c-3e8143ee40f8
# ╠═51851161-5cde-4011-856f-2da11e398e21
# ╠═d635d06a-8f39-4306-bcf3-de77664394c9
# ╠═e68631ea-1488-4afe-9996-ccbb8e16ded9
# ╠═cfeb98f2-a4e0-4218-bb95-947f76e9d09b
# ╠═74fce7a9-bbd2-40b1-976e-3cc1e8c7a72b
# ╠═685dd309-710c-431c-9591-80afd57e9ac2
