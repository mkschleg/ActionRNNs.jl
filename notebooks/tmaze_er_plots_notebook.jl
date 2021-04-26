### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 422ec05e-9afc-11eb-08e1-3363ec8d184f
using Reproduce, Plots, RollingFunctions, Statistics, FileIO, PlutoUI

# ╔═╡ 43a96d19-04e6-41d0-9b09-f055f91ce8d0
gr(fmt=:png)

# ╔═╡ 4632fbfb-76b5-4d7e-b31f-eec453e768a2
data_loc = "../local_data"

# ╔═╡ 6b12bc96-ae5b-412b-bb66-ed053bc99b05
color_scheme = [
    colorant"#44AA99",
    colorant"#332288",
    colorant"#DDCC77",
    colorant"#999933",
    colorant"#CC6677",
    colorant"#AA4499",
    colorant"#DDDDDD",
	colorant"#117733",
	colorant"#882255",
	colorant"#88CCEE",
]

# ╔═╡ 5612fa62-7927-4e79-9f7e-985d037acd7c
function mean_uneven(d::Vector{Array{F, 1}}) where {F}
    ret = zeros(F, maximum(length.(d)))
    n = zeros(Int, maximum(length.(d)))
    for v ∈ d
        ret[1:length(v)] .+= v
        n[1:length(v)] .+= 1
    end
    ret ./ n
end

# ╔═╡ d4e4f2b3-4df5-4195-9595-3ca1a0da74c5
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

# ╔═╡ 10334c8c-e3ce-4c8c-8f6d-1fc966d0274c
function new_search(f, ic::ItemCollection)

    found_items = Array{Reproduce.Item, 1}()
    for (item_idx, item) in enumerate(ic.items)
        if f(item)
            push!(found_items, item)
        end
    end
    return ItemCollection(found_items)

end

# ╔═╡ 603037f6-c95b-44c1-889d-bdf9f18becc5
function get_runs(ic, get_data)
    tmp = get_data(FileIO.load(joinpath(ic[1].folder_str, "results.jld2")))
    d = typeof(tmp)[]
    for (idx, item) ∈ enumerate(ic)
        push!(d, get_data(FileIO.load(joinpath(item.folder_str, "results.jld2"))))
    end
    d
end

# ╔═╡ e615a5ff-7ede-4d83-b55a-a44cc5368d66
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
	
	# get data of best setting
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

# ╔═╡ 8ffcfda9-1dc2-4769-a880-26c447a4afbf
function get_agg(agg, ddict, key)
    agg(ddict["results"][key])
end

# ╔═╡ 0a58bee2-b126-41e1-a0a1-53021a8a0134
begin
	get_mean(ddict, key) = get_agg(mean, ddict, key)
	get_AUC(ddict, key) = get_agg(sum, ddict, key)
	get_AUE(ddict, key, perc=0.1) = get_agg(ddict, key) do x
		sum(x[end-max(1, Int(floor(length(x)*perc))):end])
	end

	get_mean_successes(ddict) = get_mean(ddict, :successes)
	get_AUC_successes(ddict) = get_AUC(ddict, :successes)
	get_AUC_successes(ddict) = get_AUE(ddict, :successes)
end

# ╔═╡ 2be7162b-ffe6-4c66-aea7-53877e7c0356
function get_rolling_mean_line(ddict, key, n)
    rollmean(ddict["results"][key], n)
end

# ╔═╡ 63736a0f-8d18-4689-8a1d-0ca73f380ead
function heatmap_data(ic::ItemCollection, x_axis="numhidden", y_axis="truncation", internal_func=internal_func)
    ic_diff = diff(ic)

    s = zeros(length(ic_diff[y_axis]), length(ic_diff[x_axis]))
    
    for (i, x) ∈ enumerate(ic_diff[x_axis])
        for (j, y) ∈ enumerate(ic_diff[y_axis])
            sic = search(ic, Dict(x_axis=>x, y_axis=>y))
            s[j,i] = internal_func(sic; get_comp_data=get_end_successes)[2]
        end
    end
    s
end

# ╔═╡ d1683162-6958-4736-a56e-53eb02235724
function get_line_data(ic::ItemCollection, param_keys="eta")
    ic_diff = diff(ic)
    d, c = internal_func(ic, get_runs, findmax, param_keys; 
        get_comp_data=get_successes, get_data=(d)->get_rolling_mean_line(d, :successes, 100))
end

# ╔═╡ b3d6cdfb-39d7-4b02-8150-4d6a6fe5ec7a
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
	
	strg = []
	
	Threads.@threads for p_idx ∈ 1:length(params)
		p = params[p_idx]

		sub_ic = if line_keys isa String
            search(ic, Dict(line_keys=>p))
        else
            search(ic, Dict(line_keys[i]=>p[i] for i ∈ 1:length(p)))
        end

		d, c, ps = internal_func(
			sub_ic, 
			param_keys;
			get_comp_data=get_comp_data, 
			get_data=get_data)
		push!(strg, (params[p_idx], ps, d, c))
	end
	strg
end

# ╔═╡ 0deda36e-b4cd-4c7b-8548-f4cdcb5d1611
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

# ╔═╡ 325835b8-4f99-408d-a9d0-8efedb98d669
plot_line_from_data_with_params!(data_col, params; pkwargs...) = 
	plot_line_from_data_with_params!(nothing, data_col, params; pkwargs...)

# ╔═╡ 011f9607-26c5-4839-b311-df41c581e480
md"""
# TMAZE Size=6
"""

# ╔═╡ 210cf7b1-5450-42cd-8f9b-1ab580dd16ba
begin
	ic_6 = ItemCollection(joinpath(data_loc, "tmaze_er_rnn_rmsprop/data"))
	diff_dict_6 = diff(ic_6)
	max_idx_6 = length(diff_dict_6["truncation"])
	diff_dict_6
end

# ╔═╡ b9495d3d-0b34-4794-b211-ea4de939a061
let
	sub_ic = new_search(ic_6) do item
		item.parsed_args["cell"] ∈ ["MARNN", "AARNN"]
	end
	diff(sub_ic)
end

# ╔═╡ b134b36e-2eac-4b3d-b0d3-19268ef1d5d7
begin
	line_data_6 = get_line_data_for(
		ic_6, 
		["cell", "hs_learnable", "truncation", "numhidden", "update_wait"], 
		["eta", "rho"];
		comp=findmax,
		get_comp_data=get_AUC_successes,
		get_data=(x)->get_rolling_mean_line(x, :successes, 100))
end

# ╔═╡ 27cd36c3-d5c5-4ae9-9975-745e4f95593c
md"""
Truncation: $(@bind τ_6_idx Slider(1:length(diff_dict_6["truncation"])))
Learn Hidden State: $(@bind hs_learn_6 CheckBox())

Hidden state size: $(@bind numhidden_6 Select(string.(diff_dict_6["numhidden"])))
Update Frequency: $(@bind update_freq_6 Select(string.(diff_dict_6["update_wait"])))

Cells: $(@bind cells_6 MultiSelect(diff_dict_6["cell"]))
"""

# ╔═╡ 6ea328d9-a64c-4200-bcfe-a88f81c01abb
let
	trunc = diff_dict_6["truncation"][τ_6_idx]
	numhidden = parse(Int, numhidden_6) 
	update_freq = parse(Int, update_freq_6)
	plt = plot_line_from_data_with_params!(line_data_6, 
		(cells_6[1], hs_learn_6, trunc, numhidden, update_freq); 
		title="τ: $(trunc), hs_learn: $(hs_learn_6)", 
		label="$(cells_6[1])", palette=color_scheme, legendtitle="Cell, (eta, rho)",
		ylims=(0.2,1.1), legend=:bottomright)
	for cell in cells_6[2:end]
		plot_line_from_data_with_params!(plt, line_data_6, 
			(cell, hs_learn_6, trunc, numhidden, update_freq); 
			label="$(cell)", palette=color_scheme)
	end
    plt
end

# ╔═╡ d8e0b040-7395-458d-860c-1f524c7bbbab
md"""
# TMAZE size=10
"""

# ╔═╡ af15725f-64ba-4a34-9104-9825c4587e54
begin
	ic_10 = ItemCollection(joinpath(data_loc, "tmaze_er_rnn_rmsprop_10/data"))
	diff_dict_10 = diff(ic_10)
	max_idx = length(diff_dict_10["truncation"])
	diff_dict_10
end

# ╔═╡ 2c57fe03-2e9b-40fa-9cc0-b982accabe35
begin
	line_data_10 = get_line_data_for(
		ic_10, 
		["cell", "hs_learnable", "truncation", "numhidden", "update_wait"], 
		["eta"];
		comp=findmax,
		get_comp_data=get_AUC_successes,
		get_data=(x)->get_rolling_mean_line(x, :successes, 100))
	
	nothing
end

# ╔═╡ 28026f3b-1a53-4030-a596-3231d5f3cac9
md"""
Truncation: $(@bind τ_10_idx Slider(1:length(diff_dict_10["truncation"])))
Learn Hidden State: $(@bind hs_learn_10 CheckBox())

Hidden state size: $(@bind numhidden_10 Select(string.(diff_dict_10["numhidden"])))
Update Frequency: $(@bind update_freq_10 Select(string.(diff_dict_10["update_wait"])))

Cells: $(@bind cells_10 MultiSelect(diff_dict_10["cell"]))
"""

# ╔═╡ 460fe8ed-691f-4a69-8210-a543ab843112
let
	trunc = diff_dict_10["truncation"][τ_10_idx]
	numhidden = parse(Int, numhidden_10) 
	update_freq = parse(Int, update_freq_10)
	plt = plot_line_from_data_with_params!(line_data_10, 
		(cells_10[1], hs_learn_10, trunc, numhidden, update_freq); 
		title="τ: $(trunc), hs_learn: $(hs_learn_10)", label="$(cells_10[1])", palette=color_scheme, legend=:bottomright)
	for cell in cells_10[2:end]
		plot_line_from_data_with_params!(plt, line_data_10, 
			(cell, hs_learn_10, trunc, numhidden, update_freq); 
			label="$(cell)", palette=color_scheme)
	end
    plt
end

# ╔═╡ 09e46aca-5390-4f81-b07a-eeba06b20859
md"""
# TMAZE Size=20
"""

# ╔═╡ f96803cc-071c-4a40-bedc-9296da0fdd3d
begin
	ic_20 = ItemCollection(joinpath(data_loc, "tmaze_er_rnn_rmsprop_20/data"))
	diff_dict_20 = diff(ic_20)
	max_idx_20 = length(diff_dict_20["truncation"])
	diff_dict_20
end

# ╔═╡ 1e02a766-04f7-4c92-b656-a2dbe89adb0d
begin
	line_data_20 = get_line_data_for(
		ic_20, 
		["cell", "hs_learnable", "truncation", "numhidden", "update_wait"], 
		"eta";
		comp=findmax,
		get_comp_data=get_AUC_successes,
		get_data=(x)->get_rolling_mean_line(x, :successes, 100))
	nothing
end

# ╔═╡ a6bfde16-4099-4bd3-bd87-5ac2145953e7
md"""
Truncation: $(@bind τ_20_idx Slider(1:length(diff_dict_20["truncation"])))
Learn Hidden State: $(@bind hs_learn_20 CheckBox())

Hidden state size: $(@bind numhidden_20 Select(string.(diff_dict_20["numhidden"])))
Update Frequency: $(@bind update_freq_20 Select(string.(diff_dict_20["update_wait"])))

Cells: $(@bind cells_20 MultiSelect(diff_dict_20["cell"]))
"""

# ╔═╡ d9ed253f-e644-4ce7-9535-f2f36c941c8e
let
	trunc = diff_dict_20["truncation"][τ_20_idx]
	numhidden = parse(Int, numhidden_20) 
	update_freq = parse(Int, update_freq_20)
	plt = plot_line_from_data_with_params!(line_data_20, 
		(cells_20[1], hs_learn_20, trunc, numhidden, update_freq); 
		title="τ: $(trunc), hs_learn: $(hs_learn_20)", label="$(cells_20[1])", palette=color_scheme, legend=:bottomright)
	for cell in cells_20[2:end]
		plot_line_from_data_with_params!(plt, line_data_20, 
			(cell, hs_learn_20, trunc, numhidden, update_freq); 
			label="$(cell)", palette=color_scheme)
	end
    plt
end

# ╔═╡ 3012a3c2-615d-4d81-ba7c-c9678b2ee499
md"""
# Ring World size = 10
"""

# ╔═╡ d4789059-91c7-411b-8326-22c0686b540a
begin
	ic_rw_10 = ItemCollection(joinpath(data_loc, "ringworld_er_rnn_rmsprop_10/data"))
	diff_dict_rw_10 = diff(ic_rw_10)
end

# ╔═╡ 22b5e24b-72c9-4d7f-bfc8-45f9d3c4cf2d
begin
	line_data_rw_10 = get_line_data_for(
		ic_rw_10, 
		["cell", "hs_learnable", "truncation", "numhidden", "update_freq", "outhorde"], 
		"eta";
		comp=findmin,
		get_comp_data=(x)->x["results"]["all"],
		get_data=(x)->x["results"]["lc"])
	nothing
end

# ╔═╡ e1f994e7-cb44-416a-9877-1840aa088f85
md"""
Out Horde: $(@bind outhorde_rw_10 Select(diff_dict_rw_10["outhorde"]))

Truncation: $(@bind τ_rw_10_idx Slider(1:length(diff_dict_rw_10["truncation"])))
Learn Hidden State: $(@bind hs_learn_rw_10 CheckBox())

Hidden state size: $(@bind numhidden_rw_10 Select(string.(diff_dict_rw_10["numhidden"])))
Update Frequency: $(@bind update_freq_rw_10 Select(string.(diff_dict_rw_10["update_freq"])))

Cells: $(@bind cells_rw_10 MultiSelect(diff_dict_rw_10["cell"]))
"""

# ╔═╡ 9e916ee4-5836-483e-8e07-5d52db378d9f
let
	trunc = diff_dict_rw_10["truncation"][τ_rw_10_idx]
	numhidden = parse(Int, numhidden_rw_10) 
	update_freq = parse(Int, update_freq_rw_10)
	plt = plot_line_from_data_with_params!(line_data_rw_10, 
		(cells_rw_10[1], hs_learn_rw_10, trunc, numhidden, update_freq, outhorde_rw_10); 
		title="$(outhorde_rw_10) τ: $(trunc), hs_learn: $(hs_learn_rw_10)", label="$(cells_rw_10[1])", palette=color_scheme, legend=:topright)
	for cell in cells_rw_10[2:end]
		plot_line_from_data_with_params!(plt, line_data_rw_10, 
			(cell, hs_learn_rw_10, trunc, numhidden, update_freq, outhorde_rw_10); 
			label="$(cell)", palette=color_scheme)
	end
    plt
end

# ╔═╡ Cell order:
# ╠═422ec05e-9afc-11eb-08e1-3363ec8d184f
# ╠═43a96d19-04e6-41d0-9b09-f055f91ce8d0
# ╠═4632fbfb-76b5-4d7e-b31f-eec453e768a2
# ╟─6b12bc96-ae5b-412b-bb66-ed053bc99b05
# ╠═5612fa62-7927-4e79-9f7e-985d037acd7c
# ╠═d4e4f2b3-4df5-4195-9595-3ca1a0da74c5
# ╠═10334c8c-e3ce-4c8c-8f6d-1fc966d0274c
# ╠═e615a5ff-7ede-4d83-b55a-a44cc5368d66
# ╠═603037f6-c95b-44c1-889d-bdf9f18becc5
# ╠═8ffcfda9-1dc2-4769-a880-26c447a4afbf
# ╠═0a58bee2-b126-41e1-a0a1-53021a8a0134
# ╠═2be7162b-ffe6-4c66-aea7-53877e7c0356
# ╠═63736a0f-8d18-4689-8a1d-0ca73f380ead
# ╠═d1683162-6958-4736-a56e-53eb02235724
# ╠═b3d6cdfb-39d7-4b02-8150-4d6a6fe5ec7a
# ╠═0deda36e-b4cd-4c7b-8548-f4cdcb5d1611
# ╠═325835b8-4f99-408d-a9d0-8efedb98d669
# ╟─011f9607-26c5-4839-b311-df41c581e480
# ╠═210cf7b1-5450-42cd-8f9b-1ab580dd16ba
# ╠═b9495d3d-0b34-4794-b211-ea4de939a061
# ╠═b134b36e-2eac-4b3d-b0d3-19268ef1d5d7
# ╟─27cd36c3-d5c5-4ae9-9975-745e4f95593c
# ╟─6ea328d9-a64c-4200-bcfe-a88f81c01abb
# ╟─d8e0b040-7395-458d-860c-1f524c7bbbab
# ╟─af15725f-64ba-4a34-9104-9825c4587e54
# ╠═2c57fe03-2e9b-40fa-9cc0-b982accabe35
# ╟─28026f3b-1a53-4030-a596-3231d5f3cac9
# ╟─460fe8ed-691f-4a69-8210-a543ab843112
# ╟─09e46aca-5390-4f81-b07a-eeba06b20859
# ╠═f96803cc-071c-4a40-bedc-9296da0fdd3d
# ╠═1e02a766-04f7-4c92-b656-a2dbe89adb0d
# ╟─a6bfde16-4099-4bd3-bd87-5ac2145953e7
# ╠═d9ed253f-e644-4ce7-9535-f2f36c941c8e
# ╟─3012a3c2-615d-4d81-ba7c-c9678b2ee499
# ╠═d4789059-91c7-411b-8326-22c0686b540a
# ╠═22b5e24b-72c9-4d7f-bfc8-45f9d3c4cf2d
# ╟─e1f994e7-cb44-416a-9877-1840aa088f85
# ╠═9e916ee4-5836-483e-8e07-5d52db378d9f
