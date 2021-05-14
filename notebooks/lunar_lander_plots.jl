### A Pluto.jl notebook ###
# v0.14.5

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

# ╔═╡ f7f500a8-a1e9-11eb-009b-d7afdcade891
using Revise

# ╔═╡ e0d51e67-63dc-45ea-9092-9965f97660b3
using Reproduce, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ aa0be590-53a5-4a1b-be42-e5d61bd0231a
using ReproducePlotUtils

# ╔═╡ 0c746c1e-ea39-4415-a1b1-d7124b886f98
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

# ╔═╡ fe50ffef-b691-47b5-acf8-8378fbf860a1
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[1],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[end-1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[end],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ 834b1cf3-5b22-4e0a-abe9-61e427e6cfda
# function plot_line_from_data_with_params!(
# 		plt, data_col::Vector{PU.LineData}, params; pkwargs...)
#     idx = findfirst(data_col) do (ld)
#         line_params = ld.line_params
# 	all([line_params[i] == params[i] for i ∈ 1:length(line_params)])
#     end
#     d = data_col[idx]
#     if plt isa Nothing
# 		plt = plot(d; pkwargs...)
#     else
# 		plot!(plt, d; pkwargs...)
#     end
#     plt
# end

# ╔═╡ f5a1e45c-398a-40e3-8c5f-b1d72ead0a89
params = Dict("numhidden" => 20, "truncation" => 20, "cell" => "MARNN")

# ╔═╡ a4763342-d9bf-4d5c-81b9-a735ae3729e3
keys(params)

# ╔═╡ 6b253f85-9005-41b6-8544-e381f46285d8
params_keys = ("numhidden", "truncation", "cell")

# ╔═╡ d50978f3-f38f-42af-a7d1-1e3b59e68175
md"""
Eta: $(@bind eta_dir_sens Select(string.(dd_dir_10["eta"])))
NumHidden: $(@bind nh_dir_sens Select(string.(dd_dir_10["numhidden"])))
Cell: $(@bind cells_dir_sens MultiSelect(dd_dir_10["cell"]))
"""

# ╔═╡ c05c7ffa-53cd-46bf-a661-886784eecc05
plt_args_list = let
	args_list = [
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"MAGRU", "eta"=>0.0003125), #num_params = 1303
#		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"MAGRU", "eta"=>0.0003125), #num_params = 3499
#		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"AAGRU", "eta"=>0.00125), #num_params = 1053
#		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"AAGRU", "eta"=>0.0003125),
#		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AAGRU", "eta"=>0.00125), #num_params = 1703
#		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"GRU", "eta"=>0.00125),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"MARNN", "eta"=>0.0003125), #num_params = 463
#		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AARNN", "eta"=>0.0003125),
#		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"RNN", "eta"=>1.953125e-5)
	]

	FileIO.save("../final_runs/dir_tmaze_10.jld2", "args", args_list)

	plt_keys = ["numhidden", "truncation", "cell"]
	[Dict(k=>args_list[i][k] for k ∈ plt_keys) for i in 1:length(args_list)]
end

# ╔═╡ cc730813-539b-418d-8da6-67f83bd05d70
plot(data_10,
	 plt_args_list,
	 palette=color_scheme, label_idx="cell", legend=:bottomright)

# ╔═╡ 87b22774-0e25-4202-9819-3a9a15cbcacb
const RPU = ReproducePlotUtils

# ╔═╡ 1886bf05-f4be-4160-b61c-edf186a7f3cb
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ eefd2ccc-ce9b-4cf8-991f-a58f2f932e99
ic_dir, dd_dir = RPU.load_data("../local_data/lunar_lander_er_rmsprop/")

# ╔═╡ eab5c7a0-8052-432d-977e-68b967baf5ca
ic_dir[1].parsed_args["steps"]

# ╔═╡ e3ebad86-41aa-452c-a6b3-48c3b497c27d
md"""
NumHidden: $(@bind nh_dir_opt Select(string.(dd_dir["numhidden"])))
Truncation: $(@bind τ_dir_opt Select(string.(dd_dir["truncation"])))
Cell: $(@bind cells_dir_opt MultiSelect(dd_dir["cell"]))
"""

# ╔═╡ 467ed417-cf69-493d-b21c-3fc4d1fb9907
md"""
Target Update Wait: $(@bind tuw_dir Select(string.(dd_dir["target_update_wait"])))
NumHidden: $(@bind nh_dir Select(string.(dd_dir["numhidden"])))
Rho: $(@bind rho_dir Select(string.(dd_dir["rho"])))
Truncation: $(@bind τ_dir Select(string.(dd_dir["truncation"])))
Replay Size: $(@bind rs_dir Select(string.(dd_dir["replay_size"])))
Eta: $(@bind eta_dir Select(string.(dd_dir["eta"])))
Cell: $(@bind cells_dir MultiSelect(dd_dir["cell"]))
Update Wait: $(@bind uw_dir Select(string.(dd_dir["update_wait"])))
"""

# ╔═╡ 40cc0919-c56b-401b-83fa-0feb263c44b0
md"""
NumHidden: $(@bind nh_dir_sens_ Select(string.(dd_dir["numhidden"])))
Truncation: $(@bind τ_dir_sens_ Select(string.(dd_dir["truncation"])))
Cell: $(@bind cells_dir_sens_ MultiSelect(dd_dir["cell"]))
"""

# ╔═╡ 032c75e5-ddd1-4b16-b524-af2d8f99d41e
data_opt = RPU.get_line_data_for(
	ic_dir,
	["numhidden", "truncation", "cell"],
	["target_update_wait", "rho", "replay_size", "eta", "update_wait"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 1000))

# ╔═╡ 3d8b0b2c-005e-4294-8817-e6bd075a0fb8
let
	# plt = nothing
	nh = parse(Int, nh_dir_opt)
	τ = parse(Int, τ_dir_opt)
	plt = plot()
	for cell ∈ cells_dir_opt
		plt = plot!(
			  data_opt,
			  Dict("numhidden"=>nh, "truncation"=>τ, "cell"=>cell),
			  palette=RPU.custom_colorant, legend=:topleft, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", title="Lunar Lander, $(ic_dir[1].parsed_args["steps"]) steps")
	end
	plt
end

# ╔═╡ 55d1416b-d580-4112-827a-30504c21f397
data_all = RPU.get_line_data_for(
	ic_dir,
	["target_update_wait", "numhidden", "rho", "truncation", "replay_size", "eta", "cell", "update_wait"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 1000))

# ╔═╡ 2040d613-0ea8-48ce-937c-9180073812ea
let
	# plt = nothing
	tuw = parse(Int, tuw_dir)
	nh = parse(Int, nh_dir)
	rho = parse(Float64, rho_dir)
	τ = parse(Int, τ_dir)
	rs = parse(Int, rs_dir)
	eta = parse(Float64, eta_dir)
	uw = parse(Int, uw_dir)
	plt = plot()
	for cell ∈ cells_dir
		plt = plot!(
			  data_all,
			  Dict("target_update_wait"=>tuw, "numhidden"=>nh, "rho"=>rho, "truncation"=>τ, "replay_size"=>rs, "eta"=>eta, "cell"=>cell, "update_wait"=>uw),
			  palette=RPU.custom_colorant, legend=:topleft, ylim=(-500, 300))
	end
	plt
end

# ╔═╡ 7882977c-4802-4372-ab5b-1e2e45130fed
data_sens = RPU.get_line_data_for(
	ic_dir,
	["numhidden", "truncation", "cell", "eta"],
	["target_update_wait", "rho", "replay_size", "update_wait"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))

# ╔═╡ ec296879-424d-4de7-8e41-93c4d113f4ff
let
	# plt = nothing
	nh = parse(Int, nh_dir_sens_)
	τ = parse(Int, τ_dir_sens_)
	plt = plot()
	for cell ∈ cells_dir_sens_
		plt = plot!(data_sens,
	 	  Dict("numhidden"=>nh, "cell"=>cell, "truncation"=>τ);
	 	  sort_idx="eta",
		  z=1.97, lw=2, xaxis=:log,
		  palette=RPU.custom_colorant, label="cell: $(cell)",
		  color=cell_colors[cell], title="numhidden: $(nh)")
	end
	plt
end

# ╔═╡ 7c1f8e58-4bfa-4da2-833d-0cc2a4ec74a4
data_10_sens = RPU.get_line_data_for(
	ic_dir_10,
	["numhidden", "cell", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 859b98dd-1024-4fec-8741-40edf72d0287
data_10_sens[1]

# ╔═╡ 4c8483f5-3a36-43a1-b9fc-2e757b3624ab
idx = findall(data_10_sens.data) do (ld)
        line_params = ld.line_params
	all([line_params[i] == params[i] for i ∈ keys(params)])
    end

# ╔═╡ 6ed62324-1866-452e-9a59-ba968425b7c6
data_10_sens[idx[2]].line_params

# ╔═╡ 094b456c-a7ad-4f93-89a0-c403c73bdc55
data_10_sens[idx[2]].data

# ╔═╡ e2f179e9-5e75-4e6c-917b-07aef8b6f8b7
mean(data_10_sens[idx[2]].data)

# ╔═╡ 4a64f3a7-3879-407a-9654-aacc1861fc42
data_10_sens_ = RPU.get_line_data_for(
	ic_dir_10,
	["numhidden", "truncation", "cell", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 300))

# ╔═╡ 7425a013-fdbf-4d50-921d-64306d4c2741
let
	# plt = nothing
	τ = parse(Int, τ_dir_sens)
	nh = parse(Int, nh_dir_sens)
	eta = parse(Float64, eta_dir_sens)
	plt = plot()
	for cell ∈ cells_dir_sens
		plt = plot!(
			  data_10_sens_,
			  Dict("numhidden"=>nh, "truncation"=>τ, "cell"=>cell, "eta"=>eta),
			  palette=RPU.custom_colorant, legend=:topleft)
	end
	plt
end

# ╔═╡ f5c3a38a-cf78-4b90-b371-506cc2997f92
let
	# plt = nothing
	# τ = parse(Int, τ_dir_sens)
	nh = parse(Int, nh_dir_sens)
	plt = plot()
	for cell ∈ cells_dir_sens
		plt = plot!(data_10_sens,
	 	  Dict("numhidden"=>nh, "cell"=>cell);
	 	  sort_idx="eta",
		  z=1.97, lw=2, xaxis=:log,
		  palette=RPU.custom_colorant, label="cell: $(cell)",
		  color=cell_colors[cell], title="numhidden: $(nh)")
	end
	plt
end

# ╔═╡ cc33f308-56d6-496f-93c0-55d64c7b6280
function get_agg(agg, ddict, key)
    agg(ddict["results"][key])
end

# ╔═╡ f33917b7-02cc-4a2c-8fcb-94f2eaab0093
get_MUHE(ddict, key, perc=0.1) = get_agg(ddict, key) do x
    mean(x[Int(floor(length(x)/2)) - max(1, Int(floor((length(x)/2)*perc))): Int(floor(length(x)/2))])
end

# ╔═╡ fc961ab4-c7c6-4e7b-91be-3e5fbb18d667
data_10_dist = RPU.get_line_data_for(
	ic_dir_10,
	["numhidden", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->get_MUHE(x, :successes),
	get_data=(x)->get_MUHE(x, :successes))

# ╔═╡ db83dce1-29d5-42f8-b226-4412ce63c8f1
let
	args_list_l = [
		Dict("numhidden"=>26, "cell"=>"GRU"),
		Dict("numhidden"=>26, "cell"=>"AAGRU"),
		Dict("numhidden"=>15, "cell"=>"MAGRU"),
		Dict("numhidden"=>46, "cell"=>"RNN"),
		Dict("numhidden"=>46, "cell"=>"AARNN"),
		Dict("numhidden"=>27, "cell"=>"MARNN")]
	
	#args_list_l = [
	#	Dict("numhidden"=>20, "truncation"=>12, "cell"=>"GRU"),
	#	Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AAGRU"),
	#	Dict("numhidden"=>10, "truncation"=>20, "cell"=>"MAGRU"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"RNN"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"AARNN"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"MARNN")]
	boxplot(data_10_dist, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="MUE for 500k Steps")
	dotplot!(data_10_dist, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ c062a162-afba-42f2-a7cf-e9732ad0df33
get_MEANH(ddict, key) = get_agg(ddict, key) do x
    mean(x[1: Int(floor(length(x)/2))])
end

# ╔═╡ c8e944f2-ef02-4b81-8b6d-3527609f0822
data_10_dist_mean = RPU.get_line_data_for(
	ic_dir_10,
	["numhidden", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->get_MEANH(x, :successes),
	get_data=(x)->get_MEANH(x, :successes))

# ╔═╡ e0a3851d-3db5-440e-b67e-75ca7d9517e1
let
	args_list_l = [
		Dict("numhidden"=>26, "cell"=>"GRU"),
		Dict("numhidden"=>26, "cell"=>"AAGRU"),
		Dict("numhidden"=>15, "cell"=>"MAGRU"),
		Dict("numhidden"=>46, "cell"=>"RNN"),
		Dict("numhidden"=>46, "cell"=>"AARNN"),
		Dict("numhidden"=>27, "cell"=>"MARNN")]
	
	#args_list_l = [
	#	Dict("numhidden"=>20, "truncation"=>12, "cell"=>"GRU"),
	#	Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AAGRU"),
	#	Dict("numhidden"=>10, "truncation"=>20, "cell"=>"MAGRU"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"RNN"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"AARNN"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"MARNN")]
	boxplot(data_10_dist_mean, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Mean for 500k Steps")
	dotplot!(data_10_dist_mean, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ 6ef161be-ec61-461d-9c38-60510c08328b
ic_dir_10_s, dd_dir_10_s = RPU.load_data("../local_data/dir_tmaze_online_rmsprop_size10/")

# ╔═╡ 4f2e1222-9daa-439c-9d57-957f23e44657
data_10_dist_s = RPU.get_line_data_for(
	ic_dir_10_s,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :successes),
	get_data=(x)->RPU.get_MEAN(x, :successes))

# ╔═╡ 4d5dc03a-8b9b-4b5a-b002-6fc48a580b04
let
	args_list_l = [
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"GRU"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"AAGRU"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"MAGRU"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"RNN"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"AARNN"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"MARNN")]
	boxplot(data_10_dist_s, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="MEAN")
	dotplot!(data_10_dist_s, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ b8be3138-98e3-476a-adcf-564196b51ae7
data_10_dist_s_mue = RPU.get_line_data_for(
	ic_dir_10_s,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ f9e81f7b-26f3-4ce5-bca7-5054c8ddf44a
let
	args_list_l = [
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"GRU"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"AAGRU"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"MAGRU"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"RNN"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"AARNN"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"MARNN")]
	boxplot(data_10_dist_s_mue, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="MUE")
	dotplot!(data_10_dist_s_mue, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ 305f8ac8-8f5f-4ec4-84a6-867f69a8887c
ic_fac, dd_fac = RPU.load_data("../local_data/dir_tmaze_er_fac_rnn_rmsprop_10/")

# ╔═╡ 533cba3d-7fc5-4d66-b545-b15ffc8ab6d8
data_fac_sens_eta = RPU.get_line_data_for(
	ic_fac,
	["numhidden", "cell", "replay_size", "factors", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 39752286-a6db-439d-aca0-1be4821bfc2b
let
	args_list = [
		Dict("numhidden"=>15, "replay_size"=>20000, "cell"=>"FacMARNN", "factors"=>10),
		Dict("numhidden"=>15, "replay_size"=>20000, "cell"=>"FacMAGRU", "factors"=>10)]
	plot(data_fac_sens_eta, args_list; sort_idx="eta", labels=["FacMARNN" "FacMAGRU"])
end

# ╔═╡ 7d611b39-f9a8-43e4-951e-9d812cbd4384
data_fac_sens = RPU.get_line_data_for(
	ic_fac,
	["numhidden", "cell", "replay_size", "factors"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ a8949e02-61f5-456a-abee-2bad91d2df05
let
	args_list = [
		Dict("numhidden"=>15, "replay_size"=>20000, "cell"=>"FacMARNN"),
		Dict("numhidden"=>15, "replay_size"=>20000, "cell"=>"FacMAGRU")]
	plot(data_fac_sens, args_list; sort_idx="factors", labels=["FacMARNN" "FacMAGRU"])
end

# ╔═╡ 2dbcb518-2fda-44c4-bfc0-b422a8da9c35
let
	args_list = [
		Dict("numhidden"=>15, "replay_size"=>20000, "cell"=>"FacMARNN", "factors"=>10),
		Dict("numhidden"=>15, "replay_size"=>20000, "cell"=>"FacMAGRU", "factors"=>10)]
	violin(data_fac_sens, args_list; label_idx="cell", color = [cell_colors["FacMARNN"] cell_colors["FacMAGRU"]])
	dotplot!(data_fac_sens, args_list; label_idx="cell", color = [cell_colors["FacMARNN"] cell_colors["FacMAGRU"]])
	args_list_l = [
		Dict("numhidden"=>17, "truncation"=>12, "cell"=>"GRU"),
		Dict("numhidden"=>17, "truncation"=>12, "cell"=>"AAGRU"),
		Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MAGRU"),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"RNN"),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AARNN"),
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"MARNN")]
	violin!(data_10_dist, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_10_dist, args_list_l; label_idx="cell",
				color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),)
	# dotplot!(data_10_dist, args_list_l;
	# 	label_idx="cell",
	# 	color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
end

# ╔═╡ Cell order:
# ╠═f7f500a8-a1e9-11eb-009b-d7afdcade891
# ╠═e0d51e67-63dc-45ea-9092-9965f97660b3
# ╠═0c746c1e-ea39-4415-a1b1-d7124b886f98
# ╟─1886bf05-f4be-4160-b61c-edf186a7f3cb
# ╠═fe50ffef-b691-47b5-acf8-8378fbf860a1
# ╟─834b1cf3-5b22-4e0a-abe9-61e427e6cfda
# ╠═eefd2ccc-ce9b-4cf8-991f-a58f2f932e99
# ╠═eab5c7a0-8052-432d-977e-68b967baf5ca
# ╠═032c75e5-ddd1-4b16-b524-af2d8f99d41e
# ╠═e3ebad86-41aa-452c-a6b3-48c3b497c27d
# ╠═3d8b0b2c-005e-4294-8817-e6bd075a0fb8
# ╠═55d1416b-d580-4112-827a-30504c21f397
# ╠═467ed417-cf69-493d-b21c-3fc4d1fb9907
# ╠═2040d613-0ea8-48ce-937c-9180073812ea
# ╠═7882977c-4802-4372-ab5b-1e2e45130fed
# ╟─40cc0919-c56b-401b-83fa-0feb263c44b0
# ╠═ec296879-424d-4de7-8e41-93c4d113f4ff
# ╠═7c1f8e58-4bfa-4da2-833d-0cc2a4ec74a4
# ╠═859b98dd-1024-4fec-8741-40edf72d0287
# ╠═f5a1e45c-398a-40e3-8c5f-b1d72ead0a89
# ╠═a4763342-d9bf-4d5c-81b9-a735ae3729e3
# ╠═6b253f85-9005-41b6-8544-e381f46285d8
# ╠═4c8483f5-3a36-43a1-b9fc-2e757b3624ab
# ╠═6ed62324-1866-452e-9a59-ba968425b7c6
# ╠═094b456c-a7ad-4f93-89a0-c403c73bdc55
# ╠═e2f179e9-5e75-4e6c-917b-07aef8b6f8b7
# ╠═4a64f3a7-3879-407a-9654-aacc1861fc42
# ╠═d50978f3-f38f-42af-a7d1-1e3b59e68175
# ╠═7425a013-fdbf-4d50-921d-64306d4c2741
# ╠═f5c3a38a-cf78-4b90-b371-506cc2997f92
# ╠═c05c7ffa-53cd-46bf-a661-886784eecc05
# ╠═cc730813-539b-418d-8da6-67f83bd05d70
# ╠═aa0be590-53a5-4a1b-be42-e5d61bd0231a
# ╠═87b22774-0e25-4202-9819-3a9a15cbcacb
# ╠═cc33f308-56d6-496f-93c0-55d64c7b6280
# ╠═f33917b7-02cc-4a2c-8fcb-94f2eaab0093
# ╠═fc961ab4-c7c6-4e7b-91be-3e5fbb18d667
# ╠═db83dce1-29d5-42f8-b226-4412ce63c8f1
# ╠═c062a162-afba-42f2-a7cf-e9732ad0df33
# ╠═c8e944f2-ef02-4b81-8b6d-3527609f0822
# ╠═e0a3851d-3db5-440e-b67e-75ca7d9517e1
# ╠═6ef161be-ec61-461d-9c38-60510c08328b
# ╠═4f2e1222-9daa-439c-9d57-957f23e44657
# ╠═4d5dc03a-8b9b-4b5a-b002-6fc48a580b04
# ╠═b8be3138-98e3-476a-adcf-564196b51ae7
# ╠═f9e81f7b-26f3-4ce5-bca7-5054c8ddf44a
# ╠═305f8ac8-8f5f-4ec4-84a6-867f69a8887c
# ╠═533cba3d-7fc5-4d66-b545-b15ffc8ab6d8
# ╠═39752286-a6db-439d-aca0-1be4821bfc2b
# ╠═7d611b39-f9a8-43e4-951e-9d812cbd4384
# ╠═a8949e02-61f5-456a-abee-2bad91d2df05
# ╠═2dbcb518-2fda-44c4-bfc0-b422a8da9c35
