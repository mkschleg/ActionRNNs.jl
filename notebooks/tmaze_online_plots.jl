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

# ╔═╡ f7f500a8-a1e9-11eb-009b-d7afdcade891
using Revise

# ╔═╡ e0d51e67-63dc-45ea-9092-9965f97660b3
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ e53d7b29-788c-469c-9d44-573f996fa5e7
const RPU = ReproducePlotUtils

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
	colorant"#1E90FF",
]

# ╔═╡ 1886bf05-f4be-4160-b61c-edf186a7f3cb
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ fe50ffef-b691-47b5-acf8-8378fbf860a1
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
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

# ╔═╡ eefd2ccc-ce9b-4cf8-991f-a58f2f932e99
ic_dir_10, dd_dir_10 = RPU.load_data("../local_data/tmaze_online_rmsprop_size10/")

# ╔═╡ eab5c7a0-8052-432d-977e-68b967baf5ca
ic_dir_10[1].parsed_args["steps"]

# ╔═╡ 55d1416b-d580-4112-827a-30504c21f397
data_10 = RPU.get_line_data_for(
	ic_dir_10,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 500))

# ╔═╡ 467ed417-cf69-493d-b21c-3fc4d1fb9907
md"""
Truncation: $(@bind τ_dir_10 Select(string.(dd_dir_10["truncation"])))
NumHidden: $(@bind nh_dir_10 Select(string.(dd_dir_10["numhidden"])))
Cell: $(@bind cells_dir_10 MultiSelect(dd_dir_10["cell"]))
"""

# ╔═╡ 2040d613-0ea8-48ce-937c-9180073812ea
let
	# plt = nothing
	τ = parse(Int, τ_dir_10)
	nh = parse(Int, nh_dir_10)
	plt = plot()
	for cell ∈ cells_dir_10
		plt = plot!(
			  data_10,
			  Dict("numhidden"=>nh, "truncation"=>τ, "cell"=>cell),
			  palette=RPU.custom_colorant, legend=:bottomright, lw=1)
	end
	plt
end

# ╔═╡ 7c1f8e58-4bfa-4da2-833d-0cc2a4ec74a4
data_10_sens = RPU.get_line_data_for(
	ic_dir_10,
	["numhidden", "truncation", "cell", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 859b98dd-1024-4fec-8741-40edf72d0287
data_10_sens[1]

# ╔═╡ f5a1e45c-398a-40e3-8c5f-b1d72ead0a89
params = Dict("numhidden" => 20, "truncation" => 20, "cell" => "MARNN")

# ╔═╡ a4763342-d9bf-4d5c-81b9-a735ae3729e3
keys(params)

# ╔═╡ 6b253f85-9005-41b6-8544-e381f46285d8
params_keys = ("numhidden", "truncation", "cell")

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

# ╔═╡ d50978f3-f38f-42af-a7d1-1e3b59e68175
md"""
Eta: $(@bind eta_dir_sens Select(string.(dd_dir_10["eta"])))
Truncation: $(@bind τ_dir_sens Select(string.(dd_dir_10["truncation"])))
NumHidden: $(@bind nh_dir_sens Select(string.(dd_dir_10["numhidden"])))
Cell: $(@bind cells_dir_sens MultiSelect(dd_dir_10["cell"]))
"""

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
	τ = parse(Int, τ_dir_sens)
	nh = parse(Int, nh_dir_sens)
	plt = plot()
	for cell ∈ cells_dir_sens
		plt = plot!(data_10_sens,
	 	  Dict("numhidden"=>nh, "truncation"=>τ, "cell"=>cell);
	 	  sort_idx="eta",
		  z=1.97, lw=2, xaxis=:log,
		  palette=RPU.custom_colorant, label="cell: $(cell)",
		  color=cell_colors[cell])
	end
	plt
end

# ╔═╡ c6c301bd-d3c4-4d63-9b7b-46ea9ac7bc87
let

	plot(data_10_sens,
	 	  Dict("numhidden"=>15, "truncation"=>20, "cell"=>"GRU");
	 	  sort_idx="eta",
		  z=1.97, lw=2,
		  palette=RPU.custom_colorant, label="GRU",
		  color=cell_colors["GRU"])
    plot!(data_10_sens,
	 	  Dict("numhidden"=>15, "truncation"=>12, "cell"=>"AAGRU");
	 	  sort_idx="eta", z=1.97, lw=2, palette=RPU.custom_colorant,
		  label="AAGRU", color=cell_colors["AAGRU"])
	plot!(data_10_sens,
	 	 Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MAGRU");
		 sort_idx="eta",
		 z=1.97,
		 lw=2,
		 palette=RPU.custom_colorant,
		 xaxis=:log, label="MAGRU", color=cell_colors["MAGRU"])
end

# ╔═╡ b3916560-ccd5-408b-9d8a-9ecbdbe4b18d
let

	plot(data_10_sens,
	 	  Dict("numhidden"=>17, "truncation"=>20, "cell"=>"RNN");
	 	  sort_idx="eta",
		  z=1.97, lw=2,
		  palette=RPU.custom_colorant, label="RNN",
		  color=cell_colors["RNN"])
    plot!(data_10_sens,
	 	  Dict("numhidden"=>17, "truncation"=>12, "cell"=>"AARNN");
	 	  sort_idx="eta", z=1.97, lw=2, palette=RPU.custom_colorant, label="AARNN",
		  color=cell_colors["AARNN"])
	plot!(data_10_sens,
	 	 Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MARNN");
		 sort_idx="eta",
		 z=1.97,
		 lw=2,
		 palette=RPU.custom_colorant,
		 xaxis=:log, label="MARNN",
		 color=cell_colors["MARNN"])
end

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

# ╔═╡ fc961ab4-c7c6-4e7b-91be-3e5fbb18d667
data_10_dist = RPU.get_line_data_for(
	ic_dir_10,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 58fa23c9-ef42-44c2-856d-e189cad6aa9b
data_10_dist[1]

# ╔═╡ db83dce1-29d5-42f8-b226-4412ce63c8f1
let
	args_list_l = [
		Dict("numhidden"=>10, "truncation"=>12, "cell"=>"GRU"),
		Dict("numhidden"=>10, "truncation"=>12, "cell"=>"AAGRU"),
		Dict("numhidden"=>6, "truncation"=>12, "cell"=>"MAGRU"),
		Dict("numhidden"=>17, "truncation"=>12, "cell"=>"RNN"),
		Dict("numhidden"=>17, "truncation"=>12, "cell"=>"AARNN"),
		Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MARNN")]
	
	violin(data_10_dist, args_list_l, label_idx="cell", legend=false, color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :), lw=2, linecolor=:black)
	boxplot!(data_10_dist, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, ylims=(0.3, 1.0), fillalpha=0.75, outliers=false, lw=2, linecolor=:black, tickfontsize=11, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="TMaze Online")
	dotplot!(data_10_dist, args_list_l, label_idx="cell", color=:black, tickdir=:out, grid=false)
savefig("../data/paper_plots/tmaze_online_violin_and_box_plots_300k_steps_MUE_tau_12.pdf")
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
# ╠═e53d7b29-788c-469c-9d44-573f996fa5e7
# ╠═0c746c1e-ea39-4415-a1b1-d7124b886f98
# ╟─1886bf05-f4be-4160-b61c-edf186a7f3cb
# ╠═fe50ffef-b691-47b5-acf8-8378fbf860a1
# ╟─834b1cf3-5b22-4e0a-abe9-61e427e6cfda
# ╠═eefd2ccc-ce9b-4cf8-991f-a58f2f932e99
# ╠═eab5c7a0-8052-432d-977e-68b967baf5ca
# ╠═55d1416b-d580-4112-827a-30504c21f397
# ╟─467ed417-cf69-493d-b21c-3fc4d1fb9907
# ╠═2040d613-0ea8-48ce-937c-9180073812ea
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
# ╠═c6c301bd-d3c4-4d63-9b7b-46ea9ac7bc87
# ╠═b3916560-ccd5-408b-9d8a-9ecbdbe4b18d
# ╠═c05c7ffa-53cd-46bf-a661-886784eecc05
# ╠═cc730813-539b-418d-8da6-67f83bd05d70
# ╠═fc961ab4-c7c6-4e7b-91be-3e5fbb18d667
# ╠═58fa23c9-ef42-44c2-856d-e189cad6aa9b
# ╠═db83dce1-29d5-42f8-b226-4412ce63c8f1
# ╠═305f8ac8-8f5f-4ec4-84a6-867f69a8887c
# ╠═533cba3d-7fc5-4d66-b545-b15ffc8ab6d8
# ╠═39752286-a6db-439d-aca0-1be4821bfc2b
# ╠═7d611b39-f9a8-43e4-951e-9d812cbd4384
# ╠═a8949e02-61f5-456a-abee-2bad91d2df05
# ╠═2dbcb518-2fda-44c4-bfc0-b422a8da9c35
