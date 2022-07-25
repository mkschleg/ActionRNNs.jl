### A Pluto.jl notebook ###
# v0.19.9

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

# ╔═╡ ac8f3d1b-82af-4917-bf1c-d7afc16fc43a
let
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ f7f500a8-a1e9-11eb-009b-d7afdcade891
using Revise

# ╔═╡ e0d51e67-63dc-45ea-9092-9965f97660b3
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI

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

# ╔═╡ 842b3fbc-34aa-452d-81fb-2ade57dedecb
at(dir) = joinpath("../../local_data/dir_tmaze_er/", dir)

# ╔═╡ cc4a219d-9118-4c34-93ce-317afc837f6c
function get_final_argument_list(
		base_params, 
		data_col, 
		diff_dict, 
		extra_params, 
		sweep_params)
	
	args = Dict{String, Any}[]
	for pms ∈ base_params
		for eps ∈ Iterators.product([diff_dict[ep] for ep in extra_params]...)

			pms_copy = copy(pms)
			for (ep, epv) ∈ zip(extra_params, eps)
				pms_copy[ep] = epv
			end
			idx = findfirst(data_col.data) do ld
				all([pms_copy[k] == ld.line_params[k] for k in keys(pms_copy)])
			end
			ld = data_col[idx]
			for (sp, spv) ∈ zip(sweep_params, ld.swept_params)
				pms_copy[sp] = spv
			end
			push!(args, pms_copy)
		end
	end
	return args
end

# ╔═╡ 1756d1cc-1a88-4442-b55a-fdbb44f56313
md"""
# Directional TMaze size 6

"""

# ╔═╡ 0fc6fd35-5a24-4aaf-9ebb-6c4af1ba259b
ic_dir_6, dd_dir_6 = RPU.load_data(at("dir_tmaze_er_rnn_rmsprop/"))

# ╔═╡ 6211a38a-7b53-4054-970e-c29ad17de646
ic_dir_6[1].parsed_args["steps"]

# ╔═╡ e822182e-b485-4a95-a08c-efe1540ff6ad
data_6 = RPU.get_line_data_for(
	ic_dir_6,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 300))

# ╔═╡ c1b80cfd-dbb8-41c5-a778-66b112e1c091
md"""
Truncation: $(@bind τ_dir_6 Select(string.(dd_dir_6["truncation"])))
NumHidden: $(@bind nh_dir_6 Select(string.(dd_dir_6["numhidden"])))
Cell: $(@bind cells_dir_6 MultiSelect(dd_dir_6["cell"]))
"""

# ╔═╡ e4cc9109-d8b1-4bff-a176-3627e24ab757
let 
	plt = nothing
	τ = parse(Int, τ_dir_6)
	nh = parse(Int, nh_dir_6)
	plt = plot()
	for cell ∈ cells_dir_6
		plot!(plt, 
			data_6, 
			Dict("numhidden"=>nh, "truncation"=>τ, "cell"=>cell), 
			palette=RPU.custom_colorant)
	end
	plt
end

# ╔═╡ 0eb4c818-b533-4695-a0c7-53e72023281f
let 
	args_list = [
		Dict("numhidden"=>6, "truncation"=>8, "cell"=>"MAGRU"),
		Dict("numhidden"=>10, "truncation"=>8, "cell"=>"AAGRU"),
		Dict("numhidden"=>10, "truncation"=>8, "cell"=>"GRU"),
		Dict("numhidden"=>6, "truncation"=>8, "cell"=>"MARNN"),
		Dict("numhidden"=>10, "truncation"=>8, "cell"=>"AARNN"),
		Dict("numhidden"=>10, "truncation"=>8, "cell"=>"RNN")
	]
	
	plot(data_6, args_list, palette=RPU.custom_colorant)
end

# ╔═╡ 4d31c08d-d532-447d-931e-6eec9c7882f5
let
	args_list = [
		Dict("numhidden"=>6, "truncation"=>8, "cell"=>"MAGRU", "eta"=>0.00125),
		Dict("numhidden"=>10, "truncation"=>8, "cell"=>"AAGRU", "eta"=>0.00125),
		Dict("numhidden"=>10, "truncation"=>8, "cell"=>"GRU", "eta"=>1.953125e-5),
		Dict("numhidden"=>6, "truncation"=>8, "cell"=>"MARNN", "eta"=>0.0003125),
		Dict("numhidden"=>10, "truncation"=>8, "cell"=>"AARNN", "eta"=>0.0003125),
		Dict("numhidden"=>10, "truncation"=>8, "cell"=>"RNN", "eta"=>7.8125e-5)
	]
end

# ╔═╡ 5c3ddaed-47b6-465b-b4c5-bc0fb30c8601
data_6_sens = RPU.get_line_data_for(
	ic_dir_6,
	["cell", "numhidden", "truncation", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 37af922e-ed3c-4aec-a4cf-c403c49a9ba9
let
	plot(data_6_sens, 
	 	 Dict("numhidden"=>6, "truncation"=>8, "cell"=>"MAGRU"); 
		 sort_idx="eta", z=1.97, lw=2, palette=ReproducePlotUtils.custom_colorant)
	plot!(data_6_sens, 
	 	  Dict("numhidden"=>10, "truncation"=>8, "cell"=>"GRU"); 
	 	  sort_idx="eta", z=1.97, lw=2, palette=ReproducePlotUtils.custom_colorant)
    plot!(data_6_sens, 
	 	  Dict("numhidden"=>10, "truncation"=>8, "cell"=>"AAGRU"); 
	 	  sort_idx="eta", z=1.97, lw=2, palette=ReproducePlotUtils.custom_colorant)
end

# ╔═╡ 6293dc31-1428-4315-bf56-d7a20ee2a17d
md"""
# Directional TMaze size 10
"""

# ╔═╡ eefd2ccc-ce9b-4cf8-991f-a58f2f932e99
ic_dir_10, dd_dir_10 = let
	ic_dir_10, dd_dir_10 = RPU.load_data(at("dir_tmaze_er_rnn_rmsprop_10/"))
	ic_dir_p2, dd_p2 = RPU.load_data(at("dir_tmaze_er_rnn_rmsprop_10_p2/"))
	ic = ItemCollection([ic_dir_10.items; ic_dir_p2.items])
	ic, diff(ic)
end

# ╔═╡ eab5c7a0-8052-432d-977e-68b967baf5ca
ic_dir_10[1].parsed_args["steps"]

# ╔═╡ 55d1416b-d580-4112-827a-30504c21f397
data_10 = RPU.get_line_data_for(
	ic_dir_10,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 300))

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
			  palette=RPU.custom_colorant)
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

# ╔═╡ c6c301bd-d3c4-4d63-9b7b-46ea9ac7bc87
let

	plot(data_10_sens, 
	 	  Dict("numhidden"=>17, "truncation"=>20, "cell"=>"GRU"); 
	 	  sort_idx="eta", 
		  z=1.97, lw=2, 
		  palette=RPU.custom_colorant, label="GRU",
		  color=cell_colors["GRU"])
    plot!(data_10_sens, 
	 	  Dict("numhidden"=>17, "truncation"=>12, "cell"=>"AAGRU"); 
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
	 	  Dict("numhidden"=>30, "truncation"=>12, "cell"=>"RNN"); 
	 	  sort_idx="eta", 
		  z=1.97, lw=2, 
		  palette=RPU.custom_colorant, label="RNN",
		  color=cell_colors["RNN"])
    plot!(data_10_sens, 
	 	  Dict("numhidden"=>30, "truncation"=>12, "cell"=>"AARNN"); 
	 	  sort_idx="eta", z=1.97, lw=2, palette=RPU.custom_colorant, label="AARNN",
		  color=cell_colors["AARNN"])
	plot!(data_10_sens, 
	 	 Dict("numhidden"=>18, "truncation"=>12, "cell"=>"MARNN"); 
		 sort_idx="eta", 
		 z=1.97, 
		 lw=2, 
		 palette=RPU.custom_colorant, 
		 xaxis=:log, label="MARNN",
		 color=cell_colors["MARNN"])
end

# ╔═╡ c05c7ffa-53cd-46bf-a661-886784eecc05
plt_args_list_1 = let
	args_list = [
		Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MAGRU", "eta"=>0.0003125), #num_params = 1303
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"MAGRU", "eta"=>0.0003125), #num_params = 3499
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"AAGRU", "eta"=>0.00125), #num_params = 1053
		Dict("numhidden"=>17, "truncation"=>12, "cell"=>"AAGRU", "eta"=>0.0003125),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AAGRU", "eta"=>0.00125), #num_params = 1703
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"GRU", "eta"=>0.00125),
		Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MARNN", "eta"=>0.0003125), #num_params = 463
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AARNN", "eta"=>0.0003125),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"RNN", "eta"=>1.953125e-5)
	]

	# FileIO.save("../final_runs/dir_tmaze_10.jld2", "args", args_list)
	
	plt_keys = ["numhidden", "truncation", "cell"]
	[Dict(k=>args_list[i][k] for k ∈ plt_keys) for i in 1:length(args_list)]
end

# ╔═╡ cc730813-539b-418d-8da6-67f83bd05d70
plot(data_10, 
	 plt_args_list_1,
	 palette=color_scheme, label_idx="cell", legend=:bottomright)

# ╔═╡ fc961ab4-c7c6-4e7b-91be-3e5fbb18d667
data_10_dist = RPU.get_line_data_for(
	ic_dir_10,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ db83dce1-29d5-42f8-b226-4412ce63c8f1
let
	args_list_l = [
		Dict("numhidden"=>17, "truncation"=>12, "cell"=>"GRU"),
		Dict("numhidden"=>17, "truncation"=>12, "cell"=>"AAGRU"),
		Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MAGRU"),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"RNN"),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AARNN"),
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"MARNN")]
	boxplot(data_10_dist, args_list_l; 
		label_idx="cell", 
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_10_dist, args_list_l; 
		label_idx="cell", 
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ 305f8ac8-8f5f-4ec4-84a6-867f69a8887c
ic_fac, dd_fac = RPU.load_data(at("dir_tmaze_er_fac_rnn_rmsprop_10/"))

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

# ╔═╡ 4b654ad2-93cf-455e-9c7b-982766560205
let
	plts = []
	for rs ∈ dd_fac["replay_size"]
		
		args_list = [
			Dict("numhidden"=>15, "replay_size"=>rs, "cell"=>"FacMARNN"),
			Dict("numhidden"=>15, "replay_size"=>rs, "cell"=>"FacMAGRU")]
		push!(plts, plot(data_fac_sens, args_list; sort_idx="factors", labels=["FacMARNN" "FacMAGRU"], title=rs, color = 	[cell_colors["FacMARNN"] cell_colors["FacMAGRU"]], legend=nothing, z=1.97, lw=2, xlabel="Factors", ylabel="Success"))
	end
	plot(plts..., size=(800, 600))
end

# ╔═╡ 2dbcb518-2fda-44c4-bfc0-b422a8da9c35
let
	args_list = [
		Dict("numhidden"=>25, "replay_size"=>10000, "cell"=>"FacMARNN", "factors"=>10),
		Dict("numhidden"=>15, "replay_size"=>10000, "cell"=>"FacMAGRU", "factors"=>10)]
	boxplot(data_fac_sens, args_list; label_idx="cell", color = [cell_colors["FacMARNN"] cell_colors["FacMAGRU"]])
	dotplot!(data_fac_sens, args_list; label_idx="cell", color = [cell_colors["FacMARNN"] cell_colors["FacMAGRU"]])
	args_list_l = [
		Dict("numhidden"=>17, "truncation"=>12, "cell"=>"GRU"),
		Dict("numhidden"=>17, "truncation"=>12, "cell"=>"AAGRU"),
		Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MAGRU"),
		Dict("numhidden"=>30, "truncation"=>12, "cell"=>"RNN"),
		Dict("numhidden"=>30, "truncation"=>12, "cell"=>"AARNN"),
		Dict("numhidden"=>18, "truncation"=>12, "cell"=>"MARNN")]
	boxplot!(data_10_dist, args_list_l; 
		label_idx="cell", 
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_10_dist, args_list_l; label_idx="cell",
				color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),)
	# dotplot!(data_10_dist, args_list_l; 
	# 	label_idx="cell", 
	# 	color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
end

# ╔═╡ 7f630af5-a608-47d3-be13-589b9731798e
ic_fac_init, dd_fac_init = RPU.load_data(at("dir_tmaze_er_fac_rnn_init_rmsprop_10/"))

# ╔═╡ c60ee6c5-85e4-407c-8272-801085296084


# ╔═╡ f297e4f3-5826-4f90-8f24-ae731232f63b
data_fac_init_sens_eta = RPU.get_line_data_for(
	ic_fac_init,
	["numhidden", "cell", "init_style", "replay_size", "factors", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ baf539d2-bdd9-40be-bca7-2af231d7063d
let
	args_list = [
		Dict("numhidden"=>25, "replay_size"=>10000, "cell"=>"FacMARNN", "factors"=>15, "init_style"=>"tensor"),
		Dict("numhidden"=>15, "replay_size"=>10000, "cell"=>"FacMAGRU", "factors"=>15, "init_style"=>"tensor")]
	plot(data_fac_init_sens_eta, args_list; sort_idx="eta", labels=["FacMARNN" "FacMAGRU"])
end

# ╔═╡ 4480eb51-352d-49ff-8181-96e6bf03cab3
data_fac_init_sens = RPU.get_line_data_for(
	ic_fac_init,
	["numhidden", "cell", "replay_size", "factors", "init_style"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 800cd6d3-177f-4542-b4b5-38f24265876a
d = data_fac_init_sens[1]

# ╔═╡ 10a832cc-574b-4012-ab9e-4cd40f9bb9c8
d.swept_params

# ╔═╡ 8500402b-28aa-41ed-96d7-450903bc90d0
let
	plts = []
	rsrs = [10000, 20000]
	for (rs, ins) ∈ Iterators.product(rsrs, dd_fac_init["init_style"])
		
		args_list = [
			Dict("numhidden"=>25, "replay_size"=>rs, "cell"=>"FacMARNN", "init_style"=>ins),
			Dict("numhidden"=>15, "replay_size"=>rs, "cell"=>"FacMAGRU", "init_style"=>ins)]
		push!(plts, plot(data_fac_init_sens, args_list; sort_idx="factors", labels=["FacMARNN" "FacMAGRU"], title="$((rs, ins))", color = 	[cell_colors["FacMARNN"] cell_colors["FacMAGRU"]], lw = 2, legend=nothing))
	end
	plot(plts...)
end

# ╔═╡ 20e3d6d4-bec8-4a42-8e5c-01c6f60600d7
plt_args_list = let
	args_list = [
		Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MAGRU", "eta"=>0.0003125), #num_params = 1303
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"MAGRU", "eta"=>0.0003125), #num_params = 3499
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"AAGRU", "eta"=>0.00125), #num_params = 1053
		Dict("numhidden"=>17, "truncation"=>12, "cell"=>"AAGRU", "eta"=>0.0003125),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AAGRU", "eta"=>0.00125), #num_params = 1703
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"GRU", "eta"=>0.00125),
		Dict("numhidden"=>18, "truncation"=>12, "cell"=>"MARNN", "eta"=>0.0003125), #num_params = 463
		Dict("numhidden"=>30, "truncation"=>12, "cell"=>"AARNN", "eta"=>0.0003125),
		Dict("numhidden"=>30, "truncation"=>12, "cell"=>"RNN", "eta"=>1.953125e-5)
	]

	# FileIO.save("../final_runs/dir_tmaze_10.jld2", "args", args_list)
	
	plt_keys = ["numhidden", "truncation", "cell"]
	[Dict(k=>args_list[i][k] for k ∈ plt_keys) for i in 1:length(args_list)]
end

# ╔═╡ c9ab1109-4513-412c-8b84-9ae02d65acf7


# ╔═╡ 24b79fb2-acfe-47e8-976d-231fa4ce2a10
let	
	params = [
		Dict("cell"=>"FacMAGRU", "numhidden"=>15, "init_style"=>"tensor"),
		Dict("cell"=>"FacMARNN", "numhidden"=>25, "init_style"=>"tensor")
	]
	# dd_fac_init["truncation"] = [12]
	get_final_argument_list(params, data_fac_init_sens, dd_fac_init, ["factors", "replay_size"], ["eta"])
end

# ╔═╡ 7333fe0d-02fe-4d74-9427-95826c485334
ic_20, dd_20 = RPU.load_data(at("dir_tmaze_er_rnn_rmsprop_10_20k/"))

# ╔═╡ 72a17826-a498-4cd5-9523-d20a1bab5c30
data_10_dist_20 = RPU.get_line_data_for(
	ic_20,
	["cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 55e67b77-bd4f-4841-aa16-81d5630a0f0a
let
	plt = plot()
	# args_list = [
	# 	Dict("numhidden"=>15, "replay_size"=>20000, "cell"=>"FacMARNN", "factors"=>10)]
	# violin!(data_fac_sens, args_list; label_idx="cell", color = [cell_colors["FacMARNN"] cell_colors["FacMAGRU"]], linecolor = [cell_colors["FacMARNN"] cell_colors["FacMAGRU"]])
	# boxplot!(data_fac_sens, args_list; label_idx="cell", color = [cell_colors["FacMARNN"] cell_colors["FacMAGRU"]], linecolor=:black, lw=2)

	args_list_l = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU")]
	violin!(data_10_dist_20, args_list_l; label_idx="cell",
				color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :), linecolor=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
	boxplot!(data_10_dist_20, args_list_l; 
		label_idx="cell", 
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=2, ylims=(0.4, 1.0), tickdir=:out, grid=false, linecolor=:black, fillalpha=0.75)

	args_list = [Dict("numhidden"=>15, "replay_size"=>20000, "cell"=>"FacMAGRU", "factors"=>10)]
	
		violin!(data_fac_sens, args_list; label_idx="cell", color = [cell_colors["FacMAGRU"]], linecolor = [cell_colors["FacMAGRU"]])
	boxplot!(data_fac_sens, args_list; label_idx="cell", color = [cell_colors["FacMAGRU"]], linecolor=:black, lw=2)
	
	args_list_l = [		
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN")]
	violin!(data_10_dist_20, args_list_l; label_idx="cell",
				color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :), linecolor=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
	boxplot!(data_10_dist_20, args_list_l; 
		label_idx="cell", 
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=2, ylims=(0.4, 1.0), tickdir=:out, grid=false, linecolor=:black, fillalpha=0.75)
	
	
	args_list = [Dict("numhidden"=>20, "replay_size"=>20000, "cell"=>"FacMARNN", "factors"=>10)]
	
	violin!(data_fac_sens, args_list; label_idx="cell", color = [cell_colors["FacMARNN"]], linecolor = [cell_colors["FacMARNN"]])
	boxplot!(data_fac_sens, args_list; label_idx="cell", color = [cell_colors["FacMARNN"] cell_colors["FacMAGRU"]], linecolor=:black, lw=2)
	
	# savefig("../plots/dir_tmaze_20k_buffer.pdf")
	plt
	# dotplot!(data_10_dist, args_list_l; 
	# 	label_idx="cell", 
	# 	color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
end

# ╔═╡ Cell order:
# ╠═ac8f3d1b-82af-4917-bf1c-d7afc16fc43a
# ╠═f7f500a8-a1e9-11eb-009b-d7afdcade891
# ╠═e0d51e67-63dc-45ea-9092-9965f97660b3
# ╠═e53d7b29-788c-469c-9d44-573f996fa5e7
# ╠═0c746c1e-ea39-4415-a1b1-d7124b886f98
# ╟─1886bf05-f4be-4160-b61c-edf186a7f3cb
# ╠═fe50ffef-b691-47b5-acf8-8378fbf860a1
# ╠═842b3fbc-34aa-452d-81fb-2ade57dedecb
# ╠═cc4a219d-9118-4c34-93ce-317afc837f6c
# ╟─1756d1cc-1a88-4442-b55a-fdbb44f56313
# ╠═0fc6fd35-5a24-4aaf-9ebb-6c4af1ba259b
# ╟─6211a38a-7b53-4054-970e-c29ad17de646
# ╟─e822182e-b485-4a95-a08c-efe1540ff6ad
# ╟─c1b80cfd-dbb8-41c5-a778-66b112e1c091
# ╟─e4cc9109-d8b1-4bff-a176-3627e24ab757
# ╟─0eb4c818-b533-4695-a0c7-53e72023281f
# ╟─4d31c08d-d532-447d-931e-6eec9c7882f5
# ╟─5c3ddaed-47b6-465b-b4c5-bc0fb30c8601
# ╟─37af922e-ed3c-4aec-a4cf-c403c49a9ba9
# ╠═6293dc31-1428-4315-bf56-d7a20ee2a17d
# ╠═eefd2ccc-ce9b-4cf8-991f-a58f2f932e99
# ╠═eab5c7a0-8052-432d-977e-68b967baf5ca
# ╠═55d1416b-d580-4112-827a-30504c21f397
# ╠═467ed417-cf69-493d-b21c-3fc4d1fb9907
# ╠═2040d613-0ea8-48ce-937c-9180073812ea
# ╠═7c1f8e58-4bfa-4da2-833d-0cc2a4ec74a4
# ╠═c6c301bd-d3c4-4d63-9b7b-46ea9ac7bc87
# ╠═b3916560-ccd5-408b-9d8a-9ecbdbe4b18d
# ╠═c05c7ffa-53cd-46bf-a661-886784eecc05
# ╠═cc730813-539b-418d-8da6-67f83bd05d70
# ╠═fc961ab4-c7c6-4e7b-91be-3e5fbb18d667
# ╠═db83dce1-29d5-42f8-b226-4412ce63c8f1
# ╠═305f8ac8-8f5f-4ec4-84a6-867f69a8887c
# ╠═533cba3d-7fc5-4d66-b545-b15ffc8ab6d8
# ╠═39752286-a6db-439d-aca0-1be4821bfc2b
# ╠═7d611b39-f9a8-43e4-951e-9d812cbd4384
# ╠═a8949e02-61f5-456a-abee-2bad91d2df05
# ╠═4b654ad2-93cf-455e-9c7b-982766560205
# ╠═2dbcb518-2fda-44c4-bfc0-b422a8da9c35
# ╠═7f630af5-a608-47d3-be13-589b9731798e
# ╠═c60ee6c5-85e4-407c-8272-801085296084
# ╠═f297e4f3-5826-4f90-8f24-ae731232f63b
# ╠═baf539d2-bdd9-40be-bca7-2af231d7063d
# ╠═4480eb51-352d-49ff-8181-96e6bf03cab3
# ╠═800cd6d3-177f-4542-b4b5-38f24265876a
# ╠═10a832cc-574b-4012-ab9e-4cd40f9bb9c8
# ╠═8500402b-28aa-41ed-96d7-450903bc90d0
# ╠═20e3d6d4-bec8-4a42-8e5c-01c6f60600d7
# ╠═c9ab1109-4513-412c-8b84-9ae02d65acf7
# ╠═24b79fb2-acfe-47e8-976d-231fa4ce2a10
# ╠═7333fe0d-02fe-4d74-9427-95826c485334
# ╠═72a17826-a498-4cd5-9523-d20a1bab5c30
# ╠═55e67b77-bd4f-4841-aa16-81d5630a0f0a
