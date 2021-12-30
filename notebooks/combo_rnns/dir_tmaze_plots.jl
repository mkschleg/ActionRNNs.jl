### A Pluto.jl notebook ###
# v0.15.1

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

# ╔═╡ ac8f3d1b-82af-4917-bf1c-d7afc16fc43a
let
	import Pkg
	Pkg.activate("../..")
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
# Directional TMaze size 10

"""

# ╔═╡ 0fc6fd35-5a24-4aaf-9ebb-6c4af1ba259b
ic_dir, dd_dir = RPU.load_data(at("dir_tmaze_er_deep_a/"))

# ╔═╡ cefbb7a2-2a1c-43a2-ba9a-77369f80177d
ic_dir_deep_final, dd_dir_deep_final = RPU.load_data(at("dir_tmaze_er_deep_a_final/"))

# ╔═╡ c7f9c051-5024-4ff8-a19a-8cad57c05123
ic_dir_final, dd_dir_final = RPU.load_data(at("final_dir_tmaze_er_rnn_rmsprop_10_2/"))

# ╔═╡ d3c612a3-7e0c-489c-98cc-9fde857cf27a
ic_dir_combo, dd_dir_combo = RPU.load_data(at("dir_tmaze_er_combo_add/"))

# ╔═╡ 6211a38a-7b53-4054-970e-c29ad17de646
ic_dir[1].parsed_args["steps"]

# ╔═╡ 86e1a8ea-d844-469c-9b6c-a40b2f5ae8fb


# ╔═╡ e822182e-b485-4a95-a08c-efe1540ff6ad
data = RPU.get_line_data_for(
	ic_dir_combo,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 3))

# ╔═╡ 025f23f6-040e-44df-8183-915cb4d1ee25
data_steps = RPU.get_line_data_for(
	ic_dir_combo,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_steps),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 3))

# ╔═╡ c1b80cfd-dbb8-41c5-a778-66b112e1c091
md"""
Truncation: $(@bind τ_dir Select(string.(dd_dir_combo["truncation"])))
NumHidden: $(@bind nh_dir Select(string.(dd_dir_combo["numhidden"])))
"""

# ╔═╡ e4cc9109-d8b1-4bff-a176-3627e24ab757
let 
	plt = nothing
	τ = parse(Int, τ_dir)
	nh = parse(Int, nh_dir)
	plt = plot()
	# for nh_ ∈ nh_dir
		plot!(plt, 
			data_steps, 
			Dict("numhidden"=>nh, "truncation"=>τ, "cell"=>"CaddGRU"), 
			palette=RPU.custom_colorant, legend=:bottomright)
	# end
	plt
end

# ╔═╡ 0eb4c818-b533-4695-a0c7-53e72023281f
let 
	args_list = [
		Dict("numhidden"=>10, "truncation"=>12),
		Dict("numhidden"=>15, "truncation"=>12),
		Dict("numhidden"=>20, "truncation"=>12),
	]
	
	plot(data, args_list, palette=RPU.custom_colorant)
end

# ╔═╡ 35a8c33f-12bb-4859-8f49-19a852354559
let 
	args_list = [
		Dict("numhidden"=>10, "truncation"=>20),
		Dict("numhidden"=>15, "truncation"=>20),
		Dict("numhidden"=>20, "truncation"=>20),
	]
	
	plot(data, args_list, palette=RPU.custom_colorant)
end

# ╔═╡ 5c3ddaed-47b6-465b-b4c5-bc0fb30c8601
data_sens = RPU.get_line_data_for(
	ic_dir_combo,
	["numhidden", "truncation", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 37af922e-ed3c-4aec-a4cf-c403c49a9ba9
let
	plot(data_sens, 
	 	 Dict("numhidden"=>10, "truncation"=>12); 
		 sort_idx="eta", z=1.97, lw=2, palette=ReproducePlotUtils.custom_colorant)
	plot!(data_sens, 
	 	  Dict("numhidden"=>15, "truncation"=>20); 
	 	  sort_idx="eta", z=1.97, lw=2, palette=ReproducePlotUtils.custom_colorant)
	plot!(data_sens, 
	  Dict("numhidden"=>20, "truncation"=>12); 
	  sort_idx="eta", z=1.97, lw=2, palette=ReproducePlotUtils.custom_colorant)
end

# ╔═╡ fc961ab4-c7c6-4e7b-91be-3e5fbb18d667
data_dist = RPU.get_line_data_for(
	ic_dir,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 2dfb9219-2e99-4ee0-bf1b-464da02358a2
data_dist_deep_final = RPU.get_line_data_for(
	ic_dir_deep_final,
	["cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ c3e8037d-fd67-4609-9c54-716d3707d25c
data_dist_final = RPU.get_line_data_for(
	ic_dir_final,
	["numhidden", "cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ ff776339-f8a3-4204-962f-da9fd8b5bbe4
data_dist_combo = RPU.get_line_data_for(
	ic_dir_combo,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ f2c005aa-e4d6-449c-a345-ef2fea5ee03e
let
	boxplot(data_dist_deep_final, Dict("cell"=>"MAGRU"); 
		#label_idx="cell", 
		label = "DMAG",
		color=cell_colors["MAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_deep_final, Dict("cell"=>"MAGRU"); 
		#label_idx="cell", 
		label = "DMAG",
		color=cell_colors["MAGRU"])
	
	boxplot!(data_dist_deep_final, Dict("cell"=>"AAGRU"); 
		#label_idx="cell", 
		label = "DAAG",
		color=cell_colors["AAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_deep_final, Dict("cell"=>"AAGRU"); 
		#label_idx="cell", 
		label = "DAAG",
		color=cell_colors["AAGRU"])
	
	boxplot!(data_dist_deep_final, Dict("cell"=>"MARNN"); 
		#label_idx="cell", 
		label = "DMAR",
		color=cell_colors["MARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_deep_final, Dict("cell"=>"MARNN"); 
		#label_idx="cell", 
		label = "DMAR",
		color=cell_colors["MARNN"])
	
	boxplot!(data_dist_deep_final, Dict("cell"=>"AARNN"); 
		#label_idx="cell", 
		label = "DAAR",
		color=cell_colors["AARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_deep_final, Dict("cell"=>"AARNN"); 
		#label_idx="cell", 
		label = "DAAR",
		color=cell_colors["AARNN"])
	

	boxplot!(data_dist_final, Dict("numhidden"=>10, "cell"=>"MAGRU"); 
		label="MAG", 
		color=cell_colors["MAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_final, Dict("numhidden"=>10, "cell"=>"MAGRU"); 
		label="MAG", 
		color=cell_colors["MAGRU"])

	boxplot!(data_dist_final, Dict("numhidden"=>17, "cell"=>"AAGRU"); 
		label="AAG", 
		color=cell_colors["AAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_final, Dict("numhidden"=>17, "cell"=>"AAGRU"); 
		label="AAG", 
		color=cell_colors["AAGRU"])
	
	boxplot!(data_dist_final, Dict("numhidden"=>18, "cell"=>"MARNN"); 
		label="MAR", 
		color=cell_colors["MARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_final, Dict("numhidden"=>18, "cell"=>"MARNN"); 
		label="MAR", 
		color=cell_colors["MARNN"])
	
	boxplot!(data_dist_final, Dict("numhidden"=>30, "cell"=>"AARNN"); 
		label="AAR", 
		color=cell_colors["AARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_final, Dict("numhidden"=>30, "cell"=>"AARNN"); 
		label="AAR", 
		color=cell_colors["AARNN"])
	
	
	boxplot!(data_dist_combo, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddRNN"); 
		label = "CaR10",
		color=cell_colors["RNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddRNN"); 
		label = "CaR10",
		color=cell_colors["RNN"])
	
	boxplot!(data_dist_combo, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddRNN"); 
		label = "CaR15",
		color=cell_colors["AARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddRNN"); 
		label = "CaR15",
		color=cell_colors["AARNN"])
	
	boxplot!(data_dist_combo, Dict("numhidden"=>20, "truncation"=>12, "cell"=>"CaddRNN"); 
		label = "CaR20",
		color=cell_colors["MARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo, Dict("numhidden"=>20, "truncation"=>12, "cell"=>"CaddRNN"); 
		label = "CaR20",
		color=cell_colors["MARNN"])

	
# 	boxplot!(data_dist_combo, Dict("numhidden"=>6, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG6",
# 		color=cell_colors["GRU"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo, Dict("numhidden"=>6, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG6",
# 		color=cell_colors["GRU"])
	
# 	boxplot!(data_dist_combo, Dict("numhidden"=>8, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG8",
# 		color=cell_colors["AAGRU"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo, Dict("numhidden"=>8, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG8",
# 		color=cell_colors["AAGRU"])
	
# 	boxplot!(data_dist_combo, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG10",
# 		color=cell_colors["MAGRU"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG10",
# 		color=cell_colors["MAGRU"])
	
# 	boxplot!(data_dist_combo, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG15",
# 		color=cell_colors["FacMAGRU"],
# 		legend=false, lw=1.5, ylims=(0.0, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG15",
# 		color=cell_colors["FacMAGRU"], xtickfontsize=6)
	

	
end

# ╔═╡ b1beea3f-28c4-4d6e-9738-601ac71e51df
let
	boxplot(data_dist, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MAGRU"); 
		#label_idx="cell", 
		label = "DMAGRU",
		color=cell_colors["MAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MAGRU"); 
		#label_idx="cell", 
		label = "DMAGRU",
		color=cell_colors["MAGRU"])
	
	boxplot!(data_dist, Dict("numhidden"=>17, "truncation"=>12, "cell"=>"AAGRU"); 
		#label_idx="cell", 
		label = "DAAGRU",
		color=cell_colors["AAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist, Dict("numhidden"=>17, "truncation"=>12, "cell"=>"AAGRU"); 
		#label_idx="cell", 
		label = "DAAGRU",
		color=cell_colors["AAGRU"])
	
	boxplot!(data_dist, Dict("numhidden"=>18, "truncation"=>12, "cell"=>"MARNN"); 
		#label_idx="cell", 
		label = "DMARNN",
		color=cell_colors["MARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist, Dict("numhidden"=>18, "truncation"=>12, "cell"=>"MARNN"); 
		#label_idx="cell", 
		label = "DMARNN",
		color=cell_colors["MARNN"])
	
	boxplot!(data_dist, Dict("numhidden"=>30, "truncation"=>12, "cell"=>"AARNN"); 
		#label_idx="cell", 
		label = "DAARNN",
		color=cell_colors["AARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist, Dict("numhidden"=>30, "truncation"=>12, "cell"=>"AARNN"); 
		#label_idx="cell", 
		label = "DAARNN",
		color=cell_colors["AARNN"])
	
	args_list_l = [
		Dict("numhidden"=>10, "cell"=>"MAGRU"),
		Dict("numhidden"=>17, "cell"=>"AAGRU"),
		Dict("numhidden"=>18, "cell"=>"MARNN"),
		Dict("numhidden"=>30, "cell"=>"AARNN"),
	]
	boxplot!(data_dist_final, args_list_l; 
		label_idx="cell", 
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_final, args_list_l; 
		label_idx="cell", 
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ e2a991a9-580d-44eb-86ff-468686fcae11
let
	args_list_l = [
		Dict("internal"=>14, "numhidden"=>10, "replay_size"=>20000, "truncation"=>20, "cell"=>"AAGRU"),
		Dict("internal"=>20, "numhidden"=>18, "replay_size"=>20000, "truncation"=>20, "cell"=>"AARNN"),
	]
	boxplot(data_dist, args_list_l; 
		label_idx="cell", 
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist, args_list_l; 
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
ic_fac_init, dd_fac_init = RPU.load_data("../local_data/dir_tmaze_er_fac_rnn_init_rmsprop_10/")

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
ic_20, dd_20 = RPU.load_data("../local_data/dir_tmaze_er_rnn_rmsprop_10_20k/")

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
	
	savefig("../plots/dir_tmaze_20k_buffer.pdf")
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
# ╠═cefbb7a2-2a1c-43a2-ba9a-77369f80177d
# ╠═c7f9c051-5024-4ff8-a19a-8cad57c05123
# ╠═d3c612a3-7e0c-489c-98cc-9fde857cf27a
# ╟─6211a38a-7b53-4054-970e-c29ad17de646
# ╠═86e1a8ea-d844-469c-9b6c-a40b2f5ae8fb
# ╠═e822182e-b485-4a95-a08c-efe1540ff6ad
# ╠═025f23f6-040e-44df-8183-915cb4d1ee25
# ╠═c1b80cfd-dbb8-41c5-a778-66b112e1c091
# ╠═e4cc9109-d8b1-4bff-a176-3627e24ab757
# ╠═0eb4c818-b533-4695-a0c7-53e72023281f
# ╠═35a8c33f-12bb-4859-8f49-19a852354559
# ╟─5c3ddaed-47b6-465b-b4c5-bc0fb30c8601
# ╠═37af922e-ed3c-4aec-a4cf-c403c49a9ba9
# ╠═fc961ab4-c7c6-4e7b-91be-3e5fbb18d667
# ╠═2dfb9219-2e99-4ee0-bf1b-464da02358a2
# ╠═c3e8037d-fd67-4609-9c54-716d3707d25c
# ╠═ff776339-f8a3-4204-962f-da9fd8b5bbe4
# ╠═f2c005aa-e4d6-449c-a345-ef2fea5ee03e
# ╠═b1beea3f-28c4-4d6e-9738-601ac71e51df
# ╠═e2a991a9-580d-44eb-86ff-468686fcae11
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
