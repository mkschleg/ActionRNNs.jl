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
# Directional TMaze size 10

"""

# ╔═╡ 0fc6fd35-5a24-4aaf-9ebb-6c4af1ba259b
ic_dir, dd_dir = RPU.load_data(at("dir_tmaze_er_deep_a/"))

# ╔═╡ cefbb7a2-2a1c-43a2-ba9a-77369f80177d
ic_dir_deep_final, dd_dir_deep_final = RPU.load_data(at("dir_tmaze_er_deep_a_final/"))

# ╔═╡ c7f9c051-5024-4ff8-a19a-8cad57c05123
ic_dir_final, dd_dir_final = RPU.load_data(at("final_dir_tmaze_er_rnn_rmsprop_10_2/"))

# ╔═╡ 538d0e68-c644-4306-a030-a4a1bdefa97b
ic_dir_combo_cat, dd_dir_combo_cat = RPU.load_data(at("dir_tmaze_er_combo_cat/"))

# ╔═╡ f3754707-6a54-4fd8-82bd-43df03c6a57e
ic_dir_combo_add, dd_dir_combo_add = RPU.load_data(at("dir_tmaze_er_combo_add/"))

# ╔═╡ a9777b89-3716-4032-8b17-a12aba186d0c
ic_dir_combo_el, dd_dir_combo_el = RPU.load_data(at("dir_tmaze_er_combo_el/"))

# ╔═╡ 6211a38a-7b53-4054-970e-c29ad17de646
ic_dir_final[1].parsed_args["steps"]

# ╔═╡ e822182e-b485-4a95-a08c-efe1540ff6ad
data = RPU.get_line_data_for(
	ic_dir_combo_cat,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 3))

# ╔═╡ 025f23f6-040e-44df-8183-915cb4d1ee25
data_steps = RPU.get_line_data_for(
	ic_dir_combo_cat,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_steps),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 30))

# ╔═╡ c1b80cfd-dbb8-41c5-a778-66b112e1c091
md"""
Truncation: $(@bind τ_dir Select(string.(dd_dir_combo_cat["truncation"])))
NumHidden: $(@bind nh_dir Select(string.(dd_dir_combo_cat["numhidden"])))
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
			Dict("numhidden"=>nh, "truncation"=>τ, "cell"=>"CcatGRU"), 
			palette=RPU.custom_colorant, legend=:bottomright)
	# end
	plt
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

# ╔═╡ b4805558-8a0c-4942-8462-d02b34452222
data_dist_combo_cat = RPU.get_line_data_for(
	ic_dir_combo_cat,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 9110fc79-d8e9-4f5f-8d37-33f6b467678e
data_dist_combo_add = RPU.get_line_data_for(
	ic_dir_combo_add,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 94c2860e-c291-4bbc-ab5e-85abf598106e
data_dist_combo_el = RPU.get_line_data_for(
	ic_dir_combo_el,
	["numhidden", "truncation", "cell", "num_experts"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 9ffd1176-dc7d-4fb3-b66a-13bc85b5413b
let
	truncation = 12
	boxplot(data_dist_final, Dict("numhidden"=>30, "cell"=>"AARNN"); 
		label="AAR", 
		color=cell_colors["AARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_final, Dict("numhidden"=>30, "cell"=>"AARNN"); 
		label="AAR", 
		color=cell_colors["AARNN"])
	
	boxplot!(data_dist_final, Dict("numhidden"=>18, "cell"=>"MARNN"); 
		label="MAR", 
		color=cell_colors["MARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_final, Dict("numhidden"=>18, "cell"=>"MARNN"); 
		label="MAR", 
		color=cell_colors["MARNN"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"CaddRNN", "numhidden"=>15, "truncation"=>truncation, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaR",
		color=cell_colors["RNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"CaddRNN", "numhidden"=>15, "truncation"=>truncation, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaR",
		color=cell_colors["RNN"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"CaddElRNN", "numhidden"=>15, "truncation"=>truncation, "num_experts"=>0);
		#label_idx="cell", 
		label = "CaER",
		color=cell_colors["AARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"CaddElRNN", "numhidden"=>15, "truncation"=>truncation, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaER",
		color=cell_colors["AARNN"])
	
	boxplot!(data_dist_combo_cat, Dict("numhidden"=>11, "truncation"=>12, "cell"=>"CcatRNN"); 
		label = "CcR11",
		color=cell_colors["FacMARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_cat, Dict("numhidden"=>11, "truncation"=>12, "cell"=>"CcatRNN"); 
		label = "CcR11",
		color=cell_colors["FacMARNN"])
	
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>30, "truncation"=>truncation, "num_experts"=>1); 
	# 	#label_idx="cell", 
	# 	label = "MR1",
	# 	color=cell_colors["MARNN"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>30, "truncation"=>truncation, "num_experts"=>1); 
	# 	#label_idx="cell", 
	# 	label = "MR1",
	# 	color=cell_colors["MARNN"])
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>30, "truncation"=>truncation, "num_experts"=>1); 
	# 	#label_idx="cell", 
	# 	label = "MER1",
	# 	color=cell_colors["FacMARNN"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>30, "truncation"=>truncation, "num_experts"=>1); 
	# 	#label_idx="cell", 
	# 	label = "MER1",
	# 	color=cell_colors["FacMARNN"])
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>21, "truncation"=>truncation, "num_experts"=>2); 
	# 	#label_idx="cell", 
	# 	label = "MR2",
	# 	color=cell_colors["MARNN"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>21, "truncation"=>truncation, "num_experts"=>2); 
	# 	#label_idx="cell", 
	# 	label = "MR2",
	# 	color=cell_colors["MARNN"])
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>21, "truncation"=>truncation, "num_experts"=>2); 
	# 	#label_idx="cell", 
	# 	label = "MER2",
	# 	color=cell_colors["FacMARNN"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>21, "truncation"=>truncation, "num_experts"=>2); 
	# 	#label_idx="cell", 
	# 	label = "MER2",
	# 	color=cell_colors["FacMARNN"])
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>17, "truncation"=>truncation, "num_experts"=>3); 
	# 	#label_idx="cell", 
	# 	label = "MR3",
	# 	color=cell_colors["MARNN"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>17, "truncation"=>truncation, "num_experts"=>3); 
	# 	#label_idx="cell", 
	# 	label = "MR3",
	# 	color=cell_colors["MARNN"])
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>17, "truncation"=>truncation, "num_experts"=>3); 
	# 	#label_idx="cell", 
	# 	label = "MER3",
	# 	color=cell_colors["FacMARNN"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>17, "truncation"=>truncation, "num_experts"=>3); 
	# 	#label_idx="cell", 
	# 	label = "MER3",
	# 	color=cell_colors["FacMARNN"])
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>12, "truncation"=>truncation, "num_experts"=>5); 
	# 	#label_idx="cell", 
	# 	label = "MR5",
	# 	color=cell_colors["MARNN"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>12, "truncation"=>truncation, "num_experts"=>5); 
	# 	#label_idx="cell", 
	# 	label = "MR5",
	# 	color=cell_colors["MARNN"])
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>12, "truncation"=>truncation, "num_experts"=>5); 
	# 	#label_idx="cell", 
	# 	label = "MER5",
	# 	color=cell_colors["FacMARNN"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>12, "truncation"=>truncation, "num_experts"=>5); 
	# 	#label_idx="cell", 
	# 	label = "MER5",
	# 	color=cell_colors["FacMARNN"],
	# 	title="Dir TMaze ER RNN, truncation: $(truncation)")
	
end

# ╔═╡ 7103c832-4bb1-46e1-bec0-9905a45aa18f
begin	
	truncation=12
	boxplot(data_dist_final, Dict("numhidden"=>17, "cell"=>"AAGRU"); 
		label="AAG", 
		color=cell_colors["AAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_final, Dict("numhidden"=>17, "cell"=>"AAGRU"); 
		label="AAG", 
		color=cell_colors["AAGRU"])
	
	boxplot!(data_dist_final, Dict("numhidden"=>10, "cell"=>"MAGRU"); 
		label="MAG", 
		color=cell_colors["MAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_final, Dict("numhidden"=>10, "cell"=>"MAGRU"); 
		label="MAG", 
		color=cell_colors["MAGRU"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"CaddGRU", "numhidden"=>8, "truncation"=>truncation, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaG",
		color=cell_colors["GRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"CaddGRU", "numhidden"=>8, "truncation"=>truncation, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaG",
		color=cell_colors["GRU"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"CaddElGRU", "numhidden"=>8, "truncation"=>truncation, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaEG",
		color=cell_colors["AAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"CaddElGRU", "numhidden"=>8, "truncation"=>truncation, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaEG",
		color=cell_colors["AAGRU"])
	
	boxplot!(data_dist_combo_cat, Dict("numhidden"=>4, "truncation"=>12, "cell"=>"CcatGRU"); 
		label = "CcG6",
		color=cell_colors["FacMAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_cat, Dict("numhidden"=>4, "truncation"=>12, "cell"=>"CcatGRU"); 
		label = "CcG6",
		color=cell_colors["FacMAGRU"])
	
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>17, "truncation"=>truncation, "num_experts"=>1); 
	# 	#label_idx="cell", 
	# 	label = "MG1",
	# 	color=cell_colors["MAGRU"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>17, "truncation"=>truncation, "num_experts"=>1); 
	# 	#label_idx="cell", 
	# 	label = "MG1",
	# 	color=cell_colors["MAGRU"])
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>17, "truncation"=>truncation, "num_experts"=>1); 
	# 	#label_idx="cell", 
	# 	label = "MEG1",
	# 	color=cell_colors["FacMAGRU"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>17, "truncation"=>truncation, "num_experts"=>1); 
	# 	#label_idx="cell", 
	# 	label = "MEG1",
	# 	color=cell_colors["FacMAGRU"])
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>11, "truncation"=>truncation, "num_experts"=>2); 
	# 	#label_idx="cell", 
	# 	label = "MG2",
	# 	color=cell_colors["MAGRU"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>11, "truncation"=>truncation, "num_experts"=>2); 
	# 	#label_idx="cell", 
	# 	label = "MG2",
	# 	color=cell_colors["MAGRU"])
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>11, "truncation"=>truncation, "num_experts"=>2); 
	# 	#label_idx="cell", 
	# 	label = "MEG2",
	# 	color=cell_colors["FacMAGRU"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>11, "truncation"=>truncation, "num_experts"=>2); 
	# 	#label_idx="cell", 
	# 	label = "MEG2",
	# 	color=cell_colors["FacMAGRU"])
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>9, "truncation"=>truncation, "num_experts"=>3); 
	# 	#label_idx="cell", 
	# 	label = "MG3",
	# 	color=cell_colors["MAGRU"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>9, "truncation"=>truncation, "num_experts"=>3); 
	# 	#label_idx="cell", 
	# 	label = "MG3",
	# 	color=cell_colors["MAGRU"])
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>9, "truncation"=>truncation, "num_experts"=>3); 
	# 	#label_idx="cell", 
	# 	label = "MEG3",
	# 	color=cell_colors["FacMAGRU"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>9, "truncation"=>truncation, "num_experts"=>3); 
	# 	#label_idx="cell", 
	# 	label = "MEG3",
	# 	color=cell_colors["FacMAGRU"])
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>6, "truncation"=>truncation, "num_experts"=>5); 
	# 	#label_idx="cell", 
	# 	label = "MG5",
	# 	color=cell_colors["MAGRU"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>6, "truncation"=>truncation, "num_experts"=>5); 
	# 	#label_idx="cell", 
	# 	label = "MG5",
	# 	color=cell_colors["MAGRU"])
	
	# boxplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>6, "truncation"=>truncation, "num_experts"=>5); 
	# 	#label_idx="cell", 
	# 	label = "MEG5",
	# 	color=cell_colors["FacMAGRU"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>6, "truncation"=>truncation, "num_experts"=>5); 
	# 	#label_idx="cell", 
	# 	label = "MEG5",
	# 	color=cell_colors["FacMAGRU"], xtickfontsize=8, 
	# 	title="Dir TMaze ER GRU, truncation: $(truncation)")
end

# ╔═╡ 78102549-6117-4348-bff2-bbc52f92fc80
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
	
	
# 	boxplot!(data_dist_combo_add, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddRNN"); 
# 		label = "CaR10",
# 		color=cell_colors["RNN"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_add, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddRNN"); 
# 		label = "CaR10",
# 		color=cell_colors["RNN"])
	
	boxplot!(data_dist_combo_add, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddRNN"); 
		label = "CaR15",
		color=cell_colors["FacMARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_add, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddRNN"); 
		label = "CaR15",
		color=cell_colors["FacMARNN"])
	
# 	boxplot!(data_dist_combo_add, Dict("numhidden"=>20, "truncation"=>12, "cell"=>"CaddRNN"); 
# 		label = "CaR20",
# 		color=cell_colors["MARNN"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_add, Dict("numhidden"=>20, "truncation"=>12, "cell"=>"CaddRNN"); 
# 		label = "CaR20",
# 		color=cell_colors["MARNN"])

	
# 	boxplot!(data_dist_combo_add, Dict("numhidden"=>6, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG6",
# 		color=cell_colors["GRU"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_add, Dict("numhidden"=>6, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG6",
# 		color=cell_colors["GRU"])
	
	boxplot!(data_dist_combo_add, Dict("numhidden"=>8, "truncation"=>12, "cell"=>"CaddGRU"); 
		label = "CaG8",
		color=cell_colors["FacMAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_add, Dict("numhidden"=>8, "truncation"=>12, "cell"=>"CaddGRU"); 
		label = "CaG8",
		color=cell_colors["FacMAGRU"])
	
	# boxplot!(data_dist_combo_add, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddGRU"); 
	# 	label = "CaG10",
	# 	color=cell_colors["FacMAGRU"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_add, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddGRU"); 
	# 	label = "CaG10",
	# 	color=cell_colors["FacMAGRU"])
	
	# boxplot!(data_dist_combo_add, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddGRU"); 
	# 	label = "CaG15",
	# 	color=cell_colors["FacMAGRU"],
	# 	legend=false, lw=1.5, ylims=(0.0, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_add, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddGRU"); 
	# 	label = "CaG15",
	# 	color=cell_colors["FacMAGRU"], xtickfontsize=6)

	
	# boxplot!(data_dist_combo_cat, Dict("numhidden"=>8, "truncation"=>12, "cell"=>"CcatRNN"); 
	# 	label = "CcR8",
	# 	color=cell_colors["FacMARNN"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_cat, Dict("numhidden"=>8, "truncation"=>12, "cell"=>"CcatRNN"); 
	# 	label = "CcR8",
	# 	color=cell_colors["FacMARNN"])
	
	boxplot!(data_dist_combo_cat, Dict("numhidden"=>11, "truncation"=>12, "cell"=>"CcatRNN"); 
		label = "CcR11",
		color=cell_colors["FacMARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_cat, Dict("numhidden"=>11, "truncation"=>12, "cell"=>"CcatRNN"); 
		label = "CcR11",
		color=cell_colors["FacMARNN"])
	
# 	boxplot!(data_dist_combo_cat, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CcatRNN"); 
# 		label = "CcR15",
# 		color=cell_colors["MARNN"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_cat, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CcatRNN"); 
# 		label = "CcR15",
# 		color=cell_colors["MARNN"], xtickfontsize=6)
	

	boxplot!(data_dist_combo_cat, Dict("numhidden"=>4, "truncation"=>12, "cell"=>"CcatGRU"); 
		label = "CcG6",
		color=cell_colors["FacMAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_cat, Dict("numhidden"=>4, "truncation"=>12, "cell"=>"CcatGRU"); 
		label = "CcG6",
		color=cell_colors["FacMAGRU"])
	
	# boxplot!(data_dist_combo_cat, Dict("numhidden"=>6, "truncation"=>12, "cell"=>"CcatGRU"); 
	# 	label = "CcG8",
	# 	color=cell_colors["AAGRU"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_cat, Dict("numhidden"=>6, "truncation"=>12, "cell"=>"CcatGRU"); 
	# 	label = "CcG8",
	# 	color=cell_colors["FacMAGRU"])
	
	# boxplot!(data_dist_combo_cat, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CcatGRU"); 
	# 	label = "CcG10",
	# 	color=cell_colors["MAGRU"],
	# 	legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo_cat, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CcatGRU"); 
	# 	label = "CcG10",
	# 	color=cell_colors["MAGRU"])
	
end

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
	
	
	boxplot!(data_dist_combo_add, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddRNN"); 
		label = "CaR10",
		color=cell_colors["RNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_add, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddRNN"); 
		label = "CaR10",
		color=cell_colors["RNN"])
	
	boxplot!(data_dist_combo_add, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddRNN"); 
		label = "CaR15",
		color=cell_colors["AARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_add, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddRNN"); 
		label = "CaR15",
		color=cell_colors["AARNN"])
	
	boxplot!(data_dist_combo_add, Dict("numhidden"=>20, "truncation"=>12, "cell"=>"CaddRNN"); 
		label = "CaR20",
		color=cell_colors["MARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_add, Dict("numhidden"=>20, "truncation"=>12, "cell"=>"CaddRNN"); 
		label = "CaR20",
		color=cell_colors["MARNN"])

	
# 	boxplot!(data_dist_combo_add, Dict("numhidden"=>6, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG6",
# 		color=cell_colors["GRU"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_add, Dict("numhidden"=>6, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG6",
# 		color=cell_colors["GRU"])
	
# 	boxplot!(data_dist_combo_add, Dict("numhidden"=>8, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG8",
# 		color=cell_colors["AAGRU"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_add, Dict("numhidden"=>8, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG8",
# 		color=cell_colors["AAGRU"])
	
# 	boxplot!(data_dist_combo_add, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG10",
# 		color=cell_colors["MAGRU"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_add, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG10",
# 		color=cell_colors["MAGRU"])
	
# 	boxplot!(data_dist_combo_add, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG15",
# 		color=cell_colors["FacMAGRU"],
# 		legend=false, lw=1.5, ylims=(0.0, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_add, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddGRU"); 
# 		label = "CaG15",
# 		color=cell_colors["FacMAGRU"], xtickfontsize=6)

	
	boxplot!(data_dist_combo_cat, Dict("numhidden"=>8, "truncation"=>12, "cell"=>"CcatRNN"); 
		label = "CcR8",
		color=cell_colors["RNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_cat, Dict("numhidden"=>8, "truncation"=>12, "cell"=>"CcatRNN"); 
		label = "CcR8",
		color=cell_colors["RNN"])
	
	boxplot!(data_dist_combo_cat, Dict("numhidden"=>11, "truncation"=>12, "cell"=>"CcatRNN"); 
		label = "CcR11",
		color=cell_colors["AARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_cat, Dict("numhidden"=>11, "truncation"=>12, "cell"=>"CcatRNN"); 
		label = "CcR11",
		color=cell_colors["AARNN"])
	
	boxplot!(data_dist_combo_cat, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CcatRNN"); 
		label = "CcR15",
		color=cell_colors["MARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_cat, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CcatRNN"); 
		label = "CcR15",
		color=cell_colors["MARNN"], xtickfontsize=6)
	

# 	boxplot!(data_dist_combo_cat, Dict("numhidden"=>4, "truncation"=>12, "cell"=>"CcatGRU"); 
# 		label = "CcG6",
# 		color=cell_colors["GRU"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_cat, Dict("numhidden"=>4, "truncation"=>12, "cell"=>"CcatGRU"); 
# 		label = "CcG6",
# 		color=cell_colors["GRU"])
	
# 	boxplot!(data_dist_combo_cat, Dict("numhidden"=>6, "truncation"=>12, "cell"=>"CcatGRU"); 
# 		label = "CcG8",
# 		color=cell_colors["AAGRU"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_cat, Dict("numhidden"=>6, "truncation"=>12, "cell"=>"CcatGRU"); 
# 		label = "CcG8",
# 		color=cell_colors["AAGRU"])
	
# 	boxplot!(data_dist_combo_cat, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CcatGRU"); 
# 		label = "CcG10",
# 		color=cell_colors["MAGRU"],
# 		legend=false, lw=1.5, ylims=(0.0, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_cat, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CcatGRU"); 
# 		label = "CcG10",
# 		color=cell_colors["MAGRU"])
	
end

# ╔═╡ 0b15129e-a281-49bc-9c87-f3db376e01ea
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
	
	
# 	boxplot!(data_dist_combo_add, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddRNN"); 
# 		label = "CaR10",
# 		color=cell_colors["RNN"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_add, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddRNN"); 
# 		label = "CaR10",
# 		color=cell_colors["RNN"])
	
# 	boxplot!(data_dist_combo_add, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddRNN"); 
# 		label = "CaR15",
# 		color=cell_colors["AARNN"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_add, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddRNN"); 
# 		label = "CaR15",
# 		color=cell_colors["AARNN"])
	
# 	boxplot!(data_dist_combo_add, Dict("numhidden"=>20, "truncation"=>12, "cell"=>"CaddRNN"); 
# 		label = "CaR20",
# 		color=cell_colors["MARNN"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_add, Dict("numhidden"=>20, "truncation"=>12, "cell"=>"CaddRNN"); 
# 		label = "CaR20",
# 		color=cell_colors["MARNN"])

	
	boxplot!(data_dist_combo_add, Dict("numhidden"=>6, "truncation"=>12, "cell"=>"CaddGRU"); 
		label = "CaG6",
		color=cell_colors["GRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_add, Dict("numhidden"=>6, "truncation"=>12, "cell"=>"CaddGRU"); 
		label = "CaG6",
		color=cell_colors["GRU"])
	
	boxplot!(data_dist_combo_add, Dict("numhidden"=>8, "truncation"=>12, "cell"=>"CaddGRU"); 
		label = "CaG8",
		color=cell_colors["AAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_add, Dict("numhidden"=>8, "truncation"=>12, "cell"=>"CaddGRU"); 
		label = "CaG8",
		color=cell_colors["AAGRU"])
	
	boxplot!(data_dist_combo_add, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddGRU"); 
		label = "CaG10",
		color=cell_colors["MAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_add, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CaddGRU"); 
		label = "CaG10",
		color=cell_colors["MAGRU"])
	
	boxplot!(data_dist_combo_add, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddGRU"); 
		label = "CaG15",
		color=cell_colors["FacMAGRU"],
		legend=false, lw=1.5, ylims=(0.0, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_add, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CaddGRU"); 
		label = "CaG15",
		color=cell_colors["FacMAGRU"], xtickfontsize=6)

	
# 	boxplot!(data_dist_combo_cat, Dict("numhidden"=>8, "truncation"=>12, "cell"=>"CcatRNN"); 
# 		label = "CcR8",
# 		color=cell_colors["RNN"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_cat, Dict("numhidden"=>8, "truncation"=>12, "cell"=>"CcatRNN"); 
# 		label = "CcR8",
# 		color=cell_colors["RNN"])
	
# 	boxplot!(data_dist_combo_cat, Dict("numhidden"=>11, "truncation"=>12, "cell"=>"CcatRNN"); 
# 		label = "CcR11",
# 		color=cell_colors["AARNN"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_cat, Dict("numhidden"=>11, "truncation"=>12, "cell"=>"CcatRNN"); 
# 		label = "CcR11",
# 		color=cell_colors["AARNN"])
	
# 	boxplot!(data_dist_combo_cat, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CcatRNN"); 
# 		label = "CcR15",
# 		color=cell_colors["MARNN"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist_combo_cat, Dict("numhidden"=>15, "truncation"=>12, "cell"=>"CcatRNN"); 
# 		label = "CcR15",
# 		color=cell_colors["MARNN"], xtickfontsize=6)
	

	boxplot!(data_dist_combo_cat, Dict("numhidden"=>4, "truncation"=>12, "cell"=>"CcatGRU"); 
		label = "CcG6",
		color=cell_colors["GRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_cat, Dict("numhidden"=>4, "truncation"=>12, "cell"=>"CcatGRU"); 
		label = "CcG6",
		color=cell_colors["GRU"])
	
	boxplot!(data_dist_combo_cat, Dict("numhidden"=>6, "truncation"=>12, "cell"=>"CcatGRU"); 
		label = "CcG8",
		color=cell_colors["AAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_cat, Dict("numhidden"=>6, "truncation"=>12, "cell"=>"CcatGRU"); 
		label = "CcG8",
		color=cell_colors["AAGRU"])
	
	boxplot!(data_dist_combo_cat, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CcatGRU"); 
		label = "CcG10",
		color=cell_colors["MAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_cat, Dict("numhidden"=>10, "truncation"=>12, "cell"=>"CcatGRU"); 
		label = "CcG10",
		color=cell_colors["MAGRU"])
	
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

# ╔═╡ c7572279-29a6-4a76-a3b7-39c0e775ca12
md"""


"""

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
# ╠═538d0e68-c644-4306-a030-a4a1bdefa97b
# ╠═f3754707-6a54-4fd8-82bd-43df03c6a57e
# ╠═a9777b89-3716-4032-8b17-a12aba186d0c
# ╠═6211a38a-7b53-4054-970e-c29ad17de646
# ╠═e822182e-b485-4a95-a08c-efe1540ff6ad
# ╠═025f23f6-040e-44df-8183-915cb4d1ee25
# ╠═c1b80cfd-dbb8-41c5-a778-66b112e1c091
# ╠═e4cc9109-d8b1-4bff-a176-3627e24ab757
# ╠═fc961ab4-c7c6-4e7b-91be-3e5fbb18d667
# ╠═2dfb9219-2e99-4ee0-bf1b-464da02358a2
# ╠═c3e8037d-fd67-4609-9c54-716d3707d25c
# ╠═b4805558-8a0c-4942-8462-d02b34452222
# ╠═9110fc79-d8e9-4f5f-8d37-33f6b467678e
# ╠═94c2860e-c291-4bbc-ab5e-85abf598106e
# ╟─9ffd1176-dc7d-4fb3-b66a-13bc85b5413b
# ╟─7103c832-4bb1-46e1-bec0-9905a45aa18f
# ╟─78102549-6117-4348-bff2-bbc52f92fc80
# ╟─f2c005aa-e4d6-449c-a345-ef2fea5ee03e
# ╟─0b15129e-a281-49bc-9c87-f3db376e01ea
# ╟─b1beea3f-28c4-4d6e-9738-601ac71e51df
# ╠═c7572279-29a6-4a76-a3b7-39c0e775ca12
