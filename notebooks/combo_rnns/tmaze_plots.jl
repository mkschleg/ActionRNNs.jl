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
at(dir) = joinpath("../../local_data/tmaze_er/", dir)

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

# ╔═╡ 538d0e68-c644-4306-a030-a4a1bdefa97b
ic_dir_combo_el, dd_dir_combo_el = RPU.load_data(at("tmaze_er_combo_el/"))

# ╔═╡ 6211a38a-7b53-4054-970e-c29ad17de646
ic_dir[1].parsed_args["steps"]

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

# ╔═╡ b4805558-8a0c-4942-8462-d02b34452222
data_dist_combo_el = RPU.get_line_data_for(
	ic_dir_combo_el,
	["numhidden", "cell", "num_experts"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ b99580cf-a793-4014-ad30-6993439bc61c
let
	boxplot(data_dist_combo_el, Dict("cell"=>"AARNN", "numhidden"=>12, "num_experts"=>0); 
		#label_idx="cell", 
		label = "AAR",
		color=cell_colors["AARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"AARNN", "numhidden"=>12, "num_experts"=>0); 
		#label_idx="cell", 
		label = "AAR",
		color=cell_colors["AARNN"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MARNN", "numhidden"=>6, "num_experts"=>0); 
		#label_idx="cell", 
		label = "MAR",
		color=cell_colors["MARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MARNN", "numhidden"=>6, "num_experts"=>0); 
		#label_idx="cell", 
		label = "MAR",
		color=cell_colors["MARNN"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"CaddRNN", "numhidden"=>5, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaR",
		color=cell_colors["RNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"CaddRNN", "numhidden"=>5, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaR",
		color=cell_colors["RNN"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"CaddElRNN", "numhidden"=>5, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaER",
		color=cell_colors["AARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"CaddElRNN", "numhidden"=>5, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaER",
		color=cell_colors["AARNN"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"CcatRNN", "numhidden"=>4, "num_experts"=>0); 
		label = "CcR11",
		color=cell_colors["FacMARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"CcatRNN", "numhidden"=>4, "num_experts"=>0); 
		label = "CcR11",
		color=cell_colors["FacMARNN"])
	
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>12, "num_experts"=>1); 
		#label_idx="cell", 
		label = "MR1",
		color=cell_colors["MARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>12, "num_experts"=>1); 
		#label_idx="cell", 
		label = "MR1",
		color=cell_colors["MARNN"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>12, "num_experts"=>1); 
		#label_idx="cell", 
		label = "MER1",
		color=cell_colors["FacMARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>12, "num_experts"=>1); 
		#label_idx="cell", 
		label = "MER1",
		color=cell_colors["FacMARNN"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>8, "num_experts"=>2); 
		#label_idx="cell", 
		label = "MR2",
		color=cell_colors["MARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>8, "num_experts"=>2); 
		#label_idx="cell", 
		label = "MR2",
		color=cell_colors["MARNN"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>8, "num_experts"=>2); 
		#label_idx="cell", 
		label = "MER2",
		color=cell_colors["FacMARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>8, "num_experts"=>2); 
		#label_idx="cell", 
		label = "MER2",
		color=cell_colors["FacMARNN"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>6, "num_experts"=>3); 
		#label_idx="cell", 
		label = "MR3",
		color=cell_colors["MARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>6, "num_experts"=>3); 
		#label_idx="cell", 
		label = "MR3",
		color=cell_colors["MARNN"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>6, "num_experts"=>3); 
		#label_idx="cell", 
		label = "MER3",
		color=cell_colors["FacMARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>6, "num_experts"=>3); 
		#label_idx="cell", 
		label = "MER3",
		color=cell_colors["FacMARNN"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>4, "num_experts"=>5); 
		#label_idx="cell", 
		label = "MR5",
		color=cell_colors["MARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixRNN", "numhidden"=>4, "num_experts"=>5); 
		#label_idx="cell", 
		label = "MR5",
		color=cell_colors["MARNN"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>4, "num_experts"=>5); 
		#label_idx="cell", 
		label = "MER5",
		color=cell_colors["FacMARNN"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixElRNN", "numhidden"=>4, "num_experts"=>5); 
		#label_idx="cell", 
		label = "MER5",
		color=cell_colors["FacMARNN"],
		title="TMaze ER RNN, truncation: 12")
	
end

# ╔═╡ d789b99a-7c16-4d67-9fd6-0fe73e302ee6
begin	
	truncation=12
	boxplot(data_dist_combo_el, Dict("cell"=>"AAGRU", "numhidden"=>6, "num_experts"=>0); 
		label="AAG", 
		color=cell_colors["AAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"AAGRU", "numhidden"=>6, "num_experts"=>0); 
		label="AAG", 
		color=cell_colors["AAGRU"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MAGRU", "numhidden"=>3, "num_experts"=>0); 
		label="MAG", 
		color=cell_colors["MAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MAGRU", "numhidden"=>3, "num_experts"=>0); 
		label="MAG", 
		color=cell_colors["MAGRU"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"CaddGRU", "numhidden"=>2, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaG",
		color=cell_colors["GRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"CaddGRU", "numhidden"=>2, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaG",
		color=cell_colors["GRU"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"CaddElGRU", "numhidden"=>2, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaEG",
		color=cell_colors["AAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"CaddElGRU", "numhidden"=>2, "num_experts"=>0); 
		#label_idx="cell", 
		label = "CaEG",
		color=cell_colors["AAGRU"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"CcatGRU", "numhidden"=>2, "num_experts"=>0); 
		label = "CcG6",
		color=cell_colors["FacMAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"CcatGRU", "numhidden"=>2, "num_experts"=>0); 
		label = "CcG6",
		color=cell_colors["FacMAGRU"])

	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>6, "num_experts"=>1); 
		#label_idx="cell", 
		label = "MG1",
		color=cell_colors["MAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>6, "num_experts"=>1); 
		#label_idx="cell", 
		label = "MG1",
		color=cell_colors["MAGRU"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>6, "num_experts"=>1); 
		#label_idx="cell", 
		label = "MEG1",
		color=cell_colors["FacMAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>6, "num_experts"=>1); 
		#label_idx="cell", 
		label = "MEG1",
		color=cell_colors["FacMAGRU"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>4, "num_experts"=>2); 
		#label_idx="cell", 
		label = "MG2",
		color=cell_colors["MAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>4, "num_experts"=>2); 
		#label_idx="cell", 
		label = "MG2",
		color=cell_colors["MAGRU"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>4, "num_experts"=>2); 
		#label_idx="cell", 
		label = "MEG2",
		color=cell_colors["FacMAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>4, "num_experts"=>2); 
		#label_idx="cell", 
		label = "MEG2",
		color=cell_colors["FacMAGRU"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>3, "num_experts"=>3); 
		#label_idx="cell", 
		label = "MG3",
		color=cell_colors["MAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>3, "num_experts"=>3); 
		#label_idx="cell", 
		label = "MG3",
		color=cell_colors["MAGRU"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>3, "num_experts"=>3); 
		#label_idx="cell", 
		label = "MEG3",
		color=cell_colors["FacMAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>3, "num_experts"=>3); 
		#label_idx="cell", 
		label = "MEG3",
		color=cell_colors["FacMAGRU"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>2, "num_experts"=>5); 
		#label_idx="cell", 
		label = "MG5",
		color=cell_colors["MAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixGRU", "numhidden"=>2, "num_experts"=>5); 
		#label_idx="cell", 
		label = "MG5",
		color=cell_colors["MAGRU"])
	
	boxplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>2, "num_experts"=>5); 
		#label_idx="cell", 
		label = "MEG5",
		color=cell_colors["FacMAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_combo_el, Dict("cell"=>"MixElGRU", "numhidden"=>2, "num_experts"=>5); 
		#label_idx="cell", 
		label = "MEG5",
		color=cell_colors["FacMAGRU"], xtickfontsize=8, 
		title="TMaze ER GRU, truncation: 12")
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
# ╠═538d0e68-c644-4306-a030-a4a1bdefa97b
# ╟─6211a38a-7b53-4054-970e-c29ad17de646
# ╠═e822182e-b485-4a95-a08c-efe1540ff6ad
# ╠═025f23f6-040e-44df-8183-915cb4d1ee25
# ╠═c1b80cfd-dbb8-41c5-a778-66b112e1c091
# ╠═e4cc9109-d8b1-4bff-a176-3627e24ab757
# ╠═b4805558-8a0c-4942-8462-d02b34452222
# ╟─b99580cf-a793-4014-ad30-6993439bc61c
# ╟─d789b99a-7c16-4d67-9fd6-0fe73e302ee6
