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
at(dir) = joinpath("../../local_data/lunar_lander/", dir)

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
ic_dir, dd_dir = RPU.load_data(at("lunar_lander_deep_a/"))

# ╔═╡ ffbf79df-934a-4766-a797-a17dfd53c487
ic_dir_combo, dd_dir_combo = RPU.load_data(at("lunar_lander_combo/"))

# ╔═╡ c7f9c051-5024-4ff8-a19a-8cad57c05123
ic_dir_aagru_final, dd_dir_aagru_final = RPU.load_data(at("final_lunar_lander_er_relu_rmsprop_os6_sc2_aagru_4M/"))

# ╔═╡ 91bc59d5-7eee-4c34-ac50-a5c9e327c9eb
ic_dir_magru_final, dd_dir_magru_final = RPU.load_data(at("final_lunar_lander_er_relu_rmsprop_os6_sc2_magru_4M/"))

# ╔═╡ 6211a38a-7b53-4054-970e-c29ad17de646
ic_dir[1].parsed_args["steps"]

# ╔═╡ e822182e-b485-4a95-a08c-efe1540ff6ad
data = RPU.get_line_data_for(
	ic_dir,
	["numhidden", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 100))

# ╔═╡ c1b80cfd-dbb8-41c5-a778-66b112e1c091
md"""
NumHidden: $(@bind nh_dir Select(string.(dd_dir["numhidden"])))
Cell: $(@bind cells_dir MultiSelect(dd_dir["cell"]))
"""

# ╔═╡ e4cc9109-d8b1-4bff-a176-3627e24ab757
let 
	plt = nothing
	nh = parse(Int, nh_dir)
	plt = plot()
	for cell ∈ cells_dir
		plot!(plt, 
			data, 
			Dict("numhidden"=>nh, "cell"=>cell), 
			palette=RPU.custom_colorant, legend=:bottomright)
	end
	plt
end

# ╔═╡ 0eb4c818-b533-4695-a0c7-53e72023281f
let 
	args_list = [
		Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MAGRU"),
		Dict("numhidden"=>17, "truncation"=>12, "cell"=>"AAGRU"),
		Dict("numhidden"=>30, "truncation"=>12, "cell"=>"AARNN"),
	]
	
	plot(data, args_list, palette=RPU.custom_colorant)
end

# ╔═╡ 5c3ddaed-47b6-465b-b4c5-bc0fb30c8601
data_sens = RPU.get_line_data_for(
	ic_dir,
	["cell", "numhidden", "truncation", "internal", "replay_size", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 37af922e-ed3c-4aec-a4cf-c403c49a9ba9
let
	plot(data_sens, 
	 	 Dict("internal"=>14, "numhidden"=>10, "replay_size"=>20000, "truncation"=>12, "cell"=>"AAGRU"); 
		 sort_idx="eta", z=1.97, lw=2, palette=ReproducePlotUtils.custom_colorant)
	plot!(data_sens, 
	 	  Dict("internal"=>50, "numhidden"=>10, "replay_size"=>20000, "truncation"=>12, "cell"=>"AAGRU"); 
	 	  sort_idx="eta", z=1.97, lw=2, palette=ReproducePlotUtils.custom_colorant)
end

# ╔═╡ fc961ab4-c7c6-4e7b-91be-3e5fbb18d667
data_dist = RPU.get_line_data_for(
	ic_dir,
	["numhidden", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))

# ╔═╡ f4ee49dd-b164-4095-8303-66a0944f09d7
data_dist_combo = RPU.get_line_data_for(
	ic_dir_combo,
	["numhidden", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))

# ╔═╡ c3e8037d-fd67-4609-9c54-716d3707d25c
data_dist_aagru_final = RPU.get_line_data_for(
	ic_dir_aagru_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))

# ╔═╡ 9ff5c49c-c5c5-4e54-9cdd-99a5a150a280
data_dist_magru_final = RPU.get_line_data_for(
	ic_dir_magru_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))

# ╔═╡ b1beea3f-28c4-4d6e-9738-601ac71e51df
let
	boxplot(data_dist, Dict("numhidden"=>64, "cell"=>"MAGRU"); 
		label = "DMAGRU",
		color=cell_colors["MAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist, Dict("numhidden"=>64, "cell"=>"MAGRU"); 
		label = "DMAGRU",
		color=cell_colors["MAGRU"])
	
	boxplot!(data_dist, Dict("numhidden"=>152, "cell"=>"AAGRU"); 
		label = "DAAGRU",
		color=cell_colors["AAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist, Dict("numhidden"=>152, "cell"=>"AAGRU"); 
		label = "DAAGRU",
		color=cell_colors["AAGRU"])
	
	
	boxplot!(data_dist_magru_final, Dict(); 
		label = "MAGRU",
		color=cell_colors["MAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_magru_final, Dict(); 
		label = "MAGRU",
		color=cell_colors["MAGRU"])
	
	boxplot!(data_dist_aagru_final, Dict(); 
		label = "AAGRU",
		color=cell_colors["AAGRU"],
		legend=false, lw=1.5, ylims=(-50, 150), tickdir=:out, grid=false)
	dotplot!(data_dist_aagru_final, Dict(); 
		label = "AAGRU",
		color=cell_colors["AAGRU"])
	
# 	boxplot(data_dist, Dict("numhidden"=114, "cell"=>"CaddRNN"); 
# 		label = "CaddRNN",
# 		color=cell_colors["FacMARNN"],
# 		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
# 	dotplot!(data_dist, Dict("numhidden"=114, "cell"=>"CaddRNN"); 
# 		label = "CaddRNN",
# 		color=cell_colors["FacMARNN"])
	
	# boxplot!(data_dist_combo, Dict("numhidden"=>114, "cell"=>"CaddRNN"); 
	# 	label = "CaR",
	# 	color=cell_colors["FacMARNN"],
	# 	legend=false, lw=1.5, ylims=(-50, 150), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo, Dict("numhidden"=>114, "cell"=>"CaddRNN"); 
	# 	label = "CaR",
	# 	color=cell_colors["FacMARNN"])
	
	
	boxplot!(data_dist_combo, Dict("numhidden"=>54, "cell"=>"CaddGRU"); 
		label = "CaG",
		color=cell_colors["FacMAGRU"],
		legend=false, lw=1.5, ylims=(-50, 180), tickdir=:out, grid=false)
	dotplot!(data_dist_combo, Dict("numhidden"=>54, "cell"=>"CaddGRU"); 
		label = "CaG",
		color=cell_colors["FacMAGRU"])
	
	boxplot!(data_dist_combo, Dict("numhidden"=>102, "cell"=>"CaddAAGRU"); 
		label = "CaAAG",
		color=cell_colors["AAGRU"],
		legend=false, lw=1.5, ylims=(-50, 150), tickdir=:out, grid=false)
	dotplot!(data_dist_combo, Dict("numhidden"=>102, "cell"=>"CaddAAGRU"); 
		label = "CaAAG",
		color=cell_colors["AAGRU"])
	
	
	boxplot!(data_dist_combo, Dict("numhidden"=>38, "cell"=>"CaddMAGRU"); 
		label = "CaMAG",
		color=cell_colors["MAGRU"],
		legend=false, lw=1.5, ylims=(-50, 180), tickdir=:out, grid=false)
	dotplot!(data_dist_combo, Dict("numhidden"=>38, "cell"=>"CaddMAGRU"); 
		label = "CaMAG",
		color=cell_colors["MAGRU"])
	
	
	# boxplot!(data_dist_combo, Dict("numhidden"=>92, "cell"=>"CcatRNN"); 
	# 	label = "CcR",
	# 	color=cell_colors["FacMARNN"],
	# 	legend=false, lw=1.5, ylims=(-50, 150), tickdir=:out, grid=false)
	# dotplot!(data_dist_combo, Dict("numhidden"=>92, "cell"=>"CcatRNN"); 
	# 	label = "CcR",
	# 	color=cell_colors["FacMARNN"])
	
	
	boxplot!(data_dist_combo, Dict("numhidden"=>45, "cell"=>"CcatGRU"); 
		label = "CcG",
		color=cell_colors["FacMAGRU"],
		legend=false, lw=1.5, ylims=(-50, 200), tickdir=:out, grid=false)
	dotplot!(data_dist_combo, Dict("numhidden"=>45, "cell"=>"CcatGRU"); 
		label = "CcG",
		color=cell_colors["FacMAGRU"])
	
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
# ╠═ffbf79df-934a-4766-a797-a17dfd53c487
# ╠═c7f9c051-5024-4ff8-a19a-8cad57c05123
# ╠═91bc59d5-7eee-4c34-ac50-a5c9e327c9eb
# ╟─6211a38a-7b53-4054-970e-c29ad17de646
# ╠═e822182e-b485-4a95-a08c-efe1540ff6ad
# ╠═c1b80cfd-dbb8-41c5-a778-66b112e1c091
# ╠═e4cc9109-d8b1-4bff-a176-3627e24ab757
# ╠═0eb4c818-b533-4695-a0c7-53e72023281f
# ╟─5c3ddaed-47b6-465b-b4c5-bc0fb30c8601
# ╟─37af922e-ed3c-4aec-a4cf-c403c49a9ba9
# ╠═fc961ab4-c7c6-4e7b-91be-3e5fbb18d667
# ╠═f4ee49dd-b164-4095-8303-66a0944f09d7
# ╠═c3e8037d-fd67-4609-9c54-716d3707d25c
# ╠═9ff5c49c-c5c5-4e54-9cdd-99a5a150a280
# ╠═b1beea3f-28c4-4d6e-9738-601ac71e51df
