### A Pluto.jl notebook ###
# v0.14.8

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

# ╔═╡ dc2c12f4-e649-11eb-1ecb-dddf9bb514a1
using Revise, Reproduce, ReproducePlotUtils, Plots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto, JLD2

# ╔═╡ da7f3a69-fb9e-4930-b3c1-e0396e7d760a
const RPU = ReproducePlotUtils

# ╔═╡ bed98b56-5a60-4d2e-b618-6319d552b6f5
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

# ╔═╡ 589a3363-48e3-499d-8b02-2500ff1c5ae1
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ e73c879e-94f1-4f0a-a153-81de06b4a6b8
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ 609d1d3a-a6ab-48d0-a141-6b72141eb242
ic_on, dd_on = RPU.load_data("../../local_data/factored_tensor/ringworld_online_rmsprop_10_fac_tensor/")

# ╔═╡ 7e530912-6d7d-42ad-a236-83bb0f94097c
ic_on_f, dd_on_f = RPU.load_data("../../local_data/factored_tensor/final_ringworld_online_rmsprop_10_fac_tensor/")

# ╔═╡ b4e1f0ec-fccc-441f-8aee-31daa29bd3fa
ic_on_rnns_paper, dd_on_rnns_paper = RPU.load_data("../../local_data/RW_final/final_ringworld_online_rmsprop_10_t12/")

# ╔═╡ feeb86c1-9402-476e-8110-1150fb4162eb
ic_on_grus_paper, dd_on_grus_paper = RPU.load_data("../../local_data/RW_final/final_ringworld_online_rmsprop_10_t12_grus/")

# ╔═╡ ea9d1784-1d09-4db5-8012-068480342ccd
data_on = RPU.get_line_data_for(
	ic_on,
	["cell"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ 68af9767-6cd9-4f2c-b558-7a55defc49dc
data_on_f = RPU.get_line_data_for(
	ic_on_f,
	["cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ e2fba0d3-6550-4e8c-bf16-3bd99cb19e74
data_on_rnns_paper = RPU.get_line_data_for(
	ic_on_rnns_paper,
	["cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ 83feddfd-02c3-449f-8aea-d6b71a61ca6d
data_on_grus_paper = RPU.get_line_data_for(
	ic_on_grus_paper,
	["cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ 1c89d19e-284a-4ac6-9ba5-2898c84e86cf
md"""
Cell: $(@bind cells MultiSelect(dd_on["cell"]))
"""

# ╔═╡ 70193cca-97c5-41af-8b45-ba8c172e2909
let
	# plt = nothing
	plt = plot()
	for cell ∈ cells
		plt = plot!(
			  data_on,
			  Dict("cell"=>cell),
			  palette=RPU.custom_colorant, legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Online Ringworld", grid=false, tickdir=:out, legendtitle="Cell")
	end
	plt
end

# ╔═╡ 91ed24af-6cbc-4d33-8bb4-69739d42d82e
let
	# plt = nothing
	plt = plot()
	for cell ∈ cells
		plt = plot!(
			  data_on_f,
			  Dict("cell"=>cell),
			  palette=RPU.custom_colorant, legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Online Ringworld", grid=false, tickdir=:out, legendtitle="Cell")
	end
	plt
end

# ╔═╡ 8781fc6f-737c-4ebc-833f-ab983b7b022e
let
	plt = plot()
	
	arg_list_fac = [
		Dict("cell"=>"FacMARNN"),
	]
	
	label_list_fac = ["FacRNN (nh:20, fac:10, τ:12)"]
	
	for (idx, arg_dict_fac) ∈ enumerate(arg_list_fac)
		plt = plot!(
			  data_on_f,
			  arg_dict_fac,
			  palette=RPU.custom_colorant, color=cell_colors[arg_dict_fac["cell"]], label=label_list_fac[idx])
	end
	
	arg_list = [
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
	]
	
	label_list = ["RNN (nh:20, τ:12)", "AARNN (nh:20, τ:12)", "MARNN (nh:15, τ:12)"]
	
	for (idx, arg_dict) ∈ enumerate(arg_list)
		plt = plot!(
			  data_on_rnns_paper,
			  arg_dict,
			  palette=RPU.custom_colorant, legend=:topright, ylabel="RMSVE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells", grid=false, tickdir=:out, color=cell_colors[arg_dict["cell"]], fillalpha=0.4, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, label=label_list[idx])
	end
	plt
	
	#savefig("../../data/paper_plots/factored_tensor/ringworld_online_learning_curves_rnn_tau_12_fac_tensor.pdf")
end

# ╔═╡ 2b7c0c0c-bf20-45c8-b71d-7e2d8a3085ef
let
	plt = plot()
	
	arg_list_fac = [
		Dict("cell"=>"FacMAGRU"),
	]
	
	label_list_fac = ["FacGRU (nh:12, fac:10, τ:12)"]
	
	for (idx, arg_dict_fac) ∈ enumerate(arg_list_fac)
		plt = plot!(
			  data_on_f,
			  arg_dict_fac,
			  palette=RPU.custom_colorant, color=cell_colors[arg_dict_fac["cell"]], label=label_list_fac[idx])
	end
	
	arg_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
	]
	
	label_list = ["GRU (nh:12, τ:12)", "AAGRU (nh:12, τ:12)", "MAGRU (nh:9, τ:12)"]
	
	for (idx, arg_dict) ∈ enumerate(arg_list)
		plt = plot!(
			  data_on_grus_paper,
			  arg_dict,
			  palette=RPU.custom_colorant, legend=:topright, ylabel="RMSVE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells", grid=false, tickdir=:out, color=cell_colors[arg_dict["cell"]], fillalpha=0.4, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, label=label_list[idx])
	end
	plt
	
		#savefig("../../data/paper_plots/factored_tensor/ringworld_online_learning_curves_gru_tau_12_fac_tensor.pdf")
end

# ╔═╡ Cell order:
# ╠═dc2c12f4-e649-11eb-1ecb-dddf9bb514a1
# ╠═da7f3a69-fb9e-4930-b3c1-e0396e7d760a
# ╠═bed98b56-5a60-4d2e-b618-6319d552b6f5
# ╠═589a3363-48e3-499d-8b02-2500ff1c5ae1
# ╠═e73c879e-94f1-4f0a-a153-81de06b4a6b8
# ╠═609d1d3a-a6ab-48d0-a141-6b72141eb242
# ╠═7e530912-6d7d-42ad-a236-83bb0f94097c
# ╠═b4e1f0ec-fccc-441f-8aee-31daa29bd3fa
# ╠═feeb86c1-9402-476e-8110-1150fb4162eb
# ╠═ea9d1784-1d09-4db5-8012-068480342ccd
# ╠═68af9767-6cd9-4f2c-b558-7a55defc49dc
# ╠═e2fba0d3-6550-4e8c-bf16-3bd99cb19e74
# ╠═83feddfd-02c3-449f-8aea-d6b71a61ca6d
# ╟─1c89d19e-284a-4ac6-9ba5-2898c84e86cf
# ╟─70193cca-97c5-41af-8b45-ba8c172e2909
# ╟─91ed24af-6cbc-4d33-8bb4-69739d42d82e
# ╟─8781fc6f-737c-4ebc-833f-ab983b7b022e
# ╟─2b7c0c0c-bf20-45c8-b71d-7e2d8a3085ef
