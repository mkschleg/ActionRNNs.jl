### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ cb8ee5cf-eff9-46dd-b83c-fe27293ad3bb
let
	using Pkg
	Pkg.activate("")
end

# ╔═╡ e3d13806-e4e9-11eb-1851-4b51c9ec551c
using Revise, Reproduce, ReproducePlotUtils, Plots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto, JLD2

# ╔═╡ b70f28c6-9b9c-4abd-9dec-f50dc0b51b90
const RPU = ReproducePlotUtils

# ╔═╡ 288917c9-24ca-41f9-854f-6ad3e805a7ad
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

# ╔═╡ 8698a6e7-3832-4455-8177-94ccf44b09cc
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"FacTucMARNN" => color_scheme[3],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2],
	"FacTucMAGRU" => color_scheme[4])

# ╔═╡ a95b8a7d-9ddd-4e24-b6bb-7557016c033d
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ 2c7e6239-b59f-4e8c-9762-c041ad51547c
ic_on, dd_on = RPU.load_data("../../local_data/tucker_decomp/ringworld_online_rmsprop_10_fac_tuc/")

# ╔═╡ 6a30b206-90ad-4d54-a376-cf27cebbcc82
ic_on_f, dd_on_f = RPU.load_data("../../local_data/factored_tensor/final_ringworld_online_rmsprop_10_fac_tensor/")

# ╔═╡ 7824fc53-18c5-4201-b441-6f3a3f9c28c8
ic_on_rnns_paper, dd_on_rnns_paper = RPU.load_data("../../local_data/RW_final/final_ringworld_online_rmsprop_10_t12/")

# ╔═╡ d9dc73ea-61c8-437b-984c-fdd869392c99
ic_on_grus_paper, dd_on_grus_paper = RPU.load_data("../../local_data/RW_final/final_ringworld_online_rmsprop_10_t12_grus/")

# ╔═╡ c97f3d53-3a18-4aa1-af9d-20a98ab89a68
data_on = RPU.get_line_data_for(
	ic_on,
	["cell", "numhidden", "in_factors", "out_factors"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ 18f69a35-0709-4fde-8893-a4b8870206e9
data_on_f = RPU.get_line_data_for(
	ic_on_f,
	["cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ 5bb3591f-1285-49bb-b36b-7667fbefc61f
data_on_rnns_paper = RPU.get_line_data_for(
	ic_on_rnns_paper,
	["cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ b2add531-45d7-4570-8a3c-0f2e978c35a2
data_on_grus_paper = RPU.get_line_data_for(
	ic_on_grus_paper,
	["cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ 148d184f-3659-4868-92c7-0ebdaa2aa88d
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
			  palette=RPU.custom_colorant, color=cell_colors[arg_dict_fac["cell"]], label=label_list_fac[idx], fillalpha=0.2, lw=3)
	end
	
	arg_list = [
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
	]
	
	label_list = ["RNN (nh:20, τ:12)", "AARNN (nh:20, τ:12)", "MARNN (nh:15, τ:12)"]
	
	for (idx, arg_dict) ∈ enumerate(arg_list)
		plt = plot!(
			  data_on_rnns_paper,
			  arg_dict,
			  palette=RPU.custom_colorant, legend=:topright, ylabel="RMSVE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells", grid=false, tickdir=:out, color=cell_colors[arg_dict["cell"]], fillalpha=0.2, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, label=label_list[idx], lw=3)
	end
	
	arg_list = [
		Dict("cell"=>"FacTucMARNN", "numhidden"=>20, "in_factors"=>7, "out_factors"=> 7),
		Dict("cell"=>"FacTucMARNN", "numhidden"=>15, "in_factors"=>10, "out_factors"=> 10),
	]
	
	label_list = ["FacTucRNN (nh:20, fac:7, τ:12)", "FacTucRNN (nh:20, fac:10, τ:12)"]
	linestyle_list = [:dot, :dash]
	
	for (idx, arg_dict) ∈ enumerate(arg_list)
		plt = plot!(
			  data_on,
			  arg_dict,
			  palette=RPU.custom_colorant, legend=:topright, ylabel="RMSVE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells", grid=false, tickdir=:out, color=cell_colors[arg_dict["cell"]], fillalpha=0.2, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=8, titlefontsize=15, label=label_list[idx], lw=3, linestyle=linestyle_list[idx])
	end
	
	plt
	
	# savefig("../../data/paper_plots/tucker_decomp/ringworld_online_rnn_lc_tuc.pdf")
	
end

# ╔═╡ db4ed5c4-28ba-4ff7-ae0b-11c512e46acc
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
			  palette=RPU.custom_colorant, color=cell_colors[arg_dict_fac["cell"]], label=label_list_fac[idx], lw=3, fillalpha=0.2)
	end
	
	arg_list = [
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
	]
	
	label_list = ["GRU (nh:12, τ:12)", "AAGRU (nh:12, τ:12)", "MAGRU (nh:9, τ:12)"]
	
	for (idx, arg_dict) ∈ enumerate(arg_list)
		plt = plot!(
			  data_on_grus_paper,
			  arg_dict,
			  palette=RPU.custom_colorant, legend=:topright, ylabel="RMSVE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells", grid=false, tickdir=:out, color=cell_colors[arg_dict["cell"]], fillalpha=0.2, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, label=label_list[idx], lw=3)
	end
	
	arg_list = [
		Dict("cell"=>"FacTucMAGRU", "numhidden"=>12, "in_factors"=>8, "out_factors"=> 8),
		Dict("cell"=>"FacTucMAGRU", "numhidden"=>9, "in_factors"=>10, "out_factors"=> 10),
	]
	
	label_list = ["FacTucGRU (nh:12, fac:8, τ:12)", "FacTucGRU (nh:9, fac:10, τ:12)"]
	linestyle_list = [:dot, :dash]
	
	for (idx, arg_dict) ∈ enumerate(arg_list)
		plt = plot!(
			  data_on,
			  arg_dict,
			  palette=RPU.custom_colorant, legend=:topright, ylabel="RMSVE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells", grid=false, tickdir=:out, color=cell_colors[arg_dict["cell"]], fillalpha=0.2, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=8, titlefontsize=15, label=label_list[idx], lw=3, linestyle=linestyle_list[idx])
	end
	
	plt
	
	# savefig("../../data/paper_plots/tucker_decomp/ringworld_online_gru_lc_tuc.pdf")
	
end

# ╔═╡ d1e6ec6c-ceb7-4872-8ee2-f2459381611e
ic_er, dd_er = RPU.load_data("../../local_data/tucker_decomp/ringworld_er_rmsprop_10_fac_tuc_lc/")

# ╔═╡ 2dcb71c9-bf0d-4ee3-8333-703f95a5a9f9
data_er = RPU.get_line_data_for(
	ic_er,
	["cell", "numhidden", "in_factors", "out_factors", "action_factors", "factors"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ f406452e-e85d-46d0-9534-7599112f1f19
let
	plt = plot()
	
	arg_list = [
		Dict("cell"=>"AARNN", "numhidden"=>15, "in_factors"=>0, "out_factors"=> 0, "action_factors"=>0, "factors"=>0),
		Dict("cell"=>"MARNN", "numhidden"=>12, "in_factors"=>0, "out_factors"=> 0, "action_factors"=>0, "factors"=>0),
		Dict("cell"=>"FacMARNN", "numhidden"=>15, "in_factors"=>0, "out_factors"=> 0, "action_factors"=>0, "factors"=>12),
		Dict("cell"=>"FacMARNN", "numhidden"=>12, "in_factors"=>0, "out_factors"=> 0, "action_factors"=>0, "factors"=>14),
		Dict("cell"=>"FacTucMARNN", "numhidden"=>15, "in_factors"=>7, "out_factors"=> 7, "action_factors"=>2, "factors"=>0),
		Dict("cell"=>"FacTucMARNN", "numhidden"=>12, "in_factors"=>9, "out_factors"=> 9, "action_factors"=>2, "factors"=>0),
	]
	
	label_list = ["AARNN (nh:15, τ:6)", "MARNN (nh:12, τ:6)", "FacRNN (nh:15, fac:12, τ:6)", "FacRNN (nh:12, fac:14, τ:6)", "FacTucRNN (nh:15, fac:7, τ:6)", "FacTucRNN (nh:12, fac:9, τ:6)"]
	linestyle_list = [:solid, :solid, :solid, :dash, :solid, :dash]
	
	for (idx, arg_dict) ∈ enumerate(arg_list)
		plt = plot!(
			  data_er,
			  arg_dict,
			  palette=RPU.custom_colorant, legend=:topright, ylabel="RMSVE", xlabel="Steps (Thousands)", ylim=(0, 0.25), title="Ringworld ER RNN Cells", grid=false, tickdir=:out, color=cell_colors[arg_dict["cell"]], fillalpha=0.2, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=8, titlefontsize=15, label=label_list[idx], lw=3, linestyle=linestyle_list[idx])
	end
	
	plt
	
	# savefig("../../data/paper_plots/tucker_decomp/ringworld_er_rnn_lc_tuc.pdf")
	
end

# ╔═╡ 5bcee86e-2baf-40c4-8b85-966bb6bda4f4
let
	plt = plot()
	
	arg_list = [
		Dict("cell"=>"AAGRU", "numhidden"=>12, "in_factors"=>0, "out_factors"=> 0, "action_factors"=>0, "factors"=>0),
		Dict("cell"=>"MAGRU", "numhidden"=>9, "in_factors"=>0, "out_factors"=> 0, "action_factors"=>0, "factors"=>0),
		Dict("cell"=>"FacMAGRU", "numhidden"=>12, "in_factors"=>0, "out_factors"=> 0, "action_factors"=>0, "factors"=>8),
		Dict("cell"=>"FacMAGRU", "numhidden"=>9, "in_factors"=>0, "out_factors"=> 0, "action_factors"=>0, "factors"=>12),
		Dict("cell"=>"FacTucMAGRU", "numhidden"=>12, "in_factors"=>7, "out_factors"=> 7, "action_factors"=>2, "factors"=>0),
		Dict("cell"=>"FacTucMAGRU", "numhidden"=>9, "in_factors"=>9, "out_factors"=> 9, "action_factors"=>2, "factors"=>0),
	]
	
	label_list = ["AAGRU (nh:12, τ:6)", "MAGRU (nh:9, τ:6)", "FacGRU (nh:12, fac:8, τ:6)", "FacGRU (nh:9, fac:12, τ:6)", "FacTucGRU (nh:12, fac:7, τ:6)", "FacTucGRU (nh:9, fac:9, τ:6)"]
	linestyle_list = [:solid, :solid, :solid, :dash, :solid, :dash]
	
	for (idx, arg_dict) ∈ enumerate(arg_list)
		plt = plot!(
			  data_er,
			  arg_dict,
			  palette=RPU.custom_colorant, legend=:topright, ylabel="RMSVE", xlabel="Steps (Thousands)", ylim=(0, 0.25), title="Ringworld ER GRU Cells", grid=false, tickdir=:out, color=cell_colors[arg_dict["cell"]], fillalpha=0.2, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=8, titlefontsize=15, label=label_list[idx], lw=3, linestyle=linestyle_list[idx])
	end
	
	plt
	
	# savefig("../../data/paper_plots/tucker_decomp/ringworld_er_gru_lc_tuc.pdf")
	
end

# ╔═╡ Cell order:
# ╠═cb8ee5cf-eff9-46dd-b83c-fe27293ad3bb
# ╠═e3d13806-e4e9-11eb-1851-4b51c9ec551c
# ╠═b70f28c6-9b9c-4abd-9dec-f50dc0b51b90
# ╟─288917c9-24ca-41f9-854f-6ad3e805a7ad
# ╠═8698a6e7-3832-4455-8177-94ccf44b09cc
# ╠═a95b8a7d-9ddd-4e24-b6bb-7557016c033d
# ╠═2c7e6239-b59f-4e8c-9762-c041ad51547c
# ╠═6a30b206-90ad-4d54-a376-cf27cebbcc82
# ╠═7824fc53-18c5-4201-b441-6f3a3f9c28c8
# ╠═d9dc73ea-61c8-437b-984c-fdd869392c99
# ╠═c97f3d53-3a18-4aa1-af9d-20a98ab89a68
# ╠═18f69a35-0709-4fde-8893-a4b8870206e9
# ╠═5bb3591f-1285-49bb-b36b-7667fbefc61f
# ╠═b2add531-45d7-4570-8a3c-0f2e978c35a2
# ╠═148d184f-3659-4868-92c7-0ebdaa2aa88d
# ╠═db4ed5c4-28ba-4ff7-ae0b-11c512e46acc
# ╠═d1e6ec6c-ceb7-4872-8ee2-f2459381611e
# ╠═2dcb71c9-bf0d-4ee3-8333-703f95a5a9f9
# ╠═f406452e-e85d-46d0-9534-7599112f1f19
# ╠═5bcee86e-2baf-40c4-8b85-966bb6bda4f4
