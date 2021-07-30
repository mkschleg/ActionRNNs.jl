### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 18bfdc06-df2d-11eb-1ef4-0772caed819f
using Revise

# ╔═╡ 60314f42-695e-42b9-8f5f-05bf0211de2a
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 096bd6fc-5746-4b3b-8278-7e36ff1a5e4c
using JLD2

# ╔═╡ c7df63a9-0ca4-4277-b6c3-271a471692e2
const RPU = ReproducePlotUtils

# ╔═╡ 3907310e-62c0-4e1e-9b40-180a1635efe7
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

# ╔═╡ df7909ef-9f09-46c4-b4b1-c88a2db99dc7
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2],
	"GatedRNN" => color_scheme[end-1])

# ╔═╡ 2cb5a984-17e6-43c9-a73b-32fd074907d4
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ 1ba5ea26-4fa5-42ca-b42c-b0b8b347807f
ic_on_f, dd_on_f = RPU.load_data("../../local_data/factored_tensor/final_dir_tmaze_online_rmsprop_10_fac_tensor/")

# ╔═╡ a6eecff0-f022-4110-a9f4-5872710b93f5
ic_bp, dd_bp = let
	ic = ItemCollection("../../local_data/final_dir_tmaze_online_rmsprop_10_t16/")
	subic = search(ic) do ld
		ld.parsed_args["cell"][1:3] !== "Fac"
	end
	subic, diff(subic)
end

# ╔═╡ 92b8b725-54ef-4ea7-8084-0a7f42bb9044
ic_gated, dd_gated = RPU.load_data("../../local_data/gated_rnns/dir_tmaze_gated_online_rmsprop_10/")

# ╔═╡ b7406f9b-618f-485c-b79d-920c83b3d6b3
function get_300k(data)
  ns = 0
  idx = 1
  while ns < 300_000
      ns += data["results"][:total_steps][idx]
      idx += 1
  end
  data["results"][:successes][1:idx]
end

# ╔═╡ f3664996-185b-4a38-ba7e-2053b5837b61
function get_MUE(data, perc)
    mean(data[end-max(1, Int(floor(length(data)*perc))):end])
end

# ╔═╡ 462b98cc-a53f-4430-b016-ad785eb9f3d4
data_bp = RPU.get_line_data_for(
	ic_bp,
	["cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->get_MUE(get_300k(x), 0.1),
	get_data=(x)->get_MUE(get_300k(x), 0.1))

# ╔═╡ 371d0a04-2c89-4326-b0ff-c773a3aadce7
boxplot_data_f = RPU.get_line_data_for(
	ic_on_f,
	["numhidden", "factors", "cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 76cd4ca8-4cae-4d48-9a1f-e150018ecf08
boxplot_data_gated = RPU.get_line_data_for(
	ic_gated,
	["numhidden", "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 7453c283-c299-46f9-a585-b09abf4716c4
params_gru = Dict("cell"=>"FacMAGRU", "numhidden"=>26, "factors"=>21, "init_style"=>"tensor")

# ╔═╡ 25ccbbae-21e5-46e2-954d-0b7880d49e67
params_rnn = Dict("cell"=>"FacMARNN", "numhidden"=>46, "factors"=>24, "init_style"=>"tensor")

# ╔═╡ 054ed0a7-df41-4857-ab30-66ce074c4cba
let	
	plt = plot()
	
	args_list = [
		Dict("numhidden"=>15, "internal"=>43),
		Dict("numhidden"=>20, "internal"=>34),
		Dict("numhidden"=>26, "internal"=>28),
		]
	names = ["GatedRNN 15", "GatedRNN 20", "GatedRNN 26"]
	for i in 1:length(names)
		plt = violin!(boxplot_data_gated, args_list[i], label=names[i], legend=false, color=cell_colors["GatedRNN"], lw=2, linecolor=cell_colors["GatedRNN"])
		
		plt = boxplot!(boxplot_data_gated, args_list[i], label=names[i], color=color=cell_colors["GatedRNN"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=8, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		#plt = dotplot!(boxplot_data, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	plt

	#savefig("../../data/paper_plots/factored_tensor/dir_tmaze_box_plots_300k_steps_tau_16_fac_tensor.pdf")
end

# ╔═╡ 9be0ef18-0311-4922-8dd6-33fb9f4c43c2
let	
	plt = plot()
	
	args_list_1 = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
	]
	names = ["GRU", "AAGRU", "MAGRU"]

	for i in 1:length(args_list_1)
		plt = violin!(data_bp, args_list_1[i], label=names[i], legend=false, color=cell_colors[args_list_1[i]["cell"]], lw=2, linecolor=cell_colors[args_list_1[i]["cell"]])
		
		plt = boxplot!(data_bp, args_list_1[i], label=names[i], color=color=cell_colors[args_list_1[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))	
		
		
		#plt = dotplot!(data_bp, args_list_1[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(0.4, 1.0))
		
	end
	
	args_list = [
		Dict("cell"=>"FacMAGRU", "numhidden"=>26, "factors"=>21)	
		]
	names = ["FacGRU"]
	for i in 1:length(names)
		plt = violin!(boxplot_data_f, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
		plt = boxplot!(boxplot_data_f, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=10, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		#plt = dotplot!(boxplot_data_f, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=10, ylims=(0.4, 1.0))
		
	end	
	
	plt = vline!([6], linestyle=:dot, color=:white, lw=2)
	
	args_list_2 = [
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
	]
	names = ["RNN", "AARNN", "MARNN"]

	for i in 1:length(args_list_2)
		plt = violin!(data_bp, args_list_2[i], label=names[i], legend=false, color=cell_colors[args_list_2[i]["cell"]], lw=2, linecolor=cell_colors[args_list_2[i]["cell"]])
		
		plt = boxplot!(data_bp, args_list_2[i], label=names[i], color=color=cell_colors[args_list_2[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		#plt = dotplot!(data_bp, args_list_2[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(0.4, 1.0))
		
	end
	
	args_list = [
		Dict("cell"=>"FacMARNN", "numhidden"=>46, "factors"=>24)	
		]
	names = ["FacRNN"]
	for i in 1:length(names)
		plt = violin!(boxplot_data_f, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
		plt = boxplot!(boxplot_data_f, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=10, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		#plt = dotplot!(boxplot_data, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	args_list = [
		Dict("numhidden"=>15, "internal"=>43)
		]
	names = ["GatedRNN 15"]
	for i in 1:length(names)
		plt = violin!(boxplot_data_gated, args_list[i], label=names[i], legend=false, color=cell_colors["GatedRNN"], lw=2, linecolor=cell_colors["GatedRNN"])
		
		plt = boxplot!(boxplot_data_gated, args_list[i], label=names[i], color=color=cell_colors["GatedRNN"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=7, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		#plt = dotplot!(boxplot_data, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	plt

	#savefig("../../data/paper_plots/factored_tensor/dir_tmaze_box_plots_300k_steps_tau_16_fac_tensor.pdf")
end

# ╔═╡ Cell order:
# ╠═18bfdc06-df2d-11eb-1ef4-0772caed819f
# ╠═60314f42-695e-42b9-8f5f-05bf0211de2a
# ╠═096bd6fc-5746-4b3b-8278-7e36ff1a5e4c
# ╠═c7df63a9-0ca4-4277-b6c3-271a471692e2
# ╠═3907310e-62c0-4e1e-9b40-180a1635efe7
# ╠═df7909ef-9f09-46c4-b4b1-c88a2db99dc7
# ╠═2cb5a984-17e6-43c9-a73b-32fd074907d4
# ╠═1ba5ea26-4fa5-42ca-b42c-b0b8b347807f
# ╠═a6eecff0-f022-4110-a9f4-5872710b93f5
# ╠═92b8b725-54ef-4ea7-8084-0a7f42bb9044
# ╠═b7406f9b-618f-485c-b79d-920c83b3d6b3
# ╠═f3664996-185b-4a38-ba7e-2053b5837b61
# ╠═462b98cc-a53f-4430-b016-ad785eb9f3d4
# ╠═371d0a04-2c89-4326-b0ff-c773a3aadce7
# ╠═76cd4ca8-4cae-4d48-9a1f-e150018ecf08
# ╠═7453c283-c299-46f9-a585-b09abf4716c4
# ╠═25ccbbae-21e5-46e2-954d-0b7880d49e67
# ╟─054ed0a7-df41-4857-ab30-66ce074c4cba
# ╠═9be0ef18-0311-4922-8dd6-33fb9f4c43c2
