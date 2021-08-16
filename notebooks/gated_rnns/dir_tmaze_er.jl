### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 0f11b684-a3f9-4d40-9c63-33b28136d4f5
using Revise

# ╔═╡ 95498906-eea1-11eb-26f4-8574b531e0dc
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 78632921-6931-4ca0-9e8a-095a7f2c99b0
using JLD2

# ╔═╡ 8b4861e4-cfbc-4a0e-8246-7d6182c83586
const RPU = ReproducePlotUtils

# ╔═╡ edd86b13-bdb5-4811-8e0a-36230d1162ae
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

# ╔═╡ 1c2a5244-0999-44f2-a952-3f495a8ec136
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2],
	"ActionGatedRNN" => color_scheme[end-1])

# ╔═╡ f197d16b-eb04-408c-a020-9023f2eb02b2
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ a67aba27-40ef-4ba2-816a-6da61a1d523b
ic, dd = RPU.load_data("../../local_data/gated_rnns/dir_tmaze_gated_er_rmsprop_10/")

# ╔═╡ f9c41407-995c-4ca5-b5d4-46bce5f9585a
ic_gaigru, dd_gaigru = RPU.load_data("../../local_data/gated_rnns/dir_tmaze_er_rmsprop_10_gaigru/")

# ╔═╡ 3a1b767e-d09b-4079-b2a5-7e2e189a1600
ic_gaiagru, dd_gaiagru = RPU.load_data("../../local_data/gated_rnns/dir_tmaze_er_rmsprop_10_gaiagru/")

# ╔═╡ 606b1f70-976d-4adc-9d1b-a34835b947e4
ic_gaiarnn, dd_gaiarnn = RPU.load_data("../../local_data/gated_rnns/dir_tmaze_er_rmsprop_10_gaiarnn/")

# ╔═╡ a1bde34d-7216-4dc0-a08e-eae8a03148a6
ic_gaugru, dd_gaugru = RPU.load_data("../../local_data/gated_rnns/dir_tmaze_er_rmsprop_10_gaugru/")

# ╔═╡ d944303f-cdf4-413e-8932-a04507168eab
boxplot_data = RPU.get_line_data_for(
	ic,
	["cell", "numhidden", "factors", "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 9e8614a4-7e20-404b-a025-77ef51947dde
boxplot_data_gaigru = RPU.get_line_data_for(
	ic_gaigru,
	["numhidden" "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ a41a0df4-670a-474b-8453-2130d67137ba
boxplot_data_gaiagru = RPU.get_line_data_for(
	ic_gaiagru,
	["numhidden" "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 41d54ec8-aeac-40b2-a682-f9724baf4c64
boxplot_data_gaiarnn = RPU.get_line_data_for(
	ic_gaiarnn,
	["numhidden" "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 8c206951-8f90-4a68-be21-12100ba713c1
boxplot_data_gaugru = RPU.get_line_data_for(
	ic_gaugru,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ bb53e96d-77b8-44c0-9d81-2f221033aeb4
boxplot_data[2]

# ╔═╡ 864a5575-58dd-486a-a274-327a82d7c261
let	
	plt = plot()
	
	args_list = [
		Dict("cell"=>"ActionGatedRNN", "numhidden"=>10, "factors"=> 0, "internal"=>28),
		Dict("cell"=>"ActionGatedRNN", "numhidden"=>15, "factors"=> 0, "internal"=>21),
		Dict("cell"=>"ActionGatedRNN", "numhidden"=>20, "factors"=> 0, "internal"=>17),
		]
	names = ["GatedRNN 10", "GatedRNN 15", "GatedRNN 20"]
	for i in 1:length(args_list)
		plt = violin!(boxplot_data, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
	plt = boxplot!(boxplot_data, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=8, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="ER Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		#plt = dotplot!(boxplot_data, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	plt

	#savefig("../../data/paper_plots/factored_tensor/dir_tmaze_box_plots_300k_steps_tau_16_fac_tensor.pdf")
end

# ╔═╡ 9b18dc96-50d0-4668-ad99-320502902ec4
let	
	plt = plot()
	
	args_list = [
		Dict("cell"=>"GRU", "numhidden"=>17, "factors"=> 0, "internal"=>0),
		Dict("cell"=>"AAGRU", "numhidden"=>17, "factors"=> 0, "internal"=>0),
		Dict("cell"=>"MAGRU", "numhidden"=>10, "factors"=> 0, "internal"=>0),
		Dict("cell"=>"FacMAGRU", "numhidden"=>15, "factors"=> 17, "internal"=>0),
		Dict("cell"=>"RNN", "numhidden"=>30, "factors"=> 0, "internal"=>0),
		Dict("cell"=>"AARNN", "numhidden"=>30, "factors"=> 0, "internal"=>0),
		Dict("cell"=>"MARNN", "numhidden"=>18, "factors"=> 0, "internal"=>0),
		Dict("cell"=>"FacMARNN", "numhidden"=>25, "factors"=> 15, "internal"=>0),
		Dict("cell"=>"ActionGatedRNN", "numhidden"=>10, "factors"=> 0, "internal"=>28),
		]
	names = ["GRU", "AAGRU", "MAGRU", "FacGRU", "RNN", "AARNN", "MARNN", "FacRNN",  "GatedRNN 10"]
	for i in 1:length(args_list)
		if i == 5
			plt = vline!([6], linestyle=:dot, color=:white, lw=2)
		end
		plt = violin!(boxplot_data, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
	plt = boxplot!(boxplot_data, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=7, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="ER Directional TMaze, τ: 12", ylim=(0.3, 1))
		
		#plt = dotplot!(boxplot_data, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	args_list = [
		Dict()
		]
	names = ["20"]
	for i in 1:length(args_list)
		plt = violin!(boxplot_data_gaugru, args_list[i], label=names[i], legend=false, color=cell_colors["ActionGatedRNN"], lw=2, linecolor=cell_colors["ActionGatedRNN"])
		
	plt = boxplot!(boxplot_data_gaugru, args_list[i], label=names[i], color=color=cell_colors["ActionGatedRNN"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=8, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="ER Directional TMaze, τ: 12", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data_gaugru, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	plt

	#savefig("../../data/paper_plots/factored_tensor/dir_tmaze_box_plots_300k_steps_tau_16_fac_tensor.pdf")
end

# ╔═╡ 74be388f-528d-4f56-bcde-a86dc2bdb756
let	
	plt = plot()
	
	args_list = [
		Dict("numhidden"=>6, "internal"=>33),
		Dict("numhidden"=>8, "internal"=>25),
		Dict("numhidden"=>10, "internal"=>17),
		Dict("numhidden"=>13, "internal"=>9),
		Dict("numhidden"=>10, "internal"=>30),
		Dict("numhidden"=>20, "internal"=>40),
		]
	names = ["33", "25", "17", "9", "30", "40"]
	for i in 1:length(args_list)
		plt = violin!(boxplot_data_gaigru, args_list[i], label=names[i], legend=false, color=cell_colors["ActionGatedRNN"], lw=2, linecolor=cell_colors["ActionGatedRNN"])
		
	plt = boxplot!(boxplot_data_gaigru, args_list[i], label=names[i], color=color=cell_colors["ActionGatedRNN"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=8, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="ER Directional TMaze, τ: 12", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data_gaigru, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	plt

	#savefig("../../data/paper_plots/factored_tensor/dir_tmaze_box_plots_300k_steps_tau_16_fac_tensor.pdf")
end

# ╔═╡ 4d73f8cb-3848-4848-9894-a9c46ed44594
let	
	plt = plot()
	
	args_list = [
		Dict("numhidden"=>6, "internal"=>52),
		Dict("numhidden"=>8, "internal"=>38),
		Dict("numhidden"=>10, "internal"=>27),
		Dict("numhidden"=>13, "internal"=>14),
		Dict("numhidden"=>15, "internal"=>7),
		Dict("numhidden"=>20, "internal"=>50),
		]
	names = ["52", "38", "27", "14", "7", "50"]
	for i in 1:length(args_list)
		plt = violin!(boxplot_data_gaiagru, args_list[i], label=names[i], legend=false, color=cell_colors["ActionGatedRNN"], lw=2, linecolor=cell_colors["ActionGatedRNN"])
		
	plt = boxplot!(boxplot_data_gaiagru, args_list[i], label=names[i], color=color=cell_colors["ActionGatedRNN"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=8, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="ER Directional TMaze, τ: 12", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data_gaiagru, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	plt

	#savefig("../../data/paper_plots/factored_tensor/dir_tmaze_box_plots_300k_steps_tau_16_fac_tensor.pdf")
end

# ╔═╡ be41bdf6-6c72-4b0a-97b0-0926c3020d22
let	
	plt = plot()
	
	args_list = [
		Dict("numhidden"=>6, "internal"=>65),
		Dict("numhidden"=>13, "internal"=>35),
		Dict("numhidden"=>18, "internal"=>28),
		Dict("numhidden"=>25, "internal"=>20),
		Dict("numhidden"=>30, "internal"=>100),
		]
	names = ["65", "35", "28", "20", "100"]
	for i in 1:length(args_list)
		plt = violin!(boxplot_data_gaiarnn, args_list[i], label=names[i], legend=false, color=cell_colors["ActionGatedRNN"], lw=2, linecolor=cell_colors["ActionGatedRNN"])
		
	plt = boxplot!(boxplot_data_gaiarnn, args_list[i], label=names[i], color=color=cell_colors["ActionGatedRNN"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=8, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="ER Directional TMaze, τ: 12", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data_gaiarnn, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	plt

	#savefig("../../data/paper_plots/factored_tensor/dir_tmaze_box_plots_300k_steps_tau_16_fac_tensor.pdf")
end

# ╔═╡ 168a727e-bce1-43d5-8c32-215fd370acb4
let	
	plt = plot()
	
	args_list = [
		Dict()
		]
	names = ["20"]
	for i in 1:length(args_list)
		plt = violin!(boxplot_data_gaugru, args_list[i], label=names[i], legend=false, color=cell_colors["ActionGatedRNN"], lw=2, linecolor=cell_colors["ActionGatedRNN"])
		
	plt = boxplot!(boxplot_data_gaugru, args_list[i], label=names[i], color=color=cell_colors["ActionGatedRNN"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=8, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="ER Directional TMaze, τ: 12", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data_gaugru, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	plt

	#savefig("../../data/paper_plots/factored_tensor/dir_tmaze_box_plots_300k_steps_tau_16_fac_tensor.pdf")
end

# ╔═╡ Cell order:
# ╠═0f11b684-a3f9-4d40-9c63-33b28136d4f5
# ╠═95498906-eea1-11eb-26f4-8574b531e0dc
# ╠═78632921-6931-4ca0-9e8a-095a7f2c99b0
# ╠═8b4861e4-cfbc-4a0e-8246-7d6182c83586
# ╠═edd86b13-bdb5-4811-8e0a-36230d1162ae
# ╠═1c2a5244-0999-44f2-a952-3f495a8ec136
# ╠═f197d16b-eb04-408c-a020-9023f2eb02b2
# ╠═a67aba27-40ef-4ba2-816a-6da61a1d523b
# ╠═f9c41407-995c-4ca5-b5d4-46bce5f9585a
# ╠═3a1b767e-d09b-4079-b2a5-7e2e189a1600
# ╠═606b1f70-976d-4adc-9d1b-a34835b947e4
# ╠═a1bde34d-7216-4dc0-a08e-eae8a03148a6
# ╠═d944303f-cdf4-413e-8932-a04507168eab
# ╠═9e8614a4-7e20-404b-a025-77ef51947dde
# ╠═a41a0df4-670a-474b-8453-2130d67137ba
# ╠═41d54ec8-aeac-40b2-a682-f9724baf4c64
# ╠═8c206951-8f90-4a68-be21-12100ba713c1
# ╠═bb53e96d-77b8-44c0-9d81-2f221033aeb4
# ╟─864a5575-58dd-486a-a274-327a82d7c261
# ╠═9b18dc96-50d0-4668-ad99-320502902ec4
# ╠═74be388f-528d-4f56-bcde-a86dc2bdb756
# ╠═4d73f8cb-3848-4848-9894-a9c46ed44594
# ╠═be41bdf6-6c72-4b0a-97b0-0926c3020d22
# ╠═168a727e-bce1-43d5-8c32-215fd370acb4
