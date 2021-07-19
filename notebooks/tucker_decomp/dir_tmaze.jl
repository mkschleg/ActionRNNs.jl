### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 966bcd86-b9b3-4334-9775-4f833f55afc0
using Revise

# ╔═╡ ea398e7a-e4f1-11eb-35cb-37c72f021709
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 3e2820f6-9cb6-4ff8-a2d0-69e489ab8e40
using JLD2

# ╔═╡ 516a8add-7007-4d74-8ebf-196edbb0edea
const RPU = ReproducePlotUtils

# ╔═╡ 19b2a006-cfdd-4998-93b8-d2eb570b4aab
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

# ╔═╡ bd0556af-85ce-400b-9541-1ac04a27c4a6
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

# ╔═╡ ccd0ed2b-0678-4afa-8811-ec24d1d34f22
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ 4a245f08-5ce3-4c72-ac48-37b111916f8e
ic_er, dd_er = RPU.load_data("../../local_data/tucker_decomp/dir_tmaze_er_rmsprop_10_fac_tuc/")

# ╔═╡ 5ebad194-332f-473b-a8d4-9f367be0edd9
boxplot_data = RPU.get_line_data_for(
	ic_er,
	["numhidden", "out_factors", "in_factors", "action_factors", "factors", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ fce4a562-081c-42db-9de7-567b5d203647
let	
	plt = plot()
	
	args_list = [
		Dict("cell"=>"MARNN", "numhidden"=>18, "factors"=>0, "action_factors"=>0, "in_factors"=>0, "out_factors"=>0),
		Dict("cell"=>"FacMARNN", "numhidden"=>25, "factors"=>15, "action_factors"=>0, "in_factors"=>0, "out_factors"=>0),
		Dict("cell"=>"FacTucMARNN", "numhidden"=>30, "factors"=>0, "action_factors"=>3, "in_factors"=>11, "out_factors"=>11),
		Dict("cell"=>"FacTucMARNN", "numhidden"=>25, "factors"=>0, "action_factors"=>3, "in_factors"=>12, "out_factors"=>12),
		Dict("cell"=>"FacTucMARNN", "numhidden"=>18, "factors"=>0, "action_factors"=>3, "in_factors"=>14, "out_factors"=>14),
		]
	names = ["MARNN", "FacRNN", "FacTucRNN 11", "FacTucRNN 12", "FacTucRNN 14"]
	for i in 1:length(args_list)
		plt = violin!(boxplot_data, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
		plt = boxplot!(boxplot_data, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=10, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end	
	
	plt
	
end

# ╔═╡ 0942aad9-a966-46a4-9db6-742eece97a24
let	
	plt = plot()
	
	args_list = [
		Dict("cell"=>"MAGRU", "numhidden"=>10, "factors"=>0, "action_factors"=>0, "in_factors"=>0, "out_factors"=>0),
		Dict("cell"=>"FacMAGRU", "numhidden"=>15, "factors"=>17, "action_factors"=>0, "in_factors"=>0, "out_factors"=>0),
		Dict("cell"=>"FacTucMAGRU", "numhidden"=>17, "factors"=>0, "action_factors"=>3, "in_factors"=>10, "out_factors"=>10),
		Dict("cell"=>"FacTucMAGRU", "numhidden"=>15, "factors"=>0, "action_factors"=>3, "in_factors"=>11, "out_factors"=>11),
		Dict("cell"=>"FacTucMAGRU", "numhidden"=>10, "factors"=>0, "action_factors"=>3, "in_factors"=>14, "out_factors"=>14),
		]
	names = ["MAGRU", "FacGRU", "FacTucGRU 10", "FacTucGRU 11", "FacTucGRU 14"]
	for i in 1:length(args_list)
		plt = violin!(boxplot_data, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
		plt = boxplot!(boxplot_data, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=10, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end	
	
	plt
	
end

# ╔═╡ Cell order:
# ╠═966bcd86-b9b3-4334-9775-4f833f55afc0
# ╠═ea398e7a-e4f1-11eb-35cb-37c72f021709
# ╠═3e2820f6-9cb6-4ff8-a2d0-69e489ab8e40
# ╠═516a8add-7007-4d74-8ebf-196edbb0edea
# ╟─19b2a006-cfdd-4998-93b8-d2eb570b4aab
# ╠═bd0556af-85ce-400b-9541-1ac04a27c4a6
# ╠═ccd0ed2b-0678-4afa-8811-ec24d1d34f22
# ╠═4a245f08-5ce3-4c72-ac48-37b111916f8e
# ╠═5ebad194-332f-473b-a8d4-9f367be0edd9
# ╠═fce4a562-081c-42db-9de7-567b5d203647
# ╠═0942aad9-a966-46a4-9db6-742eece97a24
