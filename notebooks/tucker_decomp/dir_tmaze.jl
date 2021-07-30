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

# ╔═╡ bb060799-9106-4690-8726-6e4e110f965a
ic_er_ignore, dd_er_ignore = RPU.load_data("../../local_data/tucker_decomp/dir_tmaze_er_rmsprop_10_fac_tuc_ignore/")

# ╔═╡ 5ebad194-332f-473b-a8d4-9f367be0edd9
boxplot_data = RPU.get_line_data_for(
	ic_er,
	["numhidden", "out_factors", "in_factors", "action_factors", "factors", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ d97941d9-6e99-4655-b975-626e700c8f4e
boxplot_data_ignore = RPU.get_line_data_for(
	ic_er_ignore,
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
		
		plt = boxplot!(boxplot_data, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=10, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="ER Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		#plt = dotplot!(boxplot_data, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end	
	
	plt
	
	savefig("../../data/paper_plots/tucker_decomp/dir_tmaze_er_rnn_tuc.pdf")
	
end

# ╔═╡ da61bff0-1948-4c84-ba92-093af9644097
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
		plt = violin!(boxplot_data_ignore, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
		plt = boxplot!(boxplot_data_ignore, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=10, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="ER Directional TMaze, τ: 16, Ignore", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data_ignore, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
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
		
		plt = boxplot!(boxplot_data, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=10, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="ER Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		#plt = dotplot!(boxplot_data, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end	
	
	plt
	
	savefig("../../data/paper_plots/tucker_decomp/dir_tmaze_er_gru_tuc.pdf")
	
end

# ╔═╡ 814640ed-7b02-4034-ba2e-71e7a3aa69cc
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
		plt = violin!(boxplot_data_ignore, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
		plt = boxplot!(boxplot_data_ignore, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=10, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="ER Directional TMaze, τ: 16, Ignore", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data_ignore, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end	
	
	plt
	
end

# ╔═╡ 04a0149b-91ee-4819-b558-e5b3c93af276
ic_on, dd_on = RPU.load_data("../../local_data/tucker_decomp/dir_tmaze_online_rmsprop_10_fac_tuc/")

# ╔═╡ 85afda47-1cc6-4c85-b008-9033a9923710
ic_on_f, dd_on_f = RPU.load_data("../../local_data/factored_tensor/final_dir_tmaze_online_rmsprop_10_fac_tensor/")

# ╔═╡ 9d1104a4-c277-45c8-bda0-40515ca3be42
ic_bp, dd_bp = let
	ic = ItemCollection("../../local_data/final_dir_tmaze_online_rmsprop_10_t16/")
	subic = search(ic) do ld
		ld.parsed_args["cell"][1:3] !== "Fac"
	end
	subic, diff(subic)
end

# ╔═╡ 949d0d29-9a6a-482a-b008-7b14af91d09c
function get_300k(data)
  ns = 0
  idx = 1
  while ns < 300_000
      ns += data["results"][:total_steps][idx]
      idx += 1
  end
  data["results"][:successes][1:idx]
end

# ╔═╡ 9edd9db8-16cd-4943-8be3-2648632d343c
function get_MUE(data, perc)
    mean(data[end-max(1, Int(floor(length(data)*perc))):end])
end

# ╔═╡ 862a4fed-8c77-4e17-924b-53b0e19c6a2c
boxplot_data_on = RPU.get_line_data_for(
	ic_on,
	["numhidden", "out_factors", "in_factors", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ e6c5fa5f-ebaa-43a8-86d5-f03c8a29a6bb
data_bp = RPU.get_line_data_for(
	ic_bp,
	["cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->get_MUE(get_300k(x), 0.1),
	get_data=(x)->get_MUE(get_300k(x), 0.1))

# ╔═╡ c11e13a0-98c8-45c6-9a9a-159a649dd7ef
boxplot_data_f = RPU.get_line_data_for(
	ic_on_f,
	["numhidden", "factors", "cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 994c5e63-b01b-4059-be67-b5afd5c8c91c
let	
	plt = plot()
	
	args_list_1 = [
		Dict("cell"=>"MARNN"),
	]
	names = ["MARNN"]

	for i in 1:length(args_list_1)
		plt = violin!(data_bp, args_list_1[i], label=names[i], legend=false, color=cell_colors[args_list_1[i]["cell"]], lw=2, linecolor=cell_colors[args_list_1[i]["cell"]])
		
		plt = boxplot!(data_bp, args_list_1[i], label=names[i], color=color=cell_colors[args_list_1[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))	
		
		
		#plt = dotplot!(data_bp, args_list_1[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(0.4, 1.0))
		
	end
	
	args_list = [
		Dict("cell"=>"FacMARNN", "numhidden"=>46, "factors"=>24)	
		]
	names = ["FacRNN"]
	for i in 1:length(names)
		plt = violin!(boxplot_data_f, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
		plt = boxplot!(boxplot_data_f, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=10, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		#plt = dotplot!(boxplot_data_f, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=10, ylims=(0.4, 1.0))
		
	end
	
	args_list = [
		Dict("cell"=>"FacTucMARNN", "numhidden"=>46, "out_factors"=>16, "in_factors"=>16),
		Dict("cell"=>"FacTucMARNN", "numhidden"=>36, "out_factors"=>18, "in_factors"=>18),
		Dict("cell"=>"FacTucMARNN", "numhidden"=>27, "out_factors"=>20, "in_factors"=>20),
		]
	names = ["FacTucRNN 16", "FacTucRNN 18", "FacTucRNN 20"]
	for i in 1:length(names)
		plt = violin!(boxplot_data_on, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
		plt = boxplot!(boxplot_data_on, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=10, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		#plt = dotplot!(boxplot_data_on, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=10, ylims=(0.4, 1.0))
		
	end
	
	plt
	
	savefig("../../data/paper_plots/tucker_decomp/dir_tmaze_online_rnn_tuc.pdf")
end

# ╔═╡ eb539aba-8c2d-4737-b51f-47bc6683f66a
let	
	plt = plot()
	
	args_list_1 = [
		Dict("cell"=>"MAGRU"),
	]
	names = ["MAGRU"]

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
	
	args_list = [
		Dict("cell"=>"FacTucMAGRU", "numhidden"=>26, "out_factors"=>15, "in_factors"=>15),
		Dict("cell"=>"FacTucMAGRU", "numhidden"=>20, "out_factors"=>17, "in_factors"=>17),
		Dict("cell"=>"FacTucMAGRU", "numhidden"=>15, "out_factors"=>20, "in_factors"=>20),
		]
	names = ["FacTucGRU 15", "FacTucGRU 17", "FacTucGRU 20"]
	for i in 1:length(names)
		plt = violin!(boxplot_data_on, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
		plt = boxplot!(boxplot_data_on, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=10, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		#plt = dotplot!(boxplot_data_on, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=10, ylims=(0.4, 1.0))
		
	end
	
	plt
	
	savefig("../../data/paper_plots/tucker_decomp/dir_tmaze_online_gru_tuc.pdf")
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
# ╠═bb060799-9106-4690-8726-6e4e110f965a
# ╠═5ebad194-332f-473b-a8d4-9f367be0edd9
# ╠═d97941d9-6e99-4655-b975-626e700c8f4e
# ╠═fce4a562-081c-42db-9de7-567b5d203647
# ╟─da61bff0-1948-4c84-ba92-093af9644097
# ╠═0942aad9-a966-46a4-9db6-742eece97a24
# ╟─814640ed-7b02-4034-ba2e-71e7a3aa69cc
# ╠═04a0149b-91ee-4819-b558-e5b3c93af276
# ╠═85afda47-1cc6-4c85-b008-9033a9923710
# ╠═9d1104a4-c277-45c8-bda0-40515ca3be42
# ╠═949d0d29-9a6a-482a-b008-7b14af91d09c
# ╠═9edd9db8-16cd-4943-8be3-2648632d343c
# ╠═862a4fed-8c77-4e17-924b-53b0e19c6a2c
# ╠═e6c5fa5f-ebaa-43a8-86d5-f03c8a29a6bb
# ╠═c11e13a0-98c8-45c6-9a9a-159a649dd7ef
# ╠═994c5e63-b01b-4059-be67-b5afd5c8c91c
# ╠═eb539aba-8c2d-4737-b51f-47bc6683f66a
