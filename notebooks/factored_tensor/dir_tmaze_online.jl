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
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ 2cb5a984-17e6-43c9-a73b-32fd074907d4
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ db2f42ba-7176-4fe0-8641-118ae35cbc3a
ic_on, dd_on = RPU.load_data("../../local_data/dir_tmaze_online_rmsprop_10_fac/")

# ╔═╡ a6eecff0-f022-4110-a9f4-5872710b93f5
ic_bp, dd_bp = let
	ic = ItemCollection("../../local_data/final_dir_tmaze_online_rmsprop_10_t16/")
	subic = search(ic) do ld
		ld.parsed_args["cell"][1:3] !== "Fac"
	end
	subic, diff(subic)
end

# ╔═╡ 512b947a-914d-4ecc-9fa3-737cc8887d1f
ic_bp_frnn, dd_bp_frnn = let
	ic = ItemCollection("../../local_data/final_dir_tmaze_online_rmsprop_10_t16/")
	subic = search(ic) do ld
		ld.parsed_args["cell"] == "FacMARNN"
	end
	subic, diff(subic)
end

# ╔═╡ 6f0e7744-155c-41f6-8d9e-9e8091c1ec75
ic_bp_fgru, dd_bp_fgru = let
	ic = ItemCollection("../../local_data/final_dir_tmaze_online_rmsprop_10_t16/")
	subic = search(ic) do ld
		ld.parsed_args["cell"] == "FacMAGRU"
	end
	subic, diff(subic)
end

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

# ╔═╡ 79fa1393-4634-4e87-b80d-6a780740b688
data_bp_frnn = RPU.get_line_data_for(
	ic_bp_frnn,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->get_MUE(get_300k(x), 0.1),
	get_data=(x)->get_MUE(get_300k(x), 0.1))

# ╔═╡ 098238ac-4d65-4678-b6c0-31a14d22b7be
data_bp_fgru = RPU.get_line_data_for(
	ic_bp_fgru,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->get_MUE(get_300k(x), 0.1),
	get_data=(x)->get_MUE(get_300k(x), 0.1))

# ╔═╡ d825d218-c73f-45b6-be11-e9ad852e9fd6
boxplot_data = RPU.get_line_data_for(
	ic_on,
	["numhidden", "factors", "cell", "init_style"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 7453c283-c299-46f9-a585-b09abf4716c4
params_gru = Dict("cell"=>"FacMAGRU", "numhidden"=>26, "factors"=>21, "init_style"=>"tensor")

# ╔═╡ 25ccbbae-21e5-46e2-954d-0b7880d49e67
params_rnn = Dict("cell"=>"FacMARNN", "numhidden"=>46, "factors"=>24, "init_style"=>"tensor")

# ╔═╡ 72d01967-a96c-4ff3-9e98-b99f6d9ad728
let
	idx = findall(boxplot_data.data) do ld
        all([params_gru[i] == ld.line_params[i] for i in keys(params_gru)])
    end
    data_gru = if length(idx) == 1
        boxplot_data[idx[1]]
    else
        boxplot_data[idx]
    end
end

# ╔═╡ 18d96878-36f3-469d-bab7-014828576111
let
	idx = findall(boxplot_data.data) do ld
        all([params_rnn[i] == ld.line_params[i] for i in keys(params_rnn)])
    end
    data_gru = if length(idx) == 1
        boxplot_data[idx[1]]
    else
        boxplot_data[idx]
    end
end

# ╔═╡ 8f425f3b-3437-4b4c-b368-4bb2796e87bb
let	
	
	plt = violin(data_bp, Dict("cell"=>"MARNN"), label="MARNN OG", legend=false, color=cell_colors["MARNN"], lw=2, linecolor=cell_colors["MARNN"])
		
	plt = boxplot!(data_bp, Dict("cell"=>"MARNN"), label="MARNN OG", color=color=cell_colors["MARNN"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
	plt = dotplot!(data_bp, Dict("cell"=>"MARNN"), label="MARNN OG", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(0.4, 1.0))
	
	
	plt = violin!(data_bp_frnn, Dict(), label="FacRNN OG", legend=false, color=cell_colors["FacMARNN"], lw=2, linecolor=cell_colors["FacMARNN"])
		
	plt = boxplot!(data_bp_frnn, Dict(), label="FacRNN OG", color=color=cell_colors["FacMARNN"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
	plt = dotplot!(data_bp_frnn, Dict(), label="FacRNN OG", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(0.4, 1.0))
		
	plt = vline!([3], linestyle=:dot, color=:white, lw=2)
	
	args_list = [
		Dict("cell"=>"MARNN", "numhidden"=>27, "factors"=>0, "init_style"=>"na"),
		Dict("cell"=>"FacMARNN", "numhidden"=>46, "factors"=>24, "init_style"=>"ignore"),
		Dict("cell"=>"FacMARNN", "numhidden"=>46, "factors"=>24, "init_style"=>"standard"),
		Dict("cell"=>"FacMARNN", "numhidden"=>46, "factors"=>24, "init_style"=>"tensor")	
		]
	names = ["MARNN", "FacRNN ignr", "FacRNN std", "FacRNN tnsr"]
	for i in 1:length(names)
		plt = violin!(boxplot_data, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
		plt = boxplot!(boxplot_data, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=10, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end	
	
	plt
	
end

# ╔═╡ e82f0cdc-f48e-4a39-974c-982252141da9
let	
	
	plt = violin(data_bp, Dict("cell"=>"MAGRU"), label="MAGRU OG", legend=false, color=cell_colors["MAGRU"], lw=2, linecolor=cell_colors["MAGRU"])
		
	plt = boxplot!(data_bp, Dict("cell"=>"MAGRU"), label="MAGRU OG", color=color=cell_colors["MAGRU"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
	plt = dotplot!(data_bp, Dict("cell"=>"MAGRU"), label="MAGRU OG", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(0.4, 1.0))
	
	
	plt = violin!(data_bp_fgru, Dict(), label="FacGRU OG", legend=false, color=cell_colors["FacMAGRU"], lw=2, linecolor=cell_colors["FacMAGRU"])
		
	plt = boxplot!(data_bp_fgru, Dict(), label="FacGRU OG", color=color=cell_colors["FacMAGRU"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
	plt = dotplot!(data_bp_fgru, Dict(), label="FacGRU OG", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(0.4, 1.0))
		
	plt = vline!([3], linestyle=:dot, color=:white, lw=2)
	
	args_list = [
		Dict("cell"=>"MAGRU", "numhidden"=>15, "factors"=>0, "init_style"=>"na"),
		Dict("cell"=>"FacMAGRU", "numhidden"=>26, "factors"=>21, "init_style"=>"standard"),
		Dict("cell"=>"FacMAGRU", "numhidden"=>26, "factors"=>21, "init_style"=>"tensor")	
		]
	names = ["MAGRU", "FacGRU std", "FacGRU tnsr"]
	for i in 1:length(names)
		plt = violin!(boxplot_data, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
		plt = boxplot!(boxplot_data, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=10, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=10, ylims=(0.4, 1.0))
		
	end	
	
	plt
	
end

# ╔═╡ 054ed0a7-df41-4857-ab30-66ce074c4cba
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
	]
	names = ["GRU", "AAGRU", "MAGRU", "RNN", "AARNN", "MARNN"]
	plt = plot()
	for i in 1:length(names)
		plt = violin!(data_bp, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
		plt = boxplot!(data_bp, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
	if i == 3	
		plt = vline!([6], linestyle=:dot, color=:white, lw=2)
	end
		
		plt = dotplot!(data_bp, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(0.4, 1.0))
		
	end
	plt
	
end

# ╔═╡ Cell order:
# ╠═18bfdc06-df2d-11eb-1ef4-0772caed819f
# ╠═60314f42-695e-42b9-8f5f-05bf0211de2a
# ╠═096bd6fc-5746-4b3b-8278-7e36ff1a5e4c
# ╠═c7df63a9-0ca4-4277-b6c3-271a471692e2
# ╟─3907310e-62c0-4e1e-9b40-180a1635efe7
# ╟─df7909ef-9f09-46c4-b4b1-c88a2db99dc7
# ╠═2cb5a984-17e6-43c9-a73b-32fd074907d4
# ╠═db2f42ba-7176-4fe0-8641-118ae35cbc3a
# ╠═a6eecff0-f022-4110-a9f4-5872710b93f5
# ╠═512b947a-914d-4ecc-9fa3-737cc8887d1f
# ╠═6f0e7744-155c-41f6-8d9e-9e8091c1ec75
# ╠═b7406f9b-618f-485c-b79d-920c83b3d6b3
# ╠═f3664996-185b-4a38-ba7e-2053b5837b61
# ╠═462b98cc-a53f-4430-b016-ad785eb9f3d4
# ╠═79fa1393-4634-4e87-b80d-6a780740b688
# ╠═098238ac-4d65-4678-b6c0-31a14d22b7be
# ╠═d825d218-c73f-45b6-be11-e9ad852e9fd6
# ╠═7453c283-c299-46f9-a585-b09abf4716c4
# ╠═25ccbbae-21e5-46e2-954d-0b7880d49e67
# ╠═72d01967-a96c-4ff3-9e98-b99f6d9ad728
# ╠═18d96878-36f3-469d-bab7-014828576111
# ╠═8f425f3b-3437-4b4c-b368-4bb2796e87bb
# ╠═e82f0cdc-f48e-4a39-974c-982252141da9
# ╟─054ed0a7-df41-4857-ab30-66ce074c4cba
