### A Pluto.jl notebook ###
# v0.19.3

using Markdown
using InteractiveUtils

# ╔═╡ b8878cfc-6c16-449a-ab1f-21d03d124d09
let
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ dc7d334c-b10f-11eb-3642-795b49bfbb85
using Revise

# ╔═╡ 965ac7dc-6137-4d10-83cd-60064c042524
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI

# ╔═╡ 0c2a721e-7c63-4128-a1ca-fe22385fdf93
const RPU = ReproducePlotUtils

# ╔═╡ 7d848d29-b972-4620-91f5-3dcc105dba9e
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

# ╔═╡ 3bd68e5b-69ed-47a0-bdbf-8d2a8a134ba0
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ 9ea3293d-fe67-4058-81a9-17ffb4b28fdc
at(dir) = joinpath("../../local_data/tmaze/", dir)

# ╔═╡ 03a62bed-f474-4d55-b042-92daaabdaaed
ic_tmaze_10, dd_tmaze_10 = RPU.load_data(at("tmaze_er_rnn_rmsprop_10/"))

# ╔═╡ 03e9ffe6-ddff-462d-b6a3-6cc1eb2d4f5d
data_tmaze_10_sens = RPU.get_line_data_for(
	ic_tmaze_10,
	["numhidden", "truncation", "cell", "eta"],
	[];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 52377f59-b234-4aa7-807c-914246c71a3c
let
	plot(data_tmaze_10_sens, 
	 	 Dict("numhidden"=>15, "truncation"=>10, "cell"=>"GRU"); 
	 	 sort_idx="eta", 
		 z=1.97, lw=2, 
		 label="GRU", color=cell_colors["GRU"])
    plot!(data_tmaze_10_sens, 
	 	  Dict("numhidden"=>15, "truncation"=>10, "cell"=>"AAGRU"); 
	 	  sort_idx="eta", z=1.97, lw=2, palette=RPU.custom_colorant,
		  label="AAGRU", color=cell_colors["AAGRU"])
	plot!(data_tmaze_10_sens, 
	 	  Dict("numhidden"=>10, "truncation"=>10, "cell"=>"MAGRU"); 
		  sort_idx="eta", 
		  z=1.97, 
		  lw=2, 
		  palette=RPU.custom_colorant, 
		  xaxis=:log,
		  label="MAGRU", 
		  legend=:bottomright, color=cell_colors["MAGRU"])
end

# ╔═╡ 4d3ec9dd-f7f8-4e70-b577-4180bd19e17d
let
	plot(data_tmaze_10_sens, 
	 	 Dict("numhidden"=>20, "truncation"=>10, "cell"=>"RNN"); 
	 	 sort_idx="eta", 
		 z=1.97, lw=2, 
		 palette=RPU.custom_colorant, label="RNN",
		 color=cell_colors["RNN"])
    plot!(data_tmaze_10_sens, 
	 	  Dict("numhidden"=>20, "truncation"=>10, "cell"=>"AARNN"); 
	 	  sort_idx="eta", z=1.97, lw=2, palette=RPU.custom_colorant,
		  label="AARNN",
	color=cell_colors["AARNN"])
	plot!(data_tmaze_10_sens, 
	 	  Dict("numhidden"=>15, "truncation"=>10, "cell"=>"MARNN"); 
		  sort_idx="eta", 
		  z=1.97, 
		  lw=2, 
		  palette=RPU.custom_colorant, 
		  xaxis=:log,
		  label="MARNN", 
		  legend=:bottomright,
	color=cell_colors["MARNN"])
end

# ╔═╡ 7d3c5f42-cc09-4c5b-a63a-d41e8e7dd1d4
data_tmaze_10 = RPU.get_line_data_for(
	ic_tmaze_10,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ a7a9a2e2-2afb-4e7a-a198-9b1e21877cc4
let
	τ = 10
	
	plt = plot(grid=false, )
	
	boxplot!(data_tmaze_10, Dict("numhidden"=>6, "truncation"=>τ, "cell"=>"GRU"), label_idx="cell", legend=nothing, color=cell_colors["GRU"])
	boxplot!(data_tmaze_10, Dict("numhidden"=>6, "truncation"=>τ, "cell"=>"AAGRU"), label_idx="cell", color=cell_colors["AAGRU"])
	boxplot!(data_tmaze_10, Dict("numhidden"=>6, "truncation"=>τ, "cell"=>"MAGRU"), label_idx="cell", color=cell_colors["MAGRU"])
	boxplot!(data_tmaze_10, Dict("numhidden"=>20, "truncation"=>τ, "cell"=>"RNN"), label_idx="cell", color=cell_colors["RNN"])
	boxplot!(data_tmaze_10, Dict("numhidden"=>20, "truncation"=>τ, "cell"=>"AARNN"), label_idx="cell", color=cell_colors["AARNN"])
	boxplot!(data_tmaze_10, Dict("numhidden"=>20, "truncation"=>τ, "cell"=>"MARNN"), label_idx="cell", color=cell_colors["MARNN"])
	
	dotplot!(data_tmaze_10, Dict("numhidden"=>6, "truncation"=>τ, "cell"=>"GRU"), label_idx="cell", legend=nothing, color=cell_colors["GRU"])
	dotplot!(data_tmaze_10, Dict("numhidden"=>6, "truncation"=>τ, "cell"=>"AAGRU"), label_idx="cell", color=cell_colors["AAGRU"])
	dotplot!(data_tmaze_10, Dict("numhidden"=>6, "truncation"=>τ, "cell"=>"MAGRU"), label_idx="cell", color=cell_colors["MAGRU"])
	dotplot!(data_tmaze_10, Dict("numhidden"=>20, "truncation"=>τ, "cell"=>"RNN"), label_idx="cell", color=cell_colors["RNN"])
	dotplot!(data_tmaze_10, Dict("numhidden"=>20, "truncation"=>τ, "cell"=>"AARNN"), label_idx="cell", color=cell_colors["AARNN"])
	dotplot!(data_tmaze_10, Dict("numhidden"=>20, "truncation"=>τ, "cell"=>"MARNN"), label_idx="cell", color=cell_colors["MARNN"])
	# savefig("../plots/tmaze_er.pdf")
	plt
end

# ╔═╡ 4f61294f-4abf-490d-a963-75201431198d
let
	args = Dict{String, Any}[]
	τ = 10
	for cell ∈ dd_tmaze_10["cell"]
		params = Dict("cell"=>cell, "numhidden"=>cell[end-2:end] == "GRU" ? 6 : 20, "truncation"=>τ)
		idx = findall(data_tmaze_10.data) do ld
			all([ld.line_params[k] == params[k] for k in keys(params)])
		end
		if length(idx) == 1
			params["eta"] = data_tmaze_10[idx][1].swept_params[1]
			push!(args, params)
		end
	end
	# FileIO.save("../final_runs/tmaze_10.jld2", "args", args)
	args
end

# ╔═╡ efd6ca8b-1eb6-4b56-8f01-c80ddfdd9033
final_ic_tmaze_10, final_dd_tmaze_10 = RPU.load_data(at("final_act_tmaze_er_rnn_rmsprop_10/"))

# ╔═╡ b5ce8032-b53f-47e2-9919-766321d77407
let
	sub_ic = search(final_ic_tmaze_10, Dict("cell"=>"MARNN"))
	sub_ic[1].parsed_args
end

# ╔═╡ a1bd4e67-4a69-48fe-9cbf-45a0fd79a648
data_final_tmaze_10 = RPU.get_line_data_for(
	final_ic_tmaze_10,
	["numhidden", "cell"],
	[];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ ee415dca-6671-47a8-9054-1c43b4892d4b
let

	plt = plot(grid=false, tickdir=:out, tickfontsize=12)
	
	boxplot!(data_final_tmaze_10, Dict("cell"=>"GRU"), label_idx="cell", legend=nothing, color=cell_colors["GRU"], outliers=false)
	boxplot!(data_final_tmaze_10, Dict("cell"=>"AAGRU"), label_idx="cell", color=cell_colors["AAGRU"], outliers=false)
	boxplot!(data_final_tmaze_10, Dict("cell"=>"MAGRU"), label_idx="cell", color=cell_colors["MAGRU"], outliers=false)
	boxplot!(data_final_tmaze_10, Dict("cell"=>"RNN"), label_idx="cell", color=cell_colors["RNN"], outliers=false)
	boxplot!(data_final_tmaze_10, Dict("cell"=>"AARNN"), label_idx="cell", color=cell_colors["AARNN"], outliers=false)
	boxplot!(data_final_tmaze_10, Dict("cell"=>"MARNN"), label_idx="cell", color=cell_colors["MARNN"], outliers=false)
	
	dotplot!(data_final_tmaze_10, Dict("cell"=>"GRU"), label_idx="cell", legend=nothing, color=cell_colors["GRU"], outliers=false)
	dotplot!(data_final_tmaze_10, Dict("cell"=>"AAGRU"), label_idx="cell", color=cell_colors["AAGRU"], outliers=false)
	dotplot!(data_final_tmaze_10, Dict("cell"=>"MAGRU"), label_idx="cell", color=cell_colors["MAGRU"], outliers=false)
	dotplot!(data_final_tmaze_10, Dict("cell"=>"RNN"), label_idx="cell", color=cell_colors["RNN"], outliers=false)
	dotplot!(data_final_tmaze_10, Dict("cell"=>"AARNN"), label_idx="cell", color=cell_colors["AARNN"], outliers=false)
	dotplot!(data_final_tmaze_10, Dict("cell"=>"MARNN"), label_idx="cell", color=cell_colors["MARNN"], outliers=false)
	
	# dotplot!(data_tmaze_10, Dict("numhidden"=>6, "truncation"=>τ, "cell"=>"GRU"), label_idx="cell", legend=nothing, color=cell_colors["GRU"])
	# dotplot!(data_tmaze_10, Dict("numhidden"=>6, "truncation"=>τ, "cell"=>"AAGRU"), label_idx="cell", color=cell_colors["AAGRU"])
	# dotplot!(data_tmaze_10, Dict("numhidden"=>6, "truncation"=>τ, "cell"=>"MAGRU"), label_idx="cell", color=cell_colors["MAGRU"])
	# dotplot!(data_tmaze_10, Dict("numhidden"=>20, "truncation"=>τ, "cell"=>"RNN"), label_idx="cell", color=cell_colors["RNN"])
	# dotplot!(data_tmaze_10, Dict("numhidden"=>20, "truncation"=>τ, "cell"=>"AARNN"), label_idx="cell", color=cell_colors["AARNN"])
	# dotplot!(data_tmaze_10, Dict("numhidden"=>20, "truncation"=>τ, "cell"=>"MARNN"), label_idx="cell", color=cell_colors["MARNN"])
	# savefig("../plots/final_tmaze_er.pdf")
	plt
end

# ╔═╡ 43574db2-c5fa-4f2c-89e1-b3f618d0145e
ic_tmaze_fac_10, dd_tmaze_fac_10 = RPU.load_data(at("tmaze_fac_er_rnn_init_rmsprop_10/"))

# ╔═╡ dec90ab5-c8fd-4492-9f01-5f1cf8d55b49
data_tmaze_fac_10 = RPU.get_line_data_for(
	ic_tmaze_fac_10,
	["numhidden", "cell", "init_style", "factors", "replay_size"],
	["eta"];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ aca33fc3-952a-4837-bfc5-63b42af357b2
let
	plt = plot(grid=false, tickdir=:out, tickfontsize=12)
	
	boxplot(data_tmaze_fac_10, Dict("numhidden"=>6, "init_style"=>"tensor", "replay_size"=>1000), label_idx="cell", legend=nothing, outliers=false, sort_idx="cell")
	
end

# ╔═╡ 3fdd4480-8e40-40d6-b9cb-e7af98810f69
data_tmaze_fac_10[4]

# ╔═╡ 564984e2-b447-4fc8-af99-7476b60ce1dd
let
	plt = plot(legend=false, grid=false, tickfontsize=11, tickdir=:out, ylims=(0.45, 1.0))
	for cell ∈ ["GRU", "AAGRU", "MAGRU"]
		violin!(data_final_tmaze_10, Dict("cell"=>cell), label_idx="cell", legend=false, color=cell_colors[cell], lw=1, linecolor=cell_colors[cell])
		boxplot!(data_final_tmaze_10, Dict("cell"=>cell), label_idx="cell", color=cell_colors[cell], fillalpha=0.75, outliers=true, lw=2, linecolor=:black)
		# dotplot!(data_final_tmaze_10, Dict("cell"=>cell), label_idx="cell", color=:black)
	end
	
	violin!(data_tmaze_fac_10, Dict("numhidden"=>6, "cell"=>"FacMAGRU", "init_style"=>"tensor", "replay_size"=>1000), color=cell_colors["FacMAGRU"], label="FacGRU", lw=1, linecolor=cell_colors["FacMAGRU"])
	boxplot!(data_tmaze_fac_10, Dict("numhidden"=>6, "cell"=>"FacMAGRU", "init_style"=>"tensor", "replay_size"=>1000), color=cell_colors["FacMAGRU"], label="FacGRU", fill=0.75, lw=2, linecolor=:black)
	
	plt = vline!([6], linestyle=:dot, color=:white, lw=2)
	
	for cell ∈ ["RNN", "AARNN", "MARNN"]
		violin!(data_final_tmaze_10, Dict("cell"=>cell), label_idx="cell", legend=false, color=cell_colors[cell], lw=1, linecolor=cell_colors[cell])
		boxplot!(data_final_tmaze_10, Dict("cell"=>cell), label_idx="cell", color=cell_colors[cell], fillalpha=0.75, outliers=true, lw=2, linecolor=:black)
	end
	
	violin!(data_tmaze_fac_10, Dict("numhidden"=>20, "cell"=>"FacMARNN", "init_style"=>"tensor", "replay_size"=>1000), color=cell_colors["FacMARNN"], label="FacRNN", lw=1, linecolor=cell_colors["FacMARNN"])
	boxplot!(data_tmaze_fac_10, Dict("numhidden"=>20, "cell"=>"FacMARNN", "init_style"=>"tensor", "replay_size"=>1000), color=cell_colors["FacMARNN"], label="FacRNN", fill=0.75, lw=2, linecolor=:black)
	
	# savefig("../plots/tmaze_er.pdf")
	plt
end

# ╔═╡ Cell order:
# ╠═b8878cfc-6c16-449a-ab1f-21d03d124d09
# ╠═dc7d334c-b10f-11eb-3642-795b49bfbb85
# ╠═965ac7dc-6137-4d10-83cd-60064c042524
# ╠═0c2a721e-7c63-4128-a1ca-fe22385fdf93
# ╠═7d848d29-b972-4620-91f5-3dcc105dba9e
# ╠═3bd68e5b-69ed-47a0-bdbf-8d2a8a134ba0
# ╠═9ea3293d-fe67-4058-81a9-17ffb4b28fdc
# ╠═03a62bed-f474-4d55-b042-92daaabdaaed
# ╠═03e9ffe6-ddff-462d-b6a3-6cc1eb2d4f5d
# ╠═52377f59-b234-4aa7-807c-914246c71a3c
# ╠═4d3ec9dd-f7f8-4e70-b577-4180bd19e17d
# ╠═7d3c5f42-cc09-4c5b-a63a-d41e8e7dd1d4
# ╠═a7a9a2e2-2afb-4e7a-a198-9b1e21877cc4
# ╠═4f61294f-4abf-490d-a963-75201431198d
# ╠═efd6ca8b-1eb6-4b56-8f01-c80ddfdd9033
# ╠═b5ce8032-b53f-47e2-9919-766321d77407
# ╠═a1bd4e67-4a69-48fe-9cbf-45a0fd79a648
# ╠═ee415dca-6671-47a8-9054-1c43b4892d4b
# ╠═43574db2-c5fa-4f2c-89e1-b3f618d0145e
# ╠═dec90ab5-c8fd-4492-9f01-5f1cf8d55b49
# ╠═aca33fc3-952a-4837-bfc5-63b42af357b2
# ╠═3fdd4480-8e40-40d6-b9cb-e7af98810f69
# ╠═564984e2-b447-4fc8-af99-7476b60ce1dd
