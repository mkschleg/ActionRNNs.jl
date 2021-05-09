### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ dc7d334c-b10f-11eb-3642-795b49bfbb85
using Revise

# ╔═╡ 965ac7dc-6137-4d10-83cd-60064c042524
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

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
	colorant"#88CCEE",
]

# ╔═╡ 3bd68e5b-69ed-47a0-bdbf-8d2a8a134ba0
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[1],
	"MARNN" => color_scheme[5],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[end],
	"MAGRU" => color_scheme[6])

# ╔═╡ 03a62bed-f474-4d55-b042-92daaabdaaed
ic_tmaze_10, dd_tmaze_10 = RPU.load_data("../local_data/tmaze_er_rnn_rmsprop_10/")

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
	 	 Dict("numhidden"=>15, "truncation"=>12, "cell"=>"GRU"); 
	 	 sort_idx="eta", 
		 z=1.97, lw=2, 
		 label="GRU", color=cell_colors["GRU"])
    plot!(data_tmaze_10_sens, 
	 	  Dict("numhidden"=>15, "truncation"=>12, "cell"=>"AAGRU"); 
	 	  sort_idx="eta", z=1.97, lw=2, palette=RPU.custom_colorant,
		  label="AAGRU", color=cell_colors["AAGRU"])
	plot!(data_tmaze_10_sens, 
	 	  Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MAGRU"); 
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
	 	 Dict("numhidden"=>20, "truncation"=>12, "cell"=>"RNN"); 
	 	 sort_idx="eta", 
		 z=1.97, lw=2, 
		 palette=RPU.custom_colorant, label="RNN",
		 color=cell_colors["RNN"])
    plot!(data_tmaze_10_sens, 
	 	  Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AARNN"); 
	 	  sort_idx="eta", z=1.97, lw=2, palette=RPU.custom_colorant,
		  label="AARNN",
	color=cell_colors["AARNN"])
	plot!(data_tmaze_10_sens, 
	 	  Dict("numhidden"=>15, "truncation"=>12, "cell"=>"MARNN"); 
		  sort_idx="eta", 
		  z=1.97, 
		  lw=2, 
		  palette=RPU.custom_colorant, 
		  xaxis=:log,
		  label="MARNN", 
		  legend=:bottomright,
	color=cell_colors["MARNN"])
end

# ╔═╡ Cell order:
# ╠═dc7d334c-b10f-11eb-3642-795b49bfbb85
# ╠═965ac7dc-6137-4d10-83cd-60064c042524
# ╠═0c2a721e-7c63-4128-a1ca-fe22385fdf93
# ╠═7d848d29-b972-4620-91f5-3dcc105dba9e
# ╠═3bd68e5b-69ed-47a0-bdbf-8d2a8a134ba0
# ╠═03a62bed-f474-4d55-b042-92daaabdaaed
# ╠═03e9ffe6-ddff-462d-b6a3-6cc1eb2d4f5d
# ╠═52377f59-b234-4aa7-807c-914246c71a3c
# ╠═4d3ec9dd-f7f8-4e70-b577-4180bd19e17d
