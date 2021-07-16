### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ ea398e7a-e4f1-11eb-35cb-37c72f021709
using Revise, Reproduce, ReproducePlotUtils, Plots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto, JLD2

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
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ Cell order:
# ╠═ea398e7a-e4f1-11eb-35cb-37c72f021709
# ╠═516a8add-7007-4d74-8ebf-196edbb0edea
# ╟─19b2a006-cfdd-4998-93b8-d2eb570b4aab
# ╟─bd0556af-85ce-400b-9541-1ac04a27c4a6
