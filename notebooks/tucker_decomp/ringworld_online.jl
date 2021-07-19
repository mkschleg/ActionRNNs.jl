### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

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
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ a95b8a7d-9ddd-4e24-b6bb-7557016c033d
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ 2c7e6239-b59f-4e8c-9762-c041ad51547c
ic_on, dd_on = RPU.load_data("../../local_data/tucker_decomp/ringworld_online_rmsprop_10_tuc/")

# ╔═╡ Cell order:
# ╠═e3d13806-e4e9-11eb-1851-4b51c9ec551c
# ╠═b70f28c6-9b9c-4abd-9dec-f50dc0b51b90
# ╟─288917c9-24ca-41f9-854f-6ad3e805a7ad
# ╟─8698a6e7-3832-4455-8177-94ccf44b09cc
# ╠═a95b8a7d-9ddd-4e24-b6bb-7557016c033d
# ╠═2c7e6239-b59f-4e8c-9762-c041ad51547c
