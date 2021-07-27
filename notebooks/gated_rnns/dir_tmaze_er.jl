### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 69953223-ef7b-4f46-8a84-715c9f879d4b
using Revise

# ╔═╡ 95498906-eea1-11eb-26f4-8574b531e0dc
using Reproduce, ReproducePlotUtils, Plots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto, JLD2

# ╔═╡ 8b4861e4-cfbc-4a0e-8246-7d6182c83586
const RPU = ReproducePlotUtils

# ╔═╡ 4a327ec2-130a-4a4b-b051-98cb02ea692f
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

# ╔═╡ 152f5317-6d20-4804-8ea8-142f3dfe5e0f
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ f74830ea-1ead-46b0-8520-f99a6e0058f2
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ a67aba27-40ef-4ba2-816a-6da61a1d523b
ic, dd = RPU.load_data("../../local_data/gated_rnns/dir_tmaze_gated_er_rmsprop_10/")

# ╔═╡ d944303f-cdf4-413e-8932-a04507168eab


# ╔═╡ Cell order:
# ╠═69953223-ef7b-4f46-8a84-715c9f879d4b
# ╠═95498906-eea1-11eb-26f4-8574b531e0dc
# ╠═8b4861e4-cfbc-4a0e-8246-7d6182c83586
# ╟─4a327ec2-130a-4a4b-b051-98cb02ea692f
# ╟─152f5317-6d20-4804-8ea8-142f3dfe5e0f
# ╟─f74830ea-1ead-46b0-8520-f99a6e0058f2
# ╠═a67aba27-40ef-4ba2-816a-6da61a1d523b
# ╠═d944303f-cdf4-413e-8932-a04507168eab
