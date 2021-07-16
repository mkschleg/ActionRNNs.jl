### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ ac0f1294-e4b2-11eb-3103-a748e8af9acf
using Revise

# ╔═╡ fc952828-e754-4d95-b251-2919b90ddbcb
using Reproduce, ReproducePlotUtils, Plots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 4e340a8b-5312-4b52-83d6-1621d3862a85
using JLD2

# ╔═╡ eb1b008d-bc07-4c41-99ef-998fa9b56405
const RPU = ReproducePlotUtils

# ╔═╡ e86822ca-400f-4e97-a0d6-1d03afb32321
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

# ╔═╡ d057ba32-9c48-41fe-a0bc-ff2b0ce30bb7
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ a9d6579d-034b-4a1d-9888-343aae7a7792
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ 40ca14d8-c448-40c8-891d-162023926df1
ic, dd = RPU.load_data("../../local_data/tucker_decomp/ringworld_er_rmsprop_10_fac_tuc/")

# ╔═╡ 8461d4fe-6193-4863-a9a2-16a5a551e150
data_on = RPU.get_line_data_for(
	ic,
	["numhidden", "truncation", "out_factors", "in_factors", "action_factors", "cell"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ d21814ba-686f-4736-be1a-d5cd83132651
md"""
Truncation: $(@bind truncation Select(string.(dd["truncation"])))
NumHidden: $(@bind numhidden Select(string.(dd["numhidden"])))
Out Factors: $(@bind out_factors Select(string.(dd["out_factors"])))
In Factors: $(@bind in_factors Select(string.(dd["in_factors"])))
Action Factors: $(@bind action_factors Select(string.(dd["action_factors"])))
Cell: $(@bind cells MultiSelect(dd["cell"]))
"""

# ╔═╡ b96eee17-d20e-4ea2-b8d2-b2a0b473dde8
let
	# plt = nothing
	τ = parse(Int, truncation)
	nh = parse(Int, numhidden)
	ofac = parse(Int, out_factors)
	ifac = parse(Int, in_factors)
	afac = parse(Int, action_factors)
	plt = plot()
	for cell ∈ cells
		plt = plot!(
			  data_on,
			  Dict("numhidden"=>nh, "truncation"=>τ, "out_factors"=>ofac, "in_factors"=>ifac, "action_factors"=>afac, "cell"=>cell),
			  palette=RPU.custom_colorant, legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.30), title="ER Ringworld", grid=false, tickdir=:out, legendtitle="Cell")
	end
	plt
end

# ╔═╡ Cell order:
# ╠═ac0f1294-e4b2-11eb-3103-a748e8af9acf
# ╠═fc952828-e754-4d95-b251-2919b90ddbcb
# ╠═4e340a8b-5312-4b52-83d6-1621d3862a85
# ╠═eb1b008d-bc07-4c41-99ef-998fa9b56405
# ╟─e86822ca-400f-4e97-a0d6-1d03afb32321
# ╟─d057ba32-9c48-41fe-a0bc-ff2b0ce30bb7
# ╠═a9d6579d-034b-4a1d-9888-343aae7a7792
# ╠═40ca14d8-c448-40c8-891d-162023926df1
# ╠═8461d4fe-6193-4863-a9a2-16a5a551e150
# ╟─d21814ba-686f-4736-be1a-d5cd83132651
# ╠═b96eee17-d20e-4ea2-b8d2-b2a0b473dde8
