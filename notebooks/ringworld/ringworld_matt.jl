### A Pluto.jl notebook ###
# v0.19.3

using Markdown
using InteractiveUtils

# ╔═╡ 5c23cb06-a9c3-4972-ab4d-87983ad2feb4
let
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ 9c5f9424-b819-11eb-34d0-e38ee863d534
using Revise

# ╔═╡ 02dad445-5a03-4665-937f-edc76fcd52bb
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ c21423c4-8692-4e58-810e-8b2fbccf37c3
const RPU = ReproducePlotUtils

# ╔═╡ 7533d445-13c7-48f8-a475-4ddcc18095e8
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

# ╔═╡ 3bcf048f-479f-45f1-b8da-530a8a1a7032
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[1],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[end-1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[end],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ d8d42860-423a-4191-b519-9f6a7bd773db
ic, diff_dict = RPU.load_data("../../local_data/ringworld/final_ringworld_er_rmsprop_10/")

# ╔═╡ 8c5d7890-5a7a-4a07-b599-def065f4e939
data = RPU.get_line_data_for(
	ic,
	["numhidden", "truncation", "cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->x["results"]["end"],
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 1))

# ╔═╡ 19c7744d-6e8a-4c1b-b610-afcb3e643e02
let
	plot(data, Dict("numhidden"=>15, "truncation"=>6, "cell"=>"RNN"), z=1.96, color=cell_colors["RNN"], ylims=(0.0, 0.35))
	plot!(data, Dict("numhidden"=>9, "truncation"=>6, "cell"=>"MARNN"), z=1.96, color=cell_colors["MARNN"])
	plot!(data, Dict("numhidden"=>15, "truncation"=>6, "cell"=>"AARNN"), z=1.96, color=cell_colors["AARNN"])
end

# ╔═╡ a2cff277-e09e-40f6-95f5-d55d5f115eb4
let
	params = Dict("numhidden"=>9, "truncation"=>6, "cell"=>"MARNN")
	idx = findfirst(data.data) do ld
		all([ld.line_params[idx] == params[idx] for idx in keys(params)])
	end
	plot(data[idx].data, legend=false, ylims=(0.0, 0.4), color=cell_colors[params["cell"]])
	
	params = Dict("numhidden"=>9, "truncation"=>6, "cell"=>"AARNN")
	idx = findfirst(data.data) do ld
		all([ld.line_params[idx] == params[idx] for idx in keys(params)])
	end
	plot!(data[idx].data, legend=false, ylims=(0.0, 0.4), color=cell_colors[params["cell"]])
	
	params = Dict("numhidden"=>15, "truncation"=>6, "cell"=>"RNN")
	idx = findfirst(data.data) do ld
		all([ld.line_params[idx] == params[idx] for idx in keys(params)])
	end
	plot!(data[idx].data, legend=false, ylims=(0.0, 0.4), color=cell_colors[params["cell"]])
end

# ╔═╡ Cell order:
# ╠═5c23cb06-a9c3-4972-ab4d-87983ad2feb4
# ╠═9c5f9424-b819-11eb-34d0-e38ee863d534
# ╠═02dad445-5a03-4665-937f-edc76fcd52bb
# ╠═c21423c4-8692-4e58-810e-8b2fbccf37c3
# ╠═7533d445-13c7-48f8-a475-4ddcc18095e8
# ╠═3bcf048f-479f-45f1-b8da-530a8a1a7032
# ╠═d8d42860-423a-4191-b519-9f6a7bd773db
# ╠═8c5d7890-5a7a-4a07-b599-def065f4e939
# ╠═19c7744d-6e8a-4c1b-b610-afcb3e643e02
# ╠═a2cff277-e09e-40f6-95f5-d55d5f115eb4
