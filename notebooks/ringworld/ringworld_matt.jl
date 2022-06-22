### A Pluto.jl notebook ###
# v0.19.9

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
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"DAARNN" => color_scheme[7],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2], 
	"DAAGRU" => color_scheme[9],)

# ╔═╡ d8d42860-423a-4191-b519-9f6a7bd773db
ic, diff_dict = RPU.load_data("../../local_data/ringworld/er/final_ringworld_er_rmsprop_10/")

# ╔═╡ 8c5d7890-5a7a-4a07-b599-def065f4e939
data = RPU.get_line_data_for(
	ic,
	["numhidden", "truncation", "cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->x["results"]["end"],
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 1))

# ╔═╡ f0436522-6e08-4094-92a3-d96b518debe8
data[1]

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

# ╔═╡ cf190bf4-c3d3-47fe-99a9-7f141ab10937
let
	params = Dict("numhidden"=>9, "truncation"=>6, "cell"=>"MARNN")
	idx = findfirst(data.data) do ld
		all([ld.line_params[idx] == params[idx] for idx in keys(params)])
	end
	μ = mean(data[idx].data)
	σ = 1.97 * std(data[idx].data) ./ sqrt(length(data[idx].data))
	plot(μ, ribbon=σ, legend=false, ylims=(0.0, 0.4), color=cell_colors[params["cell"]])
	
	params = Dict("numhidden"=>9, "truncation"=>6, "cell"=>"AARNN")
	idx = findfirst(data.data) do ld
		all([ld.line_params[idx] == params[idx] for idx in keys(params)])
	end
	μ = mean(data[idx].data)
	σ = 1.97 * std(data[idx].data) ./ sqrt(length(data[idx].data))
	plot!(μ, ribbon=σ, legend=false, ylims=(0.0, 0.4), color=cell_colors[params["cell"]])
	
	params = Dict("numhidden"=>15, "truncation"=>6, "cell"=>"RNN")
	idx = findfirst(data.data) do ld
		all([ld.line_params[idx] == params[idx] for idx in keys(params)])
	end
	μ = mean(data[idx].data)
	σ = 1.97 * std(data[idx].data) ./ sqrt(length(data[idx].data))
	plot!(μ, ribbon=σ, legend=false, ylims=(0.0, 0.4), color=cell_colors[params["cell"]])
end

# ╔═╡ d0fd20c9-33e6-469d-986e-6dd0bb1f2e04
md"""# Online"""

# ╔═╡ 535a91d6-909a-4118-b375-0214e8f23402
	ic_online, diff_dict_online = RPU.load_data("../../local_data/ringworld/online/final_ringworld_online_rmsprop_10/")

# ╔═╡ 23bc366d-7c10-484d-bf50-01f06d6abe28
data_online = RPU.get_line_data_for(
	ic_online,
	["numhidden", "cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->x["results"]["end"],
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 1))

# ╔═╡ 4835983f-bdaa-46ca-b766-b0fd88711521
d = data_online[1]

# ╔═╡ 85645d25-ecf7-4796-a6ff-2ec8f3251f4a
data_online[findall((d)->d.line_params["cell"] == "FacMARNN", data_online.data)][2]

# ╔═╡ 0aa84bd4-fa49-42e9-adc5-89761956febe
let
	plt = plot(
		legend=nothing, 
		tickdir=:out, 
		grid=false, 
		# tickfontsize=18, 
		yticks=(0.0:0.05:0.4), 
		ylims=(0.0,0.4), 
		markersize=5)
	
	plot!(data_online, Dict("numhidden"=>20, "cell"=>"RNN"), z=1.96, color=cell_colors["RNN"])
	plot!(data_online, Dict("numhidden"=>15, "cell"=>"MARNN"), z=1.96, color=cell_colors["MARNN"])
	plot!(data_online, Dict("numhidden"=>20, "cell"=>"FacMARNN"), z=1.96, color=cell_colors["FacMARNN"])
	plot!(data_online, Dict("numhidden"=>20, "cell"=>"AARNN"), z=1.96, color=cell_colors["AARNN"])

	plot!(data_online, Dict("numhidden"=>12, "cell"=>"GRU"), z=1.96, color=cell_colors["GRU"])
	plot!(data_online, Dict("numhidden"=>9, "cell"=>"MAGRU"), z=1.96, color=cell_colors["MAGRU"])
	plot!(data_online, Dict("numhidden"=>12, "cell"=>"FacMAGRU"), z=1.96, color=cell_colors["FacMAGRU"])
	plot!(data_online, Dict("numhidden"=>12, "cell"=>"AAGRU"), z=1.96, color=cell_colors["AAGRU"])

	# savefig(plt, "../../plots/ringworld_online_lc.pdf")
	
	plt
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
# ╠═f0436522-6e08-4094-92a3-d96b518debe8
# ╠═19c7744d-6e8a-4c1b-b610-afcb3e643e02
# ╠═a2cff277-e09e-40f6-95f5-d55d5f115eb4
# ╠═cf190bf4-c3d3-47fe-99a9-7f141ab10937
# ╠═d0fd20c9-33e6-469d-986e-6dd0bb1f2e04
# ╠═535a91d6-909a-4118-b375-0214e8f23402
# ╠═23bc366d-7c10-484d-bf50-01f06d6abe28
# ╠═4835983f-bdaa-46ca-b766-b0fd88711521
# ╠═85645d25-ecf7-4796-a6ff-2ec8f3251f4a
# ╠═0aa84bd4-fa49-42e9-adc5-89761956febe
