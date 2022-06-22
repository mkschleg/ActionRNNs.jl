### A Pluto.jl notebook ###
# v0.19.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 18ca04ca-a249-4c06-a992-7031e0da8353
let
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ 00c1df58-d947-11eb-36c4-653906e7fe45
using Revise

# ╔═╡ de72e089-68f2-4b16-9524-bfe6bf1dbe75
using Reproduce, ReproducePlotUtils, Plots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 83912db4-fa1e-4a7b-9767-390a91ba320f
using JLD2

# ╔═╡ a9c22a1e-6fa3-4d32-8892-f81dc53013ec
const RPU = ReproducePlotUtils

# ╔═╡ 1b8ca44f-41d3-40d4-b32b-9299e24af861
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

# ╔═╡ 8292a0f5-770d-43b4-ad4f-16752b7264cd
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ ee43a945-7f64-436b-b391-ad64edab4f5c
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ 5e534ba1-d33c-4824-abde-0e9a25eaca29
at(dir) = joinpath("../../local_data/ringworld/", dir)

# ╔═╡ dfc801ef-c384-4505-88bc-09914c138e0b
ic, dd = RPU.load_data(at("er/ringworld_er_rmsprop_10_fac/"))

# ╔═╡ 48fcb600-3c42-4bd6-aa7a-33663c2e5d25
ic_on, dd_on = RPU.load_data(at("online/ringworld_online_rmsprop_10_fac/"))

# ╔═╡ eb61ceb3-f6ac-4224-90d0-d2fcdf83b483
data = RPU.get_line_data_for(
	ic,
	["numhidden", "truncation", "factors", "cell", "init_style"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ f797fa36-df84-476d-be05-f409c3793a20
data_on = RPU.get_line_data_for(
	ic_on,
	["numhidden", "truncation", "factors", "cell", "init_style"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ 98a8531b-94c5-41f7-af88-fbfd62393f71
data_sens = RPU.get_line_data_for(
	ic,
	["numhidden", "truncation", "factors", "cell", "init_style"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_AUC(x, "end"))

# ╔═╡ 21766c7b-a193-4d79-bec2-b8332edeedd2
data_sens_on = RPU.get_line_data_for(
	ic_on,
	["numhidden", "truncation", "factors", "cell", "init_style"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_AUC(x, "end"))

# ╔═╡ e96f363b-f748-4664-a8c0-e35a79af2cab
md"""
Truncation: $(@bind truncation Select(string.(dd["truncation"])))
NumHidden: $(@bind numhidden Select(string.(dd["numhidden"])))
Factors: $(@bind factors Select(string.(dd["factors"])))
Init Style: $(@bind init_style Select(string.(dd["init_style"])))
Cell: $(@bind cells MultiSelect(dd["cell"]))
"""

# ╔═╡ 4109beac-1012-446d-83d3-052a2fd2044e
let
	# plt = nothing
	τ = parse(Int, truncation)
	nh = parse(Int, numhidden)
	fac = parse(Int, factors)
	plt = plot()
	for cell ∈ cells
		plt = plot!(
			  data_on,
			  Dict("numhidden"=>nh, "truncation"=>τ, "factors"=>fac, "init_style"=>init_style, "cell"=>cell),
			  palette=RPU.custom_colorant, legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.50), title="ER Ringworld", grid=false, tickdir=:out, legendtitle="Cell")
	end
	plt
end

# ╔═╡ 7043e5cf-e413-4e59-9f0c-eef6b8fca8c9
let
	# plt = nothing
	τ = parse(Int, truncation)
	nh = parse(Int, numhidden)
	fac = parse(Int, factors)
	
	arg_dict = [
		Dict("cell"=>"AARNN", "numhidden"=>15, "factors"=>0, "init_style"=>"na"),
		Dict("cell"=>"MARNN", "numhidden"=>12, "factors"=>0, "init_style"=>"na"),
		Dict("cell"=>"FacMARNN", "numhidden"=>15, "factors"=>12, "init_style"=>"ignore"),
		Dict("cell"=>"FacMARNN", "numhidden"=>12, "factors"=>14, "init_style"=>"ignore"),
		Dict("cell"=>"FacMARNN", "numhidden"=>15, "factors"=>12, "init_style"=>"standard"),
		Dict("cell"=>"FacMARNN", "numhidden"=>12, "factors"=>14, "init_style"=>"standard"),
		Dict("cell"=>"FacMARNN", "numhidden"=>15, "factors"=>12, "init_style"=>"tensor"),
		Dict("cell"=>"FacMARNN", "numhidden"=>12, "factors"=>14, "init_style"=>"tensor")	
		]
	
	colors = ["AARNN", "MARNN", "FacMARNN", "FacMARNN", "GRU", "GRU", "AAGRU", "AAGRU"]
	
	plt = plot()
	for (i, args_) in enumerate(arg_dict)
		if i == 4 || i == 6 || i == 8
			plt = plot!(
			  data_sens,
			  args_;
			  sort_idx="truncation",
			  z=1.97, lw=2,
			  palette=RPU.custom_colorant, label="cell: $(args_["cell"]), nh: $(args_["numhidden"]), fac: $(args_["factors"]), init: $(args_["init_style"])",
			  color=cell_colors[colors[i]], title="ER RNN RingWorld Truncation Sensitivity", linestyle=:dash, ylim=(0, 0.25))
		else
			println("cell: $(args_["cell"])")
			plt = plot!(
			  data_sens,
			  args_;
			  sort_idx="truncation",
			  z=1.97, lw=2,
			  palette=RPU.custom_colorant, label="cell: $(args_["cell"]), nh: $(args_["numhidden"]), fac: $(args_["factors"]), init: $(args_["init_style"])",
			  color=cell_colors[colors[i]], title="ER RingWorld RNN Truncation Sensitivity", ylim=(0, 0.25))
		end
	end
	plt
end



# ╔═╡ b5800088-f845-46ff-b8e3-673022c2a88a
let
	# plt = nothing
	
	arg_dict = [
		Dict("cell"=>"AARNN", "numhidden"=>15, "factors"=>0, "init_style"=>"na"),
		Dict("cell"=>"MARNN", "numhidden"=>12, "factors"=>0, "init_style"=>"na"),
		Dict("cell"=>"FacMARNN", "numhidden"=>15, "factors"=>12, "init_style"=>"ignore"),
		Dict("cell"=>"FacMARNN", "numhidden"=>12, "factors"=>14, "init_style"=>"ignore"),
		Dict("cell"=>"FacMARNN", "numhidden"=>15, "factors"=>12, "init_style"=>"standard"),
		Dict("cell"=>"FacMARNN", "numhidden"=>12, "factors"=>14, "init_style"=>"standard"),
		Dict("cell"=>"FacMARNN", "numhidden"=>15, "factors"=>12, "init_style"=>"tensor"),
		Dict("cell"=>"FacMARNN", "numhidden"=>12, "factors"=>14, "init_style"=>"tensor")	
		]
	
	colors = ["AARNN", "MARNN", "FacMARNN", "FacMARNN", "GRU", "GRU", "AAGRU", "AAGRU"]
	
	plt = plot()
	for (i, args_) in enumerate(arg_dict)
		if i == 4 || i == 6 || i == 8
			plt = plot!(
			  data_sens_on,
			  args_;
			  sort_idx="truncation",
			  z=1.97, lw=2,
			  palette=RPU.custom_colorant, label="cell: $(args_["cell"]), nh: $(args_["numhidden"]), fac: $(args_["factors"]), init: $(args_["init_style"])",
			  color=cell_colors[colors[i]], title="Online RNN RingWorld Truncation Sensitivity", linestyle=:dash, ylim=(0, 0.35))
		else
			println("cell: $(args_["cell"])")
			plt = plot!(
			  data_sens_on,
			  args_;
			  sort_idx="truncation",
			  z=1.97, lw=2,
			  palette=RPU.custom_colorant, label="cell: $(args_["cell"]), nh: $(args_["numhidden"]), fac: $(args_["factors"]), init: $(args_["init_style"])",
			  color=cell_colors[colors[i]], title="Online RingWorld RNN Truncation Sensitivity", ylim=(0, 0.35))
		end
	end
	plt
end



# ╔═╡ f686dc3c-2bc5-4edd-95eb-f7d653825c45
let
	# plt = nothing
	τ = parse(Int, truncation)
	nh = parse(Int, numhidden)
	fac = parse(Int, factors)
	
	arg_dict = [
		Dict("cell"=>"AAGRU", "numhidden"=>12, "factors"=>0, "init_style"=>"na"),
		Dict("cell"=>"MAGRU", "numhidden"=>9, "factors"=>0, "init_style"=>"na"),
		Dict("cell"=>"FacMAGRU", "numhidden"=>12, "factors"=>8, "init_style"=>"standard"),
		Dict("cell"=>"FacMAGRU", "numhidden"=>9, "factors"=>12, "init_style"=>"standard"),
		Dict("cell"=>"FacMAGRU", "numhidden"=>12, "factors"=>8, "init_style"=>"tensor"),
		Dict("cell"=>"FacMAGRU", "numhidden"=>9, "factors"=>12, "init_style"=>"tensor")	
		]
	
	colors = ["AAGRU", "MAGRU", "FacMAGRU", "FacMAGRU", "GRU", "GRU"]
	
	plt = plot()
	for (i, args_) in enumerate(arg_dict)
		if i == 4 || i == 6 || i == 8
			plt = plot!(
			  data_sens,
			  args_;
			  sort_idx="truncation",
			  z=1.97, lw=2,
			  palette=RPU.custom_colorant, label="cell: $(args_["cell"]), nh: $(args_["numhidden"]), fac: $(args_["factors"]), init: $(args_["init_style"])",
			  color=cell_colors[colors[i]], title="ER RingWorld GRU Truncation Sensitivity", linestyle=:dash, ylim=(0, 0.25))
		else
			println("cell: $(args_["cell"])")
			plt = plot!(
			  data_sens,
			  args_;
			  sort_idx="truncation",
			  z=1.97, lw=2,
			  palette=RPU.custom_colorant, label="cell: $(args_["cell"]), nh: $(args_["numhidden"]), fac: $(args_["factors"]), init: $(args_["init_style"])",
			  color=cell_colors[colors[i]], title="ER RingWorld GRU Truncation Sensitivity", ylim=(0, 0.25))
		end
	end
	plt
end



# ╔═╡ 72542c73-85bc-4082-8dbb-39843ff5eeda
let
	arg_dict = [
		Dict("cell"=>"AAGRU", "numhidden"=>12, "factors"=>0, "init_style"=>"na"),
		Dict("cell"=>"MAGRU", "numhidden"=>9, "factors"=>0, "init_style"=>"na"),
		Dict("cell"=>"FacMAGRU", "numhidden"=>12, "factors"=>8, "init_style"=>"standard"),
		Dict("cell"=>"FacMAGRU", "numhidden"=>9, "factors"=>12, "init_style"=>"standard"),
		Dict("cell"=>"FacMAGRU", "numhidden"=>12, "factors"=>8, "init_style"=>"tensor"),
		Dict("cell"=>"FacMAGRU", "numhidden"=>9, "factors"=>12, "init_style"=>"tensor")	
		]
	
	colors = ["AAGRU", "MAGRU", "FacMAGRU", "FacMAGRU", "GRU", "GRU"]
	
	plt = plot()
	for (i, args_) in enumerate(arg_dict)
		if i == 4 || i == 6 || i == 8
			plt = plot!(
			  data_sens_on,
			  args_;
			  sort_idx="truncation",
			  z=1.97, lw=2,
			  palette=RPU.custom_colorant, label="cell: $(args_["cell"]), nh: $(args_["numhidden"]), fac: $(args_["factors"]), init: $(args_["init_style"])",
			  color=cell_colors[colors[i]], title="Online RingWorld GRU Truncation Sensitivity", linestyle=:dash, ylim=(0, 0.35))
		else
			println("cell: $(args_["cell"])")
			plt = plot!(
			  data_sens_on,
			  args_;
			  sort_idx="truncation",
			  z=1.97, lw=2,
			  palette=RPU.custom_colorant, label="cell: $(args_["cell"]), nh: $(args_["numhidden"]), fac: $(args_["factors"]), init: $(args_["init_style"])",
			  color=cell_colors[colors[i]], title="Online RingWorld GRU Truncation Sensitivity", ylim=(0, 0.35))
		end
	end
	plt
end



# ╔═╡ Cell order:
# ╠═18ca04ca-a249-4c06-a992-7031e0da8353
# ╠═00c1df58-d947-11eb-36c4-653906e7fe45
# ╠═de72e089-68f2-4b16-9524-bfe6bf1dbe75
# ╠═83912db4-fa1e-4a7b-9767-390a91ba320f
# ╠═a9c22a1e-6fa3-4d32-8892-f81dc53013ec
# ╟─1b8ca44f-41d3-40d4-b32b-9299e24af861
# ╟─8292a0f5-770d-43b4-ad4f-16752b7264cd
# ╠═ee43a945-7f64-436b-b391-ad64edab4f5c
# ╠═5e534ba1-d33c-4824-abde-0e9a25eaca29
# ╠═dfc801ef-c384-4505-88bc-09914c138e0b
# ╠═48fcb600-3c42-4bd6-aa7a-33663c2e5d25
# ╠═eb61ceb3-f6ac-4224-90d0-d2fcdf83b483
# ╠═f797fa36-df84-476d-be05-f409c3793a20
# ╠═98a8531b-94c5-41f7-af88-fbfd62393f71
# ╠═21766c7b-a193-4d79-bec2-b8332edeedd2
# ╟─e96f363b-f748-4664-a8c0-e35a79af2cab
# ╠═4109beac-1012-446d-83d3-052a2fd2044e
# ╠═7043e5cf-e413-4e59-9f0c-eef6b8fca8c9
# ╠═b5800088-f845-46ff-b8e3-673022c2a88a
# ╠═f686dc3c-2bc5-4edd-95eb-f7d653825c45
# ╠═72542c73-85bc-4082-8dbb-39843ff5eeda
