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

# ╔═╡ 96e3c070-95e9-4988-90cb-79488981b7c3
let
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ f7f500a8-a1e9-11eb-009b-d7afdcade891
using Revise

# ╔═╡ e0d51e67-63dc-45ea-9092-9965f97660b3
using Reproduce, ReproducePlotUtils, Plots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 7d97a3a3-4023-4537-9262-e104979ff5a2
const RPU = ReproducePlotUtils

# ╔═╡ 0c746c1e-ea39-4415-a1b1-d7124b886f98
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

# ╔═╡ 92c2600e-58b4-4ea3-8272-3fe38c0422d1
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ 24ccf89f-ab20-447e-9d6f-633380ee8c20
Base.show(io::IO, ic::ItemCollection) = print(io, "ItemCollection(Size: ", length(ic), ", Dir Hash: ", ic.dir_hash, ")")

# ╔═╡ 240dc563-fd04-4c07-85ac-4e54ad016374
# PU = ingredients("plot_utils.jl")

# ╔═╡ 265db42d-1b27-463d-b0a1-cfc62748022a
macro load_data(loc)
    return quote
		local ic = ItemCollection($(loc))
		ic, diff(ic)
	end
end

# ╔═╡ 0fc6fd35-5a24-4aaf-9ebb-6c4af1ba259b
ic_dir, dd_dir = RPU.load_data("../../local_data/ringworld/er/ringworld_er_rnn_rmsprop_10/")

# ╔═╡ 1e3868bb-a377-469d-ac22-fc1c6db295e2
function get_heatmap_matrix_and_params(data, ic, cell, y_axis, x_axis)
    
	cell_ic = search(ic,Dict("cell"=>cell))
    diff_dict = diff(cell_ic)
    y_axis_params = diff_dict[y_axis]
    x_axis_params = diff_dict[x_axis]
	prm_names = [y_axis, x_axis, "cell"]
	
    data_matrix = Array{Float64}(undef, length(y_axis_params), length(x_axis_params))

    for ((y_param_idx, y_param), (x_param_idx, x_param)) ∈ Iterators.product(enumerate(y_axis_params), enumerate(x_axis_params))
		params = [y_param, x_param, cell]
		
		# @info params
		idx = findfirst(data.data) do (ld)
        	line_params = ld.line_params
			all([line_params[prm_names[i]] == params[i] for i ∈ 1:length(line_params)])
    	end
    	d = data.data[idx].data
        data_matrix[y_param_idx, x_param_idx] = mean(d)
    end
    data_matrix, y_axis_params, x_axis_params
end

# ╔═╡ a18b7595-a7d4-44ef-9149-45892e588909
function get_layout(num_plots)
	layout = if num_plots == 1
		@layout grid(1, 1)
	elseif num_plots == 2
		@layout [a{0.5w} b{0.5w}]
    elseif num_plots == 3
		(1, 3)
	elseif num_plots == 4
		@layout [a{0.5h} b{0.5h} c{0.5h} d{0.5h}]
	elseif num_plots == 5
		(2, 2)
	elseif num_plots == 6
		(2, 3)
	elseif num_plots == 7
		(1, 7)
	end
end

# ╔═╡ ae514cc8-d39e-4275-b6fc-655a63aa9f0b
function plot_heatmaps(heatmap_matrices, titles, x_params, y_params; pkwargs...)
	heatmaps = []
	num_hm = size(heatmap_matrices, 1)
	for (i, heatmap_matrix) in enumerate(heatmap_matrices)
		hm = heatmap(string.(x_params), string.(y_params), heatmap_matrix; title=titles[i], pkwargs...)
		push!(heatmaps, hm)
	end
	heatmap(heatmaps..., layout = get_layout(num_hm))
end

# ╔═╡ 4922b08c-52f8-46e8-9a7e-6107057cf33b
FileIO.load(joinpath(ic_dir[1].folder_str, "results.jld2"))

# ╔═╡ 323923f1-4681-4ee8-aac7-3346764b2cc1
heatmap_data = RPU.get_line_data_for(
	ic_dir,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->x["results"]["end"],
	get_data=(x)->x["results"]["end"])

# ╔═╡ d97ba8f4-1a73-4f60-ae86-6bebfad1429a
heatmap_data.data[1].line_params

# ╔═╡ 032b80e1-be25-42e7-9bf6-df5219f34391
md"""
Cell: $(@bind cells_dir MultiSelect(dd_dir["cell"]))
"""

# ╔═╡ ec8d12c6-7ead-4bfc-a7c5-aaf0b3020c8f
let
	heatmap_matrices = []
	heatmap_titles = []
	y_axis_params = x_axis_params = nothing
	for cell ∈ cells_dir
		heatmap_matrix, y_axis_params, x_axis_params = get_heatmap_matrix_and_params(heatmap_data, ic_dir, cell, "numhidden", "truncation")
		push!(heatmap_matrices, heatmap_matrix)
		push!(heatmap_titles, "Cell: $(cell), Envsize:10")
	end
	plot_heatmaps(heatmap_matrices, heatmap_titles, x_axis_params, y_axis_params, xlabel="truncation", ylabel="numhidden", clims=(0, 0.50), titlefontsize=12, xguidefontsize=10, yguidefontsize=10, xtickfontsize=8, ytickfontsize=8, legend=true)
end

# ╔═╡ 262cc47f-f36a-4c4a-8a07-7681da2c5dca
# savefig("../data/heatmaps/ringworld_online_heatmap_$(cells_dir[1]).png")

# ╔═╡ 71479867-2e59-46cd-8f38-aac90344d343
# savefig("../data/heatmaps/ringworld_er_heatmap_$(cells_dir[1]).png")

# ╔═╡ Cell order:
# ╠═96e3c070-95e9-4988-90cb-79488981b7c3
# ╠═f7f500a8-a1e9-11eb-009b-d7afdcade891
# ╠═e0d51e67-63dc-45ea-9092-9965f97660b3
# ╠═7d97a3a3-4023-4537-9262-e104979ff5a2
# ╟─0c746c1e-ea39-4415-a1b1-d7124b886f98
# ╟─92c2600e-58b4-4ea3-8272-3fe38c0422d1
# ╠═24ccf89f-ab20-447e-9d6f-633380ee8c20
# ╠═240dc563-fd04-4c07-85ac-4e54ad016374
# ╟─265db42d-1b27-463d-b0a1-cfc62748022a
# ╠═0fc6fd35-5a24-4aaf-9ebb-6c4af1ba259b
# ╠═1e3868bb-a377-469d-ac22-fc1c6db295e2
# ╟─a18b7595-a7d4-44ef-9149-45892e588909
# ╠═ae514cc8-d39e-4275-b6fc-655a63aa9f0b
# ╠═4922b08c-52f8-46e8-9a7e-6107057cf33b
# ╠═323923f1-4681-4ee8-aac7-3346764b2cc1
# ╠═d97ba8f4-1a73-4f60-ae86-6bebfad1429a
# ╠═032b80e1-be25-42e7-9bf6-df5219f34391
# ╠═ec8d12c6-7ead-4bfc-a7c5-aaf0b3020c8f
# ╠═262cc47f-f36a-4c4a-8a07-7681da2c5dca
# ╠═71479867-2e59-46cd-8f38-aac90344d343
