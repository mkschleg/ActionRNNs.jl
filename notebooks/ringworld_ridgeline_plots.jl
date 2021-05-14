### A Pluto.jl notebook ###
# v0.14.2

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

# ╔═╡ f7f500a8-a1e9-11eb-009b-d7afdcade891
using Revise

# ╔═╡ e0d51e67-63dc-45ea-9092-9965f97660b3
using Reproduce, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ d533185e-8dd6-4a12-a2df-e119b3bf4174
using CairoMakie

# ╔═╡ d8ff6d57-7377-4aa0-a6a0-c7489df76053
#using Plots

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
PU = ingredients("plot_utils.jl")

# ╔═╡ 265db42d-1b27-463d-b0a1-cfc62748022a
macro load_data(loc)
    return quote
		local ic = ItemCollection($(loc))
		ic, diff(ic)
	end
end

# ╔═╡ 0fc6fd35-5a24-4aaf-9ebb-6c4af1ba259b
ic_dir, dd_dir = @load_data "../local_data/ringworld_er_rnn_rmsprop_10/data/"

# ╔═╡ f31124d5-33bd-4185-9361-ca508b5949fb
function get_density_data(data_col::Vector{PU.LineData}, params)
    idx = findall(data_col) do (ld)
        line_params = ld.line_params
	all([line_params[i] == params[i] for i ∈ 1:length(params)])
    end
    d = data_col[idx]
	
end

# ╔═╡ 0a6025d4-4f5c-47ee-8c99-95cb59c62b3b
ridgeline_data = PU.get_line_data_for(
	ic_dir,
	["numhidden", "truncation", "cell"],
	"eta";
	comp=findmin,
	get_comp_data=(x)->PU.get_AUC(x, "end"),
	get_data=(x)->PU.get_AUC(x, "end"))

# ╔═╡ 817ae141-17ba-4326-a594-0a4f834b3966
md"""
Truncation: $(@bind trunc_dir MultiSelect(string.(dd_dir["truncation"])))
NumHidden: $(@bind nh_dir MultiSelect(string.(dd_dir["numhidden"])))
Cell: $(@bind cell Select(dd_dir["cell"]))
"""

# ╔═╡ 43ceffc3-9ddc-4371-8733-9c13b5439e15
begin	
	density_data = []
	for nh ∈ nh_dir
		for trunc ∈ trunc_dir
			nh_ = parse(Int, nh)
			trunc_ = parse(Int, trunc)
			params = (nh_, trunc_, cell)
			density_data = cat(density_data, get_density_data(ridgeline_data, params), dims=1)
		end
	end
	sort!(density_data, by=x -> x.line_params[2])
	sort!(density_data, by=x -> x.line_params[1])
end

# ╔═╡ c80be3c8-ac71-45a7-9735-839e2938997f
begin
	y_axis = ["nh: $(data.line_params[1]), τ: $(data.line_params[2])" for data in density_data]
	x_axis_size = [maximum(data.data) - minimum(data.data) for data in density_data]
	#y_axis_range = 10 * minimum(x_axis_size)
	y_axis_range = 0.03
end

# ╔═╡ c9bb6c8c-1fef-4ffe-97ae-2d0b59c929b5
begin
	y_s = size(y_axis, 1)
	#colors = Dict(1=>:red, 2=>:green, 4=>:blue, 6=>:grey, 8=>:orange, 10=>:brown, 12=>:yellow, 15=>:purple)
	colors = Dict(3=>:red, 6=>:green, 9=>:blue, 12=>:grey, 15=>:orange, 17=>:yellow, 20=>:brown)
	
	f = Figure()
	axes = Axis(f[1, 1], title = "Cell: $(cell), EnvSize: 10", yticklabelsize=12,
		yticks = ((1:y_s) ./ y_axis_range,  y_axis), xlabel="RMSE (Final 50k steps)")

	for (i, data) in enumerate(density_data)
		mi = minimum(data.data)
		ma = maximum(data.data)
		bonus = (ma - mi)*0.5
		d = density!(data.data, offset = i / y_axis_range,
			color = (colors[data.line_params[1]], 0.4), colormap = :thermal, colorrange = (-5, 5), bandwidth=0.004)
		# this helps with layering in GLMakie
		translate!(d, 0, 0, -0.1i)
	end
	xlims!(axes, [-0.01, 0.35])
	f
end

# ╔═╡ 17e05d35-9e0a-4ff0-acb1-2ad15dda8519
#FileIO.save("../data/ringworld_ridgeline_plots/ridgeplot_er_ridgeline_plot_$(cell).png", f)

# ╔═╡ c0859846-6454-4154-93e5-dade64ad3f33
#FileIO.save("../data/ringworld_ridgeline_plots/ridgeplot_online_ridgeline_plot_$(cell).png", f)

# ╔═╡ Cell order:
# ╠═f7f500a8-a1e9-11eb-009b-d7afdcade891
# ╠═e0d51e67-63dc-45ea-9092-9965f97660b3
# ╠═d533185e-8dd6-4a12-a2df-e119b3bf4174
# ╠═d8ff6d57-7377-4aa0-a6a0-c7489df76053
# ╟─92c2600e-58b4-4ea3-8272-3fe38c0422d1
# ╟─24ccf89f-ab20-447e-9d6f-633380ee8c20
# ╟─240dc563-fd04-4c07-85ac-4e54ad016374
# ╟─265db42d-1b27-463d-b0a1-cfc62748022a
# ╠═0fc6fd35-5a24-4aaf-9ebb-6c4af1ba259b
# ╟─f31124d5-33bd-4185-9361-ca508b5949fb
# ╠═0a6025d4-4f5c-47ee-8c99-95cb59c62b3b
# ╟─817ae141-17ba-4326-a594-0a4f834b3966
# ╠═43ceffc3-9ddc-4371-8733-9c13b5439e15
# ╠═c80be3c8-ac71-45a7-9735-839e2938997f
# ╠═c9bb6c8c-1fef-4ffe-97ae-2d0b59c929b5
# ╠═17e05d35-9e0a-4ff0-acb1-2ad15dda8519
# ╠═c0859846-6454-4154-93e5-dade64ad3f33
