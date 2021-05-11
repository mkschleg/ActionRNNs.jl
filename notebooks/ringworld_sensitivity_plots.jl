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
using Reproduce, Plots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

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
PU = ingredients("plot_utils.jl")

# ╔═╡ 265db42d-1b27-463d-b0a1-cfc62748022a
macro load_data(loc)
    return quote
		local ic = ItemCollection($(loc))
		ic, diff(ic)
	end
end

# ╔═╡ 62d657ae-bdd9-454e-8d4f-39e7d044d6dd
function plot_sensitivity_from_data_with_params!(
		plt, data_col::Vector{PU.LineData}, params, x_axis; pkwargs...)
    idx = findall(data_col) do (ld)
        line_params = ld.line_params
	all([line_params[i] == params[i] for i ∈ 1:length(params)])
    end
    d = data_col[idx]
	sort!(d, by=x -> x.line_params[3])
    label = if :label ∈ keys(pkwargs)
	"numhidden: $(pkwargs[:label])"
    else
	"missing label"
    end
	d_m = []
	d_s = []
	for i in 1:length(d)
		push!(d_m, mean(d[i].data))
		push!(d_s, std(d[i].data))
	end
    if plt isa Nothing
	plt = plot(x_axis, d_m, yerror=d_s; pkwargs..., label=label)
    else
	plot!(plt, x_axis, d_m, yerror=d_s; pkwargs..., label=label)
    end
    plt
end

# ╔═╡ 0fc6fd35-5a24-4aaf-9ebb-6c4af1ba259b
ic_dir, dd_dir = @load_data "../ringworld_er_rnn_rmsprop_10/data/"

# ╔═╡ 7355c9d1-eaf7-4c06-81bd-645e1206f748
sensitivity_data = PU.get_line_data_for(
	ic_dir,
	["numhidden", "cell", "truncation"],
	"eta";
	comp=findmin,
	get_comp_data=(x)->PU.get_AUC(x, "end"),
	get_data=(x)->PU.get_AUC(x, "end"))

# ╔═╡ 032b80e1-be25-42e7-9bf6-df5219f34391
md"""
NumHidden: $(@bind nh_dir MultiSelect(string.(dd_dir["numhidden"])))
Cell: $(@bind cell Select(dd_dir["cell"]))
"""

# ╔═╡ 586ba32f-79b6-4c2e-80af-ef2c1224fd31
let 
	plt = nothing
	trunc = dd_dir["truncation"]
	for nh ∈ nh_dir
		nh_ = parse(Int, nh)
		plt = plot_sensitivity_from_data_with_params!(plt, sensitivity_data, (nh_, cell), trunc; label=nh, palette=color_scheme,legend=:topright, ylabel="Standard Error", xlabel="Truncation", ylim=(0, 0.35), title="Cell: $(cell), Envsize: 10")
	end
	plt
end

# ╔═╡ 388ee597-b238-4cc0-a2f5-9989f05849b4
savefig("../data/sensitivity_plots/ringworld_online_sensitivity_plot_$(cell).png")

# ╔═╡ 3796ccd1-8d1b-43b3-8079-61045ed76c52
savefig("../data/sensitivity_plots/ringworld_er_sensitivity_plot_$(cell).png")

# ╔═╡ Cell order:
# ╠═f7f500a8-a1e9-11eb-009b-d7afdcade891
# ╠═e0d51e67-63dc-45ea-9092-9965f97660b3
# ╟─0c746c1e-ea39-4415-a1b1-d7124b886f98
# ╟─92c2600e-58b4-4ea3-8272-3fe38c0422d1
# ╟─24ccf89f-ab20-447e-9d6f-633380ee8c20
# ╠═240dc563-fd04-4c07-85ac-4e54ad016374
# ╟─265db42d-1b27-463d-b0a1-cfc62748022a
# ╟─62d657ae-bdd9-454e-8d4f-39e7d044d6dd
# ╠═0fc6fd35-5a24-4aaf-9ebb-6c4af1ba259b
# ╠═7355c9d1-eaf7-4c06-81bd-645e1206f748
# ╟─032b80e1-be25-42e7-9bf6-df5219f34391
# ╠═586ba32f-79b6-4c2e-80af-ef2c1224fd31
# ╠═388ee597-b238-4cc0-a2f5-9989f05849b4
# ╠═3796ccd1-8d1b-43b3-8079-61045ed76c52
