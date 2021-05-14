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

# ╔═╡ 94700b85-0982-47e1-9e08-8380dd585cac
nh_colors = Dict(
	3 => color_scheme[3],
	6 => color_scheme[1],
	9 => color_scheme[5],
	12 => color_scheme[end-1],
	15 => color_scheme[4],
	20 => color_scheme[end],
	21 => color_scheme[6],
	22 => color_scheme[end-2])

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
	sort!(d, by=x -> x.line_params[end])
	d_m = []
	d_s = []
	for i in 1:length(d)
		push!(d_m, mean(d[i].data))
		push!(d_s, std(d[i].data))
	end
    if plt isa Nothing
		plt = plot(x_axis, d_m, yerror=d_s; pkwargs...)
    else
		plot!(plt, x_axis, d_m, yerror=d_s; pkwargs...)
    end
    plt
end

# ╔═╡ 0fc6fd35-5a24-4aaf-9ebb-6c4af1ba259b
ic_dir, dd_dir = @load_data "../local_data/final_ringworld_online_rmsprop_10/data/"

# ╔═╡ a117f7c7-830e-455d-8417-1ce949901937
#eta_data = PU.get_line_data_for(
#	ic_dir,
#	["numhidden", "cell", "truncation", "eta"];
#	comp=findmin,
#	get_comp_data=(x)->PU.get_AUC(x, "end"),
#	get_data=(x)->PU.get_AUC(x, "end"))

# ╔═╡ 032b80e1-be25-42e7-9bf6-df5219f34391
md"""
truncation: $(@bind truncation Select(string.(dd_dir["truncation"])))
NumHidden: $(@bind numhidden Select(string.(dd_dir["numhidden"])))
Cell: $(@bind cells MultiSelect(dd_dir["cell"]))
"""

# ╔═╡ a0ecd484-08ad-439b-92fe-33dc8071dc5d
let 
	plt = nothing
	eta = log.(dd_dir["eta"])
	nh = parse(Int, numhidden)
	trunc = parse(Int, truncation)
	for cell ∈ cells
		plt = plot_sensitivity_from_data_with_params!(plt, eta_data, (nh, cell, trunc), eta; label="Cell: $(cell)", palette=color_scheme,legend=:topright, ylabel="RMSE", xlabel="eta", ylim=(0, 0.50), title="numhidden: $(nh), truncation: $(trunc)")
	end
	plt
end

# ╔═╡ 8dee9c88-4a33-4fd5-9095-f4fccae6bbf7
sensitivity_data = PU.get_line_data_for(
	ic_dir,
	["numhidden", "cell", "truncation"],
	[];
	comp=findmin,
	get_comp_data=(x)->PU.get_AUC(x, "end"),
	get_data=(x)->PU.get_AUC(x, "end"))

# ╔═╡ 7002dd0e-01b8-4598-929c-dbff02fb9143
md"""
NumHidden: $(@bind nh_dir MultiSelect(string.(dd_dir["numhidden"])))
Cell: $(@bind cell Select(dd_dir["cell"]))
"""

# ╔═╡ 586ba32f-79b6-4c2e-80af-ef2c1224fd31
let 
	plt = nothing
	lstyle = [:solid, :dash, :dot, :solid, :dash, :dot]
	mshape = [:circle, :rect, :star5, :diamond, :hexagon, :utriangle]
	trunc = dd_dir["truncation"]
	for (i, nh) ∈ enumerate(nh_dir)
		nh_ = parse(Int, nh)
		plt = plot_sensitivity_from_data_with_params!(plt, sensitivity_data, (nh_, cell), trunc; label="$(nh)", palette=color_scheme, color=nh_colors[nh_], legend=:topright, ylabel="RMSE (Final 50k steps)", xlabel="Truncation", ylim=(0, 0.35), title="Cell: $(cell), Envsize: 10", markershape=mshape[i], markersize=5, linestyle=:solid, grid=false, tickdir=:out, legendtitlefontsize=10, legendfontsize=8, legendfonthalign=:center, lw=2)
	end
	plt
end

# ╔═╡ c0e0a43a-4bf7-416d-b412-c1ee684e1e0e
savefig("../data/sensitivity_plots/final_ringworld_online_sensitivity_plot_$(cell).png")

# ╔═╡ d7ae12f9-ec9d-4d07-83ab-97205ffab69b
begin
	args_list = Dict{String, Any}[]

	for data in sensitivity_data
		if data.line_params[3] != 9 && data.line_params[3] != 11 && data.line_params[1] != 17
			push_dict = Dict("numhidden"=>data.line_params[1], "truncation"=>data.line_params[3], "cell"=>data.line_params[2], "eta"=>data.swept_params)
			push!(args_list, push_dict)
		end
	end
end

# ╔═╡ b7e8ab78-9867-4b5b-9d7c-adc5af3c9ab2
# FileIO.save("../final_runs/ringworld_er_10.jld2", "args", args_list)

# ╔═╡ Cell order:
# ╠═f7f500a8-a1e9-11eb-009b-d7afdcade891
# ╠═e0d51e67-63dc-45ea-9092-9965f97660b3
# ╟─0c746c1e-ea39-4415-a1b1-d7124b886f98
# ╠═94700b85-0982-47e1-9e08-8380dd585cac
# ╟─92c2600e-58b4-4ea3-8272-3fe38c0422d1
# ╟─24ccf89f-ab20-447e-9d6f-633380ee8c20
# ╠═240dc563-fd04-4c07-85ac-4e54ad016374
# ╟─265db42d-1b27-463d-b0a1-cfc62748022a
# ╟─62d657ae-bdd9-454e-8d4f-39e7d044d6dd
# ╠═0fc6fd35-5a24-4aaf-9ebb-6c4af1ba259b
# ╠═a117f7c7-830e-455d-8417-1ce949901937
# ╠═032b80e1-be25-42e7-9bf6-df5219f34391
# ╠═a0ecd484-08ad-439b-92fe-33dc8071dc5d
# ╠═8dee9c88-4a33-4fd5-9095-f4fccae6bbf7
# ╠═7002dd0e-01b8-4598-929c-dbff02fb9143
# ╠═586ba32f-79b6-4c2e-80af-ef2c1224fd31
# ╠═c0e0a43a-4bf7-416d-b412-c1ee684e1e0e
# ╠═d7ae12f9-ec9d-4d07-83ab-97205ffab69b
# ╠═b7e8ab78-9867-4b5b-9d7c-adc5af3c9ab2
