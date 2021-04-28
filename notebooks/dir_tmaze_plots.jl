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

# ╔═╡ 834b1cf3-5b22-4e0a-abe9-61e427e6cfda
function plot_line_from_data_with_params!(
		plt, data_col::Vector{PU.LineData}, params; pkwargs...)
    idx = findfirst(data_col) do (ld)
        line_params = ld.line_params
	all([line_params[i] == params[i] for i ∈ 1:length(line_params)])
    end
    d = data_col[idx]
    label = if :label ∈ keys(pkwargs)
	"$(pkwargs[:label]), $(d.swept_params))"
    else
	d.swept_params
    end
    if plt isa Nothing
	plt = plot(PU.mean_uneven(d.data), ribbon=PU.std_uneven(d.data); pkwargs..., label=label)
    else
	plot!(plt, PU.mean_uneven(d.data), ribbon=PU.std_uneven(d.data); pkwargs..., label=label)
    end
    plt
end

# ╔═╡ 0fc6fd35-5a24-4aaf-9ebb-6c4af1ba259b
ic_dir_6, dd_dir_6 = @load_data "../local_data/dir_tmaze_er_rnn_rmsprop/data/"

# ╔═╡ 6211a38a-7b53-4054-970e-c29ad17de646
ic_dir_6[1].parsed_args["steps"]

# ╔═╡ e822182e-b485-4a95-a08c-efe1540ff6ad
data_6 = PU.get_line_data_for(
	ic_dir_6,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->PU.get_AUC(x, :successes),
	get_data=(x)->PU.get_rolling_mean_line(x, :successes, 300))

# ╔═╡ c1b80cfd-dbb8-41c5-a778-66b112e1c091
md"""
Truncation: $(@bind τ_dir_6 Select(string.(dd_dir_6["truncation"])))
NumHidden: $(@bind nh_dir_6 Select(string.(dd_dir_6["numhidden"])))
Cell: $(@bind cells_dir_6 MultiSelect(dd_dir_6["cell"]))
"""

# ╔═╡ e4cc9109-d8b1-4bff-a176-3627e24ab757
let 
	plt = nothing
	τ = parse(Int, τ_dir_6)
	nh = parse(Int, nh_dir_6)
	for cell ∈ cells_dir_6
		if cell[1:2] == "AA"
			plt = plot_line_from_data_with_params!(plt, data_6, (nh, τ, cell); label=cell, palette=color_scheme,legend=:bottomright, ylabel="Perc Success", xlabel="Episode")
		else
			plt = plot_line_from_data_with_params!(plt, data_6, (nh, τ, cell); label=cell, palette=color_scheme)
		end
	end
	plt
	# savefig("example_plot.pdf")
end

# ╔═╡ b265f4e5-7405-4fd9-bb91-377cb9e32789
let 
	plt = nothing
	τ = parse(Int, τ_dir_6)
	nh = parse(Int, nh_dir_6)
	for cell ∈ cells_dir_6
		if cell[1:2] == "AA"
			plt = plot_line_from_data_with_params!(plt, data_6, (Int(1.5*nh), τ, cell); label=cell, palette=color_scheme)
		else
			plt = plot_line_from_data_with_params!(plt, data_6, (nh, τ, cell); label=cell, palette=color_scheme)
		end
	end
	plt
end

# ╔═╡ 0eb4c818-b533-4695-a0c7-53e72023281f
let 
	plt = nothing
	args_list = [
		Dict("numhidden"=>6, "truncation"=>8, "cell"=>"MAGRU", "eta"=>0.00125),
		Dict("numhidden"=>9, "truncation"=>8, "cell"=>"AAGRU", "eta"=>0.00125),
		Dict("numhidden"=>6, "truncation"=>8, "cell"=>"AAGRU", "eta"=>1.953125e-5),
		Dict("numhidden"=>9, "truncation"=>8, "cell"=>"GRU", "eta"=>1.953125e-5),
		Dict("numhidden"=>6, "truncation"=>8, "cell"=>"MARNN", "eta"=>0.0003125),
		Dict("numhidden"=>9, "truncation"=>8, "cell"=>"AARNN", "eta"=>0.00125),
		Dict("numhidden"=>6, "truncation"=>8, "cell"=>"AARNN", "eta"=>0.00125),
		Dict("numhidden"=>9, "truncation"=>8, "cell"=>"RNN", "eta"=>0.0003125)
	]
	for args ∈ args_list
		plt = plot_line_from_data_with_params!(
			plt, 
			data_6, 
			(args["numhidden"], args["truncation"], args["cell"]); 
			label=args["cell"], 
			palette=color_scheme)
	end
	plt
end

# ╔═╡ Cell order:
# ╠═f7f500a8-a1e9-11eb-009b-d7afdcade891
# ╠═e0d51e67-63dc-45ea-9092-9965f97660b3
# ╟─0c746c1e-ea39-4415-a1b1-d7124b886f98
# ╟─92c2600e-58b4-4ea3-8272-3fe38c0422d1
# ╠═24ccf89f-ab20-447e-9d6f-633380ee8c20
# ╠═240dc563-fd04-4c07-85ac-4e54ad016374
# ╠═265db42d-1b27-463d-b0a1-cfc62748022a
# ╠═834b1cf3-5b22-4e0a-abe9-61e427e6cfda
# ╠═0fc6fd35-5a24-4aaf-9ebb-6c4af1ba259b
# ╠═6211a38a-7b53-4054-970e-c29ad17de646
# ╠═e822182e-b485-4a95-a08c-efe1540ff6ad
# ╠═c1b80cfd-dbb8-41c5-a778-66b112e1c091
# ╠═e4cc9109-d8b1-4bff-a176-3627e24ab757
# ╠═b265f4e5-7405-4fd9-bb91-377cb9e32789
# ╠═0eb4c818-b533-4695-a0c7-53e72023281f
