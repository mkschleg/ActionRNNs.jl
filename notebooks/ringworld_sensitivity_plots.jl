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
using Reproduce, ReproducePlotUtils, Plots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 2ae940a2-9624-4426-8bba-1e85b51db18e
using JLD2

# ╔═╡ fbdbb061-f620-4704-ba0f-9e4d00ddcc8f
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
	colorant"#1E90FF",
]

# ╔═╡ 55b8266e-f72e-4cdf-a987-5321f1e5d953
push!(RPU.stats_plot_types, :dotplot)

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
		plt, data_col::Vector{PU.LineData}, params, x_axis; z=nothing, pkwargs...)
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
		if !(z isa Nothing)
			push!(d_s, z * (std(d[i].data)/sqrt(length(d[i].data))))
		else
			push!(d_s, std(d[i].data))
		end
	end
    if plt isa Nothing
		plt = plot(x_axis, d_m, yerror=d_s; pkwargs...)
    else
		plot!(plt, x_axis, d_m, yerror=d_s; pkwargs...)
    end
    plt
end

# ╔═╡ dde0ed32-3b8a-40ac-ae06-16dbb7c62856
function plot_line_from_data_with_params!(
 		plt, data_col::Vector{PU.LineData}, params; pkwargs...)
    idx = findfirst(data_col) do (ld)
        line_params = ld.line_params
 	all([line_params[i] == params[i] for i ∈ 1:length(line_params)])
    end
    d = data_col[idx]
	
    if plt isa Nothing
		plt = plot(PU.mean_uneven(d.data), ribbon=PU.std_uneven(d.data); pkwargs...)
    else
		plot!(plt, PU.mean_uneven(d.data), ribbon=PU.std_uneven(d.data); pkwargs...)
	end
	plt
 end

# ╔═╡ 24f92c1e-3b6c-4d0e-bfd1-6f45bc1a72c0
ic_dir_rpu_er, dd_dir_rpu_er = RPU.load_data("../local_data/final_ringworld_er_rmsprop_10/")

# ╔═╡ 47a4712f-8833-4916-b240-96f17ee6b504
ic_dir_rpu_online_all, dd_dir_rpu_online_all = RPU.load_data("../local_data/ringworld_online_rmsprop_size10/")

# ╔═╡ 4fd60991-9844-4b1b-a756-51312dbbe33b
ic_dir_rpu_online, dd_dir_rpu_online = RPU.load_data("../local_data/RW_final/final_ringworld_online_rmsprop_10/")

# ╔═╡ 98ed8227-78e6-410e-8937-3c2c18b305ef
ic_dir_rpu_online_t12, dd_dir_rpu_online_t12 = RPU.load_data("../local_data/RW_final/final_ringworld_online_rmsprop_10_t12/")

# ╔═╡ 98858fea-5e35-4e16-bf8f-df95e95ca6db
ic_dir_rpu_online_t12_grus, dd_dir_rpu_online_t12_grus = RPU.load_data("../local_data/RW_final/final_ringworld_online_rmsprop_10_t12_grus/")

# ╔═╡ c5c76526-8e14-469d-ae96-bf0835864f55
ic_dir_rpu_online_1M, dd_dir_rpu_online_1M = RPU.load_data("../local_data/final_ringworld_online_rmsprop_10_1M/")

# ╔═╡ 8ae03ea3-fecd-413b-b612-9201e4cab552
ic_dir_rpu_online_1M_slr, dd_dir_rpu_online_1M_slr = RPU.load_data("../local_data/final_ringworld_online_rmsprop_10_1M_slr/")

# ╔═╡ 59838b82-2402-4c76-811f-2d8896f10e2f
ic_dir_rpu_online_1M_mlr, dd_dir_rpu_online_1M_mlr = RPU.load_data("../local_data/final_ringworld_online_rmsprop_10_1M_mlr/")

# ╔═╡ 8d9a77ff-974f-4cb1-a01a-e03b8951fb70
ic_dir_rpu_online_facmarnn, dd_dir_rpu_online_facmarnn = RPU.load_data("../local_data/ringworld_online_rmsprop_10_facmarnn/")

# ╔═╡ 3a5ae1d2-394b-410b-b3bf-0b2d90198db3
ic_dir_rpu_online_facmagru, dd_dir_rpu_online_facmagru = RPU.load_data("../local_data/ringworld_online_rmsprop_10_facmagru/")

# ╔═╡ d6098ab5-4090-4568-92e7-83e659ee17eb
data_rpu_er = RPU.get_line_data_for(
	ic_dir_rpu_er,
	["numhidden", "truncation", "cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 1))

# ╔═╡ 6afe6d05-c4d3-4875-b240-d9fd7ab0759e
data_online_sens = RPU.get_line_data_for(
	ic_dir_rpu_online_all,
	["numhidden", "truncation", "cell", "eta"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_AUC(x, "end"))

# ╔═╡ 8bee01d3-ee5a-4bc5-beb3-00095d487a78
dd_dir_all = diff(ic_dir_rpu_online_all)

# ╔═╡ 736b6799-2dd7-473e-934a-2bf3c2fb6fa0
md"""
Number Hidden: $(@bind nh_dir_sens Select(string.(dd_dir_all["numhidden"])))
Truncation: $(@bind τ_dir_sens Select(string.(dd_dir_all["truncation"])))
Cell: $(@bind cells_dir_sens MultiSelect(dd_dir_all["cell"]))
"""

# ╔═╡ 1377c495-47e0-41c1-9ca6-eebc1a5a9dde
begin
	sub_ic_rnn = search(ic_dir_rpu_online, Dict("cell"=>"RNN"))
	sub_ic_aarnn = search(ic_dir_rpu_online, Dict("cell"=>"AARNN"))
	sub_ic_marnn = search(ic_dir_rpu_online, Dict("cell"=>"MARNN"))
	sub_ic_facmarnn = search(ic_dir_rpu_online, Dict("cell"=>"FacMARNN"))
	sub_ic_rnn_t12 = search(ic_dir_rpu_online_t12, Dict("cell"=>"RNN"))
	sub_ic_aarnn_t12 = search(ic_dir_rpu_online_t12, Dict("cell"=>"AARNN"))
	sub_ic_marnn_t12 = search(ic_dir_rpu_online_t12, Dict("cell"=>"MARNN"))
	sub_ic_facmarnn_t12 = search(ic_dir_rpu_online_t12, Dict("cell"=>"FacMARNN"))
	sub_ic_gru = search(ic_dir_rpu_online, Dict("cell"=>"GRU"))
	sub_ic_aagru = search(ic_dir_rpu_online, Dict("cell"=>"AAGRU"))
	sub_ic_magru = search(ic_dir_rpu_online, Dict("cell"=>"MAGRU"))
	sub_ic_facmagru = search(ic_dir_rpu_online, Dict("cell"=>"FacMAGRU"))
	sub_ic_gru_t12 = search(ic_dir_rpu_online_t12_grus, Dict("cell"=>"GRU"))
	sub_ic_aagru_t12 = search(ic_dir_rpu_online_t12_grus, Dict("cell"=>"AAGRU"))
	sub_ic_magru_t12 = search(ic_dir_rpu_online_t12_grus, Dict("cell"=>"MAGRU"))
	sub_ic_facmagru_t12 = search(ic_dir_rpu_online_t12_grus, Dict("cell"=>"FacMAGRU"))
end

# ╔═╡ 3022d29f-e500-4669-ba68-b516f0c00244
begin
	data_online_rnn = RPU.get_line_data_for(
	sub_ic_rnn,
	["numhidden"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	data_online_aarnn = RPU.get_line_data_for(
	sub_ic_aarnn,
	["numhidden"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	data_online_marnn = RPU.get_line_data_for(
	sub_ic_marnn,
	["numhidden"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	data_online_facmarnn = RPU.get_line_data_for(
	sub_ic_facmarnn,
	["numhidden"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	data_online_rnn_t12 = RPU.get_line_data_for(
	sub_ic_rnn_t12,
	[],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	data_online_aarnn_t12 = RPU.get_line_data_for(
	sub_ic_aarnn_t12,
	[],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	data_online_marnn_t12 = RPU.get_line_data_for(
	sub_ic_marnn_t12,
	[],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	data_online_facmarnn_t12 = RPU.get_line_data_for(
	sub_ic_facmarnn_t12,
	[],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	data_online_gru = RPU.get_line_data_for(
	sub_ic_gru,
	[],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	data_online_aagru = RPU.get_line_data_for(
	sub_ic_aagru,
	[],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	data_online_magru = RPU.get_line_data_for(
	sub_ic_magru,
	[],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	data_online_facmagru = RPU.get_line_data_for(
	sub_ic_facmagru,
	[],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	
	
	data_online_gru_t12 = RPU.get_line_data_for(
	sub_ic_gru_t12,
	[],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	data_online_aagru_t12 = RPU.get_line_data_for(
	sub_ic_aagru_t12,
	[],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	data_online_magru_t12 = RPU.get_line_data_for(
	sub_ic_magru_t12,
	[],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
	
	data_online_facmagru_t12 = RPU.get_line_data_for(
	sub_ic_facmagru_t12,
	[],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))
end

# ╔═╡ cf697ad0-ee10-4511-bfce-4459ff0540b9
data_rpu_online = RPU.get_line_data_for(
	ic_dir_rpu_online,
	["numhidden", "truncation", "cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ 7441f85a-d8ba-47ab-983c-f32dc1761b41
data_rpu_online_1M = RPU.get_line_data_for(
	ic_dir_rpu_online_1M,
	["cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ 1dd607c8-b816-4995-bb63-7e6fdef68bd6
data_rpu_online_1M_slr = RPU.get_line_data_for(
	ic_dir_rpu_online_1M_slr,
	["cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ 5d5aad1f-4864-474f-ba31-34e007daceec
data_rpu_online_1M_mlr = RPU.get_line_data_for(
	ic_dir_rpu_online_1M_mlr,
	["cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ 1d636cfc-d163-495d-adf3-049ffee74c3b
data_rpu_online_facmarnn = RPU.get_line_data_for(
	ic_dir_rpu_online_facmarnn,
	["numhidden_factors"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ 15a9152e-ad82-4fe1-945c-4e87c5a434ef
data_rpu_online_facmagru = RPU.get_line_data_for(
	ic_dir_rpu_online_facmagru,
	["numhidden_factors"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ 67666b0f-fabd-4a8a-aca8-365c66526462
ic_dir, dd_dir = ic_dir_rpu_online, dd_dir_rpu_online

# ╔═╡ 72d108e9-5c72-4784-b31f-fd0cc34f5ff2
ic_dir[1].parsed_args["steps"]

# ╔═╡ 032b80e1-be25-42e7-9bf6-df5219f34391
md"""
truncation: $(@bind truncation Select(string.(dd_dir["truncation"])))
NumHidden: $(@bind numhidden Select(string.(dd_dir["numhidden"])))
Cell: $(@bind cells MultiSelect(dd_dir["cell"]))
"""

# ╔═╡ 55e74d54-a303-4285-8f28-e8b7947f9555
let
	# plt = nothing
	τ = parse(Int, truncation)
	nh = parse(Int, numhidden)
	plt = plot()
	for cell ∈ cells
		plt = plot!(
			  data_rpu_online,
			  Dict("numhidden"=>nh, "truncation"=>τ, "cell"=>cell),
			  palette=RPU.custom_colorant, legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.50), title="ER, numhidden: $(nh), truncation: $(trunc)", grid=false, tickdir=:out, legendtitle="Cell")
	end
	plt
end

# ╔═╡ 379f4f22-c716-4a24-b75f-3eb127b94334
#savefig("../data/ringworld_lc_plots/final_ringworld_er_lc_plot_same_params.pdf")

# ╔═╡ 1e0f4ca4-0ade-433d-a4dd-13537e5fc6b4
data = data_rpu_online

# ╔═╡ bd790756-66ec-4b82-af61-10eaf209e1ff
data_rpu_online

# ╔═╡ 242424b3-7135-4209-afe4-82429a858b66
lc_data = RPU.get_line_data_for(
	ic_dir_rpu_online_all,
	["numhidden", "cell", "truncation"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ 19ea46d1-40f2-4e4f-8c97-72db8fb86482
lc_data_final = RPU.get_line_data_for(
	ic_dir_rpu_online,
	["numhidden", "cell", "truncation"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_rolling_mean_line(x, "lc", 10))

# ╔═╡ c8c20326-d661-438f-bd23-8982f1a6329e
data_dist_online = RPU.get_line_data_for(
	ic_dir_rpu_online,
	["numhidden", "truncation", "cell"],
	[];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_AUC(x, "end"))

# ╔═╡ 69df05a1-dcd2-498a-8fc4-da95b62b5add
data_dist_online_facmarnn = RPU.get_line_data_for(
	ic_dir_rpu_online_facmarnn,
	["numhidden_factors"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_AUC(x, "end"))

# ╔═╡ 22db7508-72f0-40cc-9d8e-b7843dba4d3e
#savefig("../data/ringworld_lc_plots/final_ringworld_online_lc_plot_same_params.pdf")

# ╔═╡ 8dee9c88-4a33-4fd5-9095-f4fccae6bbf7
sensitivity_data = PU.get_line_data_for(
	ic_dir_rpu_online_all,
	["numhidden", "cell", "truncation"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->PU.get_AUC(x, "end"),
	get_data=(x)->PU.get_AUC(x, "end"))

# ╔═╡ 1d888ffa-0d14-4e71-9cb4-607836adc405
sensitivity_data_ne = PU.get_line_data_for(
	ic_dir_rpu_online,
	["numhidden", "cell", "truncation"],
	[];
	comp=findmin,
	get_comp_data=(x)->PU.get_AUC(x, "end"),
	get_data=(x)->PU.get_AUC(x, "end"))

# ╔═╡ b2952b4c-4d2e-4c65-9a0c-5b067053a89d
begin
	eta_val = nothing
	for data in sensitivity_data
		if data.line_params == (12, "GRU", 12)
			eta_val = data
		end
	end
	eta_val
end

# ╔═╡ 63da8a2e-01e9-4ae6-aef7-1c13f743fdf0
sub_ic_g = search(ic_dir_rpu_online, Dict("cell"=>"GRU", "numhidden"=>12, "truncation"=>12))

# ╔═╡ b0ca03a1-20f4-4ba2-b0fb-8a48fe9696b2
sub_ic_g[1].parsed_args["eta"]

# ╔═╡ 028b3b1c-63cd-4c5f-a472-90f058df2a00
begin
	args_list_12 = [
		Dict("numhidden"=>12, "truncation"=>12, "cell"=>"GRU", "eta"=>0.0009095),
		Dict("numhidden"=>12, "truncation"=>12, "cell"=>"AAGRU", "eta"=>0.003725),
		Dict("numhidden"=>9, "truncation"=>12, "cell"=>"MAGRU", "eta"=>0.003725), 
		Dict("numhidden"=>12, "factors"=>10, "truncation"=>12, "cell"=>"FacMAGRU", "eta"=>0.003725)
		]
	
	FileIO.save("../final_runs/ringworld_online_10_t12_grus.jld2", "args", args_list_12)
end

# ╔═╡ 0ab2174b-9e6d-431a-a9cd-10d2518a759b
begin
	different_vals = []
	for data in sensitivity_data
		a = data.line_params
		if a[3] != 9 && a[3] != 11
			final_eta = search(ic_dir_rpu_online, Dict("cell"=>a[2], "numhidden"=>a[1], "truncation"=>a[3]))[1].parsed_args["eta"]
			if data.swept_params[1] != final_eta
				push!(different_vals, data)
			end
		end
	end
	different_vals
end

# ╔═╡ da215349-0686-405a-a7f7-8012a86382ab
ic_dir_rpu_online

# ╔═╡ 7002dd0e-01b8-4598-929c-dbff02fb9143
md"""
NumHidden: $(@bind nh_dir MultiSelect(string.(dd_dir_all["numhidden"])))
Cell: $(@bind cell Select(dd_dir_all["cell"]))
"""

# ╔═╡ 92e35913-4eec-47cc-ac8e-48ec0a22533e

savefig("../data/sensitivity_plots/sweep_ringworld_online_sensitivity_plot_95c_$(cell).png")

# ╔═╡ d864b0dd-90a3-4e91-9022-8146568575cc
let 
	plt = nothing
	lstyle = [:solid, :dash, :dot, :solid, :dash, :dot]
	mshape = [:circle, :rect, :star5, :diamond, :hexagon, :utriangle]
	trunc = dd_dir["truncation"]
	for (i, nh) ∈ enumerate(nh_dir)
		nh_ = parse(Int, nh)
		plt = plot_sensitivity_from_data_with_params!(plt, sensitivity_data_ne, (nh_, cell), trunc; label="$(nh)", palette=color_scheme, color=nh_colors[nh_], legend=:topright, ylabel="RMSE (Final 50k steps)", xlabel="Truncation", ylim=(0, 0.35), title="Cell: $(cell), Envsize: 10", markershape=mshape[i], markersize=5, linestyle=:solid, grid=false, tickdir=:out, legendtitlefontsize=10, legendfontsize=8, legendfonthalign=:center, lw=2, z=1.97)
	end
	plt
end

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

# ╔═╡ cbe39710-5202-411f-bd42-368de800e5f1
begin
	the_arg = nothing
	for  arg in args_list
		if arg["numhidden"] == 9 && arg["truncation"] == 12 && arg["cell"] == "MAGRU"
			the_arg = arg
		end
	end
	the_arg
end

# ╔═╡ 1a02cb1b-2495-482d-9588-2a93f3abac47
begin
	args_list_hc = [
		Dict("numhidden"=>20, "truncation"=>8, "cell"=>"RNN", "eta"=>0.003725),
		Dict("numhidden"=>20, "truncation"=>8, "cell"=>"AARNN", "eta"=>0.003725),
		Dict("numhidden"=>15, "truncation"=>8, "cell"=>"MARNN", "eta"=>0.0009095), 
		Dict("numhidden"=>20, "factors"=>10, "truncation"=>8, "cell"=>"FacMARNN", "eta"=>0.0009095),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"RNN", "eta"=>0.00005421),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"AARNN", "eta"=>0.003725),
		Dict("numhidden"=>9, "truncation"=>8, "cell"=>"MARNN", "eta"=>0.003725), 
		Dict("numhidden"=>12, "factors"=>10, "truncation"=>8, "cell"=>"FacMARNN", "eta"=>0.0009095),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"GRU", "eta"=>0.00005421), 
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"AAGRU", "eta"=>0.003725),
		Dict("numhidden"=>9, "truncation"=>8, "cell"=>"MAGRU", "eta"=>0.003725), 
		Dict("numhidden"=>12, "factors"=>10, "truncation"=>8, "cell"=>"FacMAGRU", "eta"=>0.003725)
		]
	
	FileIO.save("../final_runs/ringworld_online_10.jld2", "args", args_list_hc)
end

# ╔═╡ 8c60f5fd-0c62-4d1a-b7d8-91b02a14b97d
f=jldopen("../final_runs/ringworld_online_10_old.jld2", "r")

# ╔═╡ be347bbc-1549-4dcd-add3-1ee3e91a76f7
begin
	f__=jldopen("/Users/Vtkachuk/Desktop/Work/Research/Martha White/Matt Schlegel/ActionRNNs.jl/local_data/final_ringworld_online_rmsprop_10/settings/settings_0xed290a678f32f129.jld2", "r")
	data__ = read(f, keys(f)[1])
end

# ╔═╡ e510936c-2b38-4669-99ab-29ef6ab3ce16
begin
	settings_val = 1
	for  arg in data__
		if arg["numhidden"] == 9 && arg["truncation"] == 12 && arg["cell"] == "MAGRU"
			settings_val = arg
		end
	end
	settings_val
end

# ╔═╡ a83ddc94-9fa8-4411-a78a-fef19444096e
data_ = read(f, keys(f)[1])

# ╔═╡ 64919972-1b56-4ba5-b9e5-c7aa247aed85
color_scheme_ = [
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

# ╔═╡ e78b5abb-3ee3-48fd-901a-40d49f48f664
cell_colors = Dict(
	"RNN" => color_scheme_[3],
	"AARNN" => color_scheme_[end],
	"MARNN" => color_scheme_[5],
	"FacMARNN" => color_scheme_[1],
	"GRU" => color_scheme_[4],
	"AAGRU" => color_scheme_[2],
	"MAGRU" => color_scheme_[6],
	"FacMAGRU" => color_scheme_[end-2])

# ╔═╡ 85d2153a-8cb1-40f8-aa3d-3d3faa1fa10e
let
	# plt = nothing
	nh = parse(Int, nh_dir_sens)
	τ = parse(Int, τ_dir_sens)
	plt = plot()
	for cell ∈ cells_dir_sens
		plt = plot!(data_online_sens,
	 	  Dict("cell"=>cell, "truncation"=>τ, "numhidden"=>nh);
	 	  sort_idx="eta",
		  z=1.97, lw=2, xaxis=:log,
		  palette=RPU.custom_colorant, label="cell: $(cell)",
		  color=cell_colors[cell], title="Truncation: $(τ), numhidden: $(nh)")
	end
	plt
end

# ╔═╡ 427d801e-f2a1-4417-b505-14cd90968f67
let 
	args_list = [
		Dict("numhidden"=>9, "truncation"=>6, "cell"=>"MARNN"),
		Dict("numhidden"=>12, "truncation"=>6, "cell"=>"AARNN"),
		Dict("numhidden"=>12, "truncation"=>6, "cell"=>"RNN")
	]
	plt = plot()
	for args ∈ args_list
		plt = plot!(data_rpu_er, args; z=1.97, lw=2, label="$(args["cell"]) (nh: $(args["numhidden"]), τ: $(args["truncation"]))", palette=RPU.custom_colorant,legend=:none, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld ER, Size: 10", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"])
	end
	plt
end

# ╔═╡ f6f5a8ac-e1a3-41e2-a38c-5f31d2c60505
let 
	#args_list = [
	#	Dict("numhidden"=>15, "truncation"=>12, "cell"=>"MARNN"),
	#	Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AARNN"),
	#	Dict("numhidden"=>20, "truncation"=>12, "cell"=>"RNN")
	#]
	args_list = [
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"MARNN"),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AARNN"),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"RNN")
	]
	plt = plot()
	for args ∈ args_list
		plt = plot!(data_rpu_online, args; z=1.97, lw=2, label="$(args["cell"]) (nh: $(args["numhidden"]), τ: $(args["truncation"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online, Size: 10", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"])
	end
	plt
end

# ╔═╡ 6c795e94-7196-4832-ab0f-aba17ea551f2
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
end

# ╔═╡ 093eec7f-b57d-4e6f-a971-f1b60cde0220
let 
	plt = plot(data_online_rnn_t12, [Dict()]; z=1.97, lw=2, label="RNN (nh: 20, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.4)
	
		plt = plot!(data_online_aarnn_t12, [Dict()]; z=1.97, lw=2, label="AARNN (nh: 20, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_marnn_t12, [Dict()]; z=1.97, lw=2, label="MARNN (nh: 15, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.4)

	plt
end

# ╔═╡ 39bb5c7b-3414-49c9-824c-ca50dee45cbb
let 
	plt = plot(data_online_rnn, [Dict("numhidden"=>20)]; z=1.97, lw=2, label="RNN (nh: 20, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.4)
	
		plt = plot!(data_online_aarnn, [Dict("numhidden"=>20)]; z=1.97, lw=2, label="AARNN (nh: 20, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_marnn, [Dict("numhidden"=>15)]; z=1.97, lw=2, label="MARNN (nh: 15, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.4)

	plt
end

# ╔═╡ 3ee90492-bc4e-401a-94ae-7388e259f6f9
let 
	plt = plot(data_online_rnn, [Dict("numhidden"=>12)]; z=1.97, lw=2, label="RNN (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.4)
	
		plt = plot!(data_online_aarnn, [Dict("numhidden"=>12)]; z=1.97, lw=2, label="AARNN (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_marnn, [Dict("numhidden"=>9)]; z=1.97, lw=2, label="MARNN (nh: 9, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.4)

	plt
end

# ╔═╡ 46580846-3fa3-4311-a554-40a2c5ec3ccf
let 
	plt = plot(data_online_gru, [Dict()]; z=1.97, lw=2, label="GRU (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["GRU"], fillalpha=0.4)
	
		plt = plot!(data_online_aagru, [Dict()]; z=1.97, lw=2, label="AAGRU (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AAGRU"], fillalpha=0.4)
	
		plt = plot!(data_online_magru, [Dict()]; z=1.97, lw=2, label="MAGRU (nh: 9, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MAGRU"], fillalpha=0.4)

	plt
end

# ╔═╡ 346dd791-4b42-4f7d-a5f6-e6e1032cade9
let 
	plt = plot(data_online_rnn_t12, [Dict()]; z=1.97, lw=2, label="RNN (nh: 20, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.4)
	
		plt = plot!(data_online_aarnn_t12, [Dict()]; z=1.97, lw=2, label="AARNN (nh: 20, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_marnn_t12, [Dict()]; z=1.97, lw=2, label="MARNN (nh: 15, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_facmarnn_t12, [Dict()]; z=1.97, lw=2, label="FacMARNN (nh: 20, fac: 10, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["FacMARNN"], fillalpha=0.4)

	plt
end

# ╔═╡ 84edfef4-f525-452b-8f5d-d44e17b93a3b
let 
	plt = plot(data_online_rnn, [Dict("numhidden"=>20)]; z=1, lw=2, label="RNN (nh: 20, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.4)
	
		plt = plot!(data_online_aarnn, [Dict("numhidden"=>20)]; z=1, lw=2, label="AARNN (nh: 20, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_marnn, [Dict("numhidden"=>15)]; z=1, lw=2, label="MARNN (nh: 15, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_facmarnn, [Dict("numhidden"=>20)]; z=1, lw=2, label="FacMARNN (nh: 20, fac: 10, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["FacMARNN"], fillalpha=0.4)

	plt
end

# ╔═╡ 7cf29ef3-b311-44e8-9036-de788e85cfcc
let 
	plt = plot(data_online_rnn, [Dict("numhidden"=>12)]; z=1, lw=2, label="RNN (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.4)
	
		plt = plot!(data_online_aarnn, [Dict("numhidden"=>12)]; z=1, lw=2, label="AARNN (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_marnn, [Dict("numhidden"=>9)]; z=1, lw=2, label="MARNN (nh: 9, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_facmarnn, [Dict("numhidden"=>12)]; z=1, lw=2, label="FacMARNN (nh: 12, fac: 10, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["FacMARNN"], fillalpha=0.4)

	plt
end

# ╔═╡ 533a195d-746b-4f28-b811-ff1af27d29c5
let 
	plt = plot(data_online_rnn_t12, [Dict()]; z=1, lw=2, label="RNN (nh: 20, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.4)
	
		plt = plot!(data_online_aarnn_t12, [Dict()]; z=1, lw=2, label="AARNN (nh: 20, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_marnn_t12, [Dict()]; z=1, lw=2, label="MARNN (nh: 15, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_facmarnn_t12, [Dict()]; z=1, lw=2, label="FacRNN (nh: 20, fac: 10, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSVE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells", grid=false, tickdir=:out, color=cell_colors["FacMARNN"], fillalpha=0.4, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15)

	plt
	
	savefig("../data/paper_plots/ringworld_online_learning_curves_with_fac_RNN_300K_steps_tau_12.pdf")
end

# ╔═╡ 6e15e9b9-1bae-43bd-ae4c-408fac574a14
let 
	plt = plot(data_online_gru_t12, [Dict()]; z=1, lw=2, label="GRU (nh: 12, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["GRU"], fillalpha=0.4)
	
		plt = plot!(data_online_aagru_t12, [Dict()]; z=1, lw=2, label="AAGRU (nh: 12, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AAGRU"], fillalpha=0.4)
	
		plt = plot!(data_online_magru_t12, [Dict()]; z=1, lw=2, label="MAGRU (nh: 9, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MAGRU"], fillalpha=0.4)
	
		plt = plot!(data_online_facmagru_t12, [Dict()]; z=1, lw=2, label="FacMAGRU (nh: 12, fac: 10, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSVE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells", grid=false, tickdir=:out, color=cell_colors["FacMAGRU"], fillalpha=0.4, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15)

	plt
	
	savefig("../data/paper_plots/ringworld_online_learning_curves_with_fac_GRU_300K_steps_tau_12.pdf")
end

# ╔═╡ 06cec7af-094f-44b4-8e9d-cfacefed52f6
let 
	plt = plot(data_online_rnn_t12, [Dict()]; z=1.97, lw=2, label="RNN (nh: 20, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.4)
	
		plt = plot!(data_online_aarnn_t12, [Dict()]; z=1.97, lw=2, label="AARNN (nh: 20, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_marnn_t12, [Dict()]; z=1.97, lw=2, label="MARNN (nh: 15, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.4)
	
	
	plt = plot!(data_online_gru, [Dict()]; z=1.97, lw=2, label="GRU (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["GRU"], fillalpha=0.4)
	
		plt = plot!(data_online_aagru, [Dict()]; z=1.97, lw=2, label="AAGRU (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AAGRU"], fillalpha=0.4)
	
		plt = plot!(data_online_magru, [Dict()]; z=1.97, lw=2, label="MAGRU (nh: 9, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN + GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MAGRU"], fillalpha=0.4)

	plt
end

# ╔═╡ 53c68570-f342-4051-9b2d-8f3db95d4427
let 
	plt = plot(data_online_rnn, [Dict("numhidden"=>20)]; z=1.97, lw=2, label="RNN (nh: 20, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.4)
	
		plt = plot!(data_online_aarnn, [Dict("numhidden"=>20)]; z=1.97, lw=2, label="AARNN (nh: 20, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_marnn, [Dict("numhidden"=>15)]; z=1.97, lw=2, label="MARNN (nh: 15, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.4)
	
	
	plt = plot!(data_online_gru, [Dict()]; z=1.97, lw=2, label="GRU (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["GRU"], fillalpha=0.4)
	
		plt = plot!(data_online_aagru, [Dict()]; z=1.97, lw=2, label="AAGRU (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AAGRU"], fillalpha=0.4)
	
		plt = plot!(data_online_magru, [Dict()]; z=1.97, lw=2, label="MAGRU (nh: 9, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN + GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MAGRU"], fillalpha=0.4)

	plt
end

# ╔═╡ 90c17f22-1141-4a06-b784-cd1c582af806
let 
	plt = plot(data_online_rnn, [Dict("numhidden"=>12)]; z=1.97, lw=2, label="RNN (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.4)
	
		plt = plot!(data_online_aarnn, [Dict("numhidden"=>12)]; z=1.97, lw=2, label="AARNN (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_marnn, [Dict("numhidden"=>9)]; z=1.97, lw=2, label="MARNN (nh: 9, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.4)
	
	
	plt = plot!(data_online_gru, [Dict()]; z=1.97, lw=2, label="GRU (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["GRU"], fillalpha=0.4)
	
		plt = plot!(data_online_aagru, [Dict()]; z=1.97, lw=2, label="AAGRU (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AAGRU"], fillalpha=0.4)
	
		plt = plot!(data_online_magru, [Dict()]; z=1.97, lw=2, label="MAGRU (nh: 9, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN + GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MAGRU"], fillalpha=0.4)

	plt
end

# ╔═╡ b297dea8-5af3-4005-b982-77a562255004
let 
	plt = plot(data_online_rnn, [Dict("numhidden"=>12)]; z=1.97, lw=3, label="RNN (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.25)
	
		plt = plot!(data_online_aarnn, [Dict("numhidden"=>12)]; z=1.97, lw=3, label="AARNN (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.25)
	
		plt = plot!(data_online_marnn, [Dict("numhidden"=>9)]; z=1.97, lw=3, label="MARNN (nh: 9, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.25)
	
	
	plt = plot!(data_online_gru, [Dict()]; z=1.97, lw=3, label="GRU (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["GRU"], fillalpha=0.25)
	
		plt = plot!(data_online_aagru, [Dict()]; z=1.97, lw=3, label="AAGRU (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AAGRU"], fillalpha=0.25)
	
		plt = plot!(data_online_magru, [Dict()]; z=1.97, lw=3, label="MAGRU (nh: 9, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN + GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MAGRU"], fillalpha=0.25)

	plt
end

# ╔═╡ 04d7d7ea-306e-479a-b79d-24500bdb068c
let 
	plt = plot(data_online_rnn, [Dict("numhidden"=>12)]; z=1, lw=3, label="RNN (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.25)
	
		plt = plot!(data_online_aarnn, [Dict("numhidden"=>12)]; z=1, lw=3, label="AARNN (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.25)
	
		plt = plot!(data_online_marnn, [Dict("numhidden"=>9)]; z=1, lw=3, label="MARNN (nh: 9, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.25)
	
	
	plt = plot!(data_online_gru, [Dict()]; z=1, lw=3, label="GRU (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["GRU"], fillalpha=0.25)
	
		plt = plot!(data_online_aagru, [Dict()]; z=1, lw=3, label="AAGRU (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AAGRU"], fillalpha=0.25)
	
		plt = plot!(data_online_magru, [Dict()]; z=1, lw=3, label="MAGRU (nh: 9, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN + GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MAGRU"], fillalpha=0.25)

	plt
end

# ╔═╡ 232c4d00-74ea-4f77-ad45-dbe3f16d6964
let 
	plt = plot(data_online_rnn, [Dict("numhidden"=>20)]; z=1, lw=3, label="RNN (nh: 20, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.25)
	
		plt = plot!(data_online_aarnn, [Dict("numhidden"=>20)]; z=1, lw=3, label="AARNN (nh: 20, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.25)
	
		plt = plot!(data_online_marnn, [Dict("numhidden"=>15)]; z=1, lw=3, label="MARNN (nh: 15, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.25)
	
	
	plt = plot!(data_online_gru, [Dict()]; z=1, lw=3, label="GRU (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["GRU"], fillalpha=0.25)
	
		plt = plot!(data_online_aagru, [Dict()]; z=1, lw=3, label="AAGRU (nh: 12, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AAGRU"], fillalpha=0.25)
	
		plt = plot!(data_online_magru, [Dict()]; z=1, lw=3, label="MAGRU (nh: 9, τ: 8)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN + GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MAGRU"], fillalpha=0.25)

	plt
end

# ╔═╡ 9b44540c-0e14-4cde-8060-dd9595b12c5b
let 
	plt = plot(data_online_rnn_t12, [Dict()]; z=1, lw=3, label="RNN (nh: 20, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.25)
	
		plt = plot!(data_online_aarnn_t12, [Dict()]; z=1, lw=3, label="AARNN (nh: 20, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.25)
	
		plt = plot!(data_online_marnn_t12, [Dict()]; z=1, lw=3, label="MARNN (nh: 15, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.25)
	
	
	plt = plot!(data_online_gru_t12, [Dict()]; z=1, lw=3, label="GRU (nh: 12, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["GRU"], fillalpha=0.25)
	
		plt = plot!(data_online_aagru_t12, [Dict()]; z=1, lw=3, label="AAGRU (nh: 12, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AAGRU"], fillalpha=0.25)
	
		plt = plot!(data_online_magru_t12, [Dict()]; z=1, lw=3, label="MAGRU (nh: 9, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN + GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MAGRU"], fillalpha=0.4)

	plt
end

# ╔═╡ a7e885ac-4b74-49ed-b395-74d99ccdbbba
let 
	args_list = [
		Dict("numhidden"=>9, "truncation"=>8, "cell"=>"MAGRU"),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"AAGRU"),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"GRU"),
		Dict("numhidden"=>15, "truncation"=>8, "cell"=>"RNN"),
		Dict("numhidden"=>20, "truncation"=>8, "cell"=>"AARNN"),
		Dict("numhidden"=>20, "truncation"=>8, "cell"=>"MARNN")
	]
	plt = plot()
	for args ∈ args_list
		plt = plot!(lc_data, args; z=1, lw=2, label="$(args["cell"]) (nh: $(args["numhidden"]), τ: $(args["truncation"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online, Size: 10", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"])
	end
	plt
end

# ╔═╡ c2f393d1-b4a4-42e5-b2c4-66585b03b63d
let 
	args_list = [
		Dict("numhidden"=>9, "truncation"=>8, "cell"=>"MARNN"),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"AARNN"),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"RNN")
	]
	plt = plot()
	for args ∈ args_list
		plt = plot!(lc_data_final, args; z=1, lw=2, label="$(args["cell"]) (nh: $(args["numhidden"]), τ: $(args["truncation"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online, Size: 10", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"])
	end
	plt
end

# ╔═╡ 5d68923a-125c-4fe8-8764-7cca881862f5
let 
	plt = plot(data_online_rnn_t12, [Dict()]; z=1, lw=2, label="RNN (nh: 20, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["RNN"], fillalpha=0.4)
	
		plt = plot!(data_online_aarnn_t12, [Dict()]; z=1, lw=2, label="AARNN (nh: 20, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AARNN"], fillalpha=0.4)
	
		plt = plot!(data_online_marnn_t12, [Dict()]; z=1, lw=2, label="MARNN (nh: 15, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["MARNN"], fillalpha=0.4)
	
	
	plt = plot!(data_online_gru_t12, [Dict()]; z=1, lw=2, label="GRU (nh: 12, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["GRU"], fillalpha=0.4)
	
		plt = plot!(data_online_aagru_t12, [Dict()]; z=1, lw=2, label="AAGRU (nh: 12, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["AAGRU"], fillalpha=0.4)
	
		plt = plot!(data_online_magru_t12, [Dict()]; z=1, lw=2, label="MAGRU (nh: 9, τ: 12)", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online", grid=false, tickdir=:out, color=cell_colors["MAGRU"], fillalpha=0.4, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15)

	plt
	
	savefig("../data/paper_plots/ringworld_online_learning_curves_300K_steps_tau_12.pdf")
end

# ╔═╡ f9149118-1f58-4b32-a731-e9b5e4a3079a
let 
	args_list_fac = [
		Dict("numhidden_factors"=>[20, 10])
	]
	plt = plot()
	for args ∈ args_list_fac
		plt = plot!(data_rpu_online_facmarnn, args; z=1.97, lw=3, label="FacMARNN (nh_fac: $(args["numhidden_factors"])", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online, Size: 10", grid=false, tickdir=:out, color=cell_colors["FacMARNN"], fillalpha=0.3)
	end
	
	args_list = [
		Dict("numhidden"=>15, "truncation"=>8, "cell"=>"MARNN"),
		Dict("numhidden"=>20, "truncation"=>8, "cell"=>"AARNN"),
		Dict("numhidden"=>20, "truncation"=>8, "cell"=>"RNN")
	]

	for args ∈ args_list
		plt = plot!(data_rpu_online, args; z=1.97, lw=3, label="$(args["cell"]) (nh: $(args["numhidden"]), τ: $(args["truncation"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online RNN Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"], fillalpha=0.3)
	end
	plt
end

# ╔═╡ 2b2f2f39-b861-40af-a964-d22b6f2f85ea
let
	args_list_2 = [
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"RNN")]
	boxplot(data_dist_online, args_list_2;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_2, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_dist_online, args_list_2;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_2, "cell")), 1, :))
	
	
	args_list_rnn = [
		Dict("numhidden_factors"=>[15, 15])
	]
	boxplot!(data_dist_online_facmarnn, args_list_rnn; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMARNN"] cell_colors["FacMARNN"] cell_colors["FacMARNN"]])
	dotplot!(data_dist_online_facmarnn, args_list_rnn; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMARNN"] cell_colors["FacMARNN"] cell_colors["FacMARNN"]])
	
	args_list_3 = [
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AARNN"),
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"MARNN")]
	boxplot!(data_dist_online, args_list_3;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_3, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Ringworld Online AUC, Steps: 300k")
	dotplot!(data_dist_online, args_list_3;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_3, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ 5a87d233-ebc9-4158-a23e-b2b0d5cef768
let 
	plt = plot()
	
	args_list = [
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"RNN")
	]

	for args ∈ args_list
		plt = plot!(data_rpu_online_1M, args; z=1.97, lw=2, label="$(args["cell"])", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online 1M Steps, Size: 10, eta: 0.003725", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"], fillalpha=0.6)
	end
	plt
end

# ╔═╡ 0a32b684-9569-466e-bb6b-7ea86657de65
let 
	plt = plot()
	
	args_list = [
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"RNN")
	]

	for args ∈ args_list
		plt = plot!(data_rpu_online_1M_slr, args; z=1.97, lw=2, label="$(args["cell"])", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online 1M Steps, Size: 10, eta:0.00005421", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"], fillalpha=0.6)
	end
	plt
end

# ╔═╡ a4aa5d99-8764-4782-a241-9ff0dcbf3615
let 
	plt = plot()
	
	args_list = [
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"RNN")
	]

	for args ∈ args_list
		plt = plot!(data_rpu_online_1M_mlr, args; z=1.97, lw=2, label="$(args["cell"])", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online 1M Steps, Size: 10", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"], fillalpha=0.6)
	end
	plt
end

# ╔═╡ 53150fb9-7330-4435-97fd-8b47d10255d0
let
	params = Dict("cell"=>"MARNN")
	idx = findfirst(data_rpu_online_1M.data) do ld
		all([ld.line_params[idx] == params[idx] for idx in keys(params)])
	end
	plot(data_rpu_online_1M[idx].data, legend=false, ylims=(0.0, 0.4), color=cell_colors[params["cell"]])
	params = Dict("cell"=>"AARNN")
	idx = findfirst(data_rpu_online_1M.data) do ld
		all([ld.line_params[idx] == params[idx] for idx in keys(params)])
	end
	plot!(data_rpu_online_1M[idx].data, legend=false, ylims=(0.0, 0.4), color=cell_colors[params["cell"]])
	
	params = Dict("cell"=>"FacMARNN")
	idx = findfirst(data_rpu_online_1M.data) do ld
		all([ld.line_params[idx] == params[idx] for idx in keys(params)])
	end
	plot!(data_rpu_online_1M[idx].data, legend=false, ylims=(0.0, 0.4), color=cell_colors[params["cell"]])
end

# ╔═╡ e3b20ee0-4c21-4c76-b908-21e8476ce79b
let 
	plt = plot()
	
	args_list = [
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"GRU")
	]

	for args ∈ args_list
		plt = plot!(data_rpu_online_1M, args; z=1.97, lw=2, label="$(args["cell"])", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online 1M Steps, Size: 10, eta: eta: 0.003725", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"], fillalpha=0.6)
	end
	plt
end

# ╔═╡ d001e998-6f14-4359-9dc6-186dbc95ad84
let 
	plt = plot()
	
	args_list = [
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"GRU")
	]

	for args ∈ args_list
		plt = plot!(data_rpu_online_1M_slr, args; z=1.97, lw=2, label="$(args["cell"])", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online 1M Steps, Size: 10, eta: 0.00005421", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"], fillalpha=0.6)
	end
	plt
end

# ╔═╡ 808769bb-e655-4f7e-941a-a76724206af2
let 
	plt = plot()
	
	args_list = [
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"GRU")
	]

	for args ∈ args_list
		plt = plot!(data_rpu_online_1M_mlr, args; z=1.97, lw=2, label="$(args["cell"])", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online 1M Steps, Size: 10, eta:0.0009095", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"], fillalpha=0.6)
	end
	plt
end

# ╔═╡ 1764be4f-79a0-4824-915f-ad4eddcfd193
let
	params = Dict("cell"=>"MAGRU")
	idx = findfirst(data_rpu_online_1M.data) do ld
		all([ld.line_params[idx] == params[idx] for idx in keys(params)])
	end
	plot(data_rpu_online_1M[idx].data, legend=false, ylims=(0.0, 0.4), color=cell_colors[params["cell"]])
	params = Dict("cell"=>"AAGRU")
	idx = findfirst(data_rpu_online_1M.data) do ld
		all([ld.line_params[idx] == params[idx] for idx in keys(params)])
	end
	plot!(data_rpu_online_1M[idx].data, legend=false, ylims=(0.0, 0.4), color=cell_colors[params["cell"]])
	
	params = Dict("cell"=>"FacMAGRU")
	idx = findfirst(data_rpu_online_1M.data) do ld
		all([ld.line_params[idx] == params[idx] for idx in keys(params)])
	end
	plot!(data_rpu_online_1M[idx].data, legend=false, ylims=(0.0, 0.4), color=cell_colors[params["cell"]])
end

# ╔═╡ 50e88e24-daf7-4891-b09a-a8afde96bd06
let 
	args_list = [
		Dict("numhidden"=>9, "truncation"=>8, "cell"=>"MAGRU"),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"AAGRU"),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"GRU")
	]
	plt = plot()
	for args ∈ args_list
		plt = plot!(data_rpu_online, args; z=1.97, lw=2, label="$(args["cell"]) (nh: $(args["numhidden"]), τ: $(args["truncation"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online, Size: 10", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"])
	end
	plt
end

# ╔═╡ 574f7bed-477e-4f06-9af0-a1efb22a0501
let 
	args_list_fac = [
		Dict("numhidden_factors"=>[12, 10])
	]
	plt = plot()
	for args ∈ args_list_fac
		plt = plot!(data_rpu_online_facmagru, args; z=1.97, lw=3, label="FacMAGRU (nh_fac: $(args["numhidden_factors"])", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["FacMAGRU"], fillalpha=0.3)
	end
	
	args_list = [
		Dict("numhidden"=>9, "truncation"=>8, "cell"=>"MAGRU"),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"AAGRU"),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"GRU")
	]

	for args ∈ args_list
		plt = plot!(data_rpu_online, args; z=1.97, lw=3, label="$(args["cell"]) (nh: $(args["numhidden"]), τ: $(args["truncation"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online GRU Cells, Size: 10", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"], fillalpha=0.3)
	end
	plt
end

# ╔═╡ 586ba32f-79b6-4c2e-80af-ef2c1224fd31
let 
	plt = nothing
	lstyle = [:solid, :dash, :dot, :solid, :dash, :dot]
	mshape = [:circle, :rect, :star5, :diamond, :hexagon, :utriangle]
	trunc = dd_dir_all["truncation"]
	for (i, nh) ∈ enumerate(nh_dir)
		nh_ = parse(Int, nh)
		plt = plot_sensitivity_from_data_with_params!(plt, sensitivity_data, (nh_, cell), trunc; label="$(nh)", palette=color_scheme, color=cell_colors[cell], legend=:topright, ylabel="RMSVE (final 50k steps)", xlabel="Truncation", ylim=(0, 0.40), title="Cell: $(cell)", markershape=mshape[i], markersize=5, linestyle=:solid, grid=false, tickdir=:out, legendtitlefontsize=10, legendfonthalign=:center, lw=2, z=1.97, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, legendtitle="nh")
	end
	plt
	
end

# ╔═╡ 29de4184-0745-4407-8bc3-5bbe1e6869f5
cell_colors_ = Dict(
	"RNN" => color_scheme_[3],
	"AARNN" => color_scheme_[end],
	"MARNN" => color_scheme_[5],
	"FacMARNN" => color_scheme_[1],
	"GRU" => color_scheme_[4],
	"AAGRU" => color_scheme_[2],
	"MAGRU" => color_scheme_[6],
	"FacMAGRU" => color_scheme_[end-2])

# ╔═╡ 17d3c458-ec9f-4dd3-afff-3b187a437c42
let 
	plt = plot(data_online_rnn_t12, [Dict()]; z=1, lw=2, palette=RPU.custom_colorant,legend=:topright, ylim=(0, 0.40), grid=false, tickdir=:out, color=cell_colors_["RNN"], label=nothing)
	
	plt = plot!(data_online_aarnn_t12, [Dict()]; z=1, lw=2, palette=RPU.custom_colorant, ylim=(0, 0.40), grid=false, tickdir=:out, color=cell_colors_["AARNN"], label=nothing)
	
	plt = plot!(data_online_marnn_t12, [Dict()]; z=1, lw=2, palette=RPU.custom_colorant, ylim=(0, 0.40), grid=false, tickdir=:out, color=cell_colors_["MARNN"], label=nothing)
	
	
	plt = plot!(data_online_gru_t12, [Dict()]; z=1, lw=2, 
		palette=RPU.custom_colorant, ylim=(0, 0.40), grid=false, tickdir=:out, color=cell_colors_["GRU"], label=nothing)
	
	plt = plot!(data_online_aagru_t12, [Dict()]; z=1, lw=2, palette=RPU.custom_colorant, ylim=(0, 0.40), grid=false, tickdir=:out, color=cell_colors_["AAGRU"], label=nothing)
	
	plt = plot!(data_online_magru_t12, [Dict()]; z=1, lw=2, palette=RPU.custom_colorant, ylim=(0, 0.35), grid=false, tickdir=:out, color=cell_colors_["MAGRU"], label=nothing, legend=nothing, tickfontsize=12)

	plt
	savefig("../data/paper_plots/ringworld_online_learning_curves_300K_steps_tau_12.pdf")
end

# ╔═╡ Cell order:
# ╠═f7f500a8-a1e9-11eb-009b-d7afdcade891
# ╠═e0d51e67-63dc-45ea-9092-9965f97660b3
# ╠═fbdbb061-f620-4704-ba0f-9e4d00ddcc8f
# ╠═0c746c1e-ea39-4415-a1b1-d7124b886f98
# ╠═55b8266e-f72e-4cdf-a987-5321f1e5d953
# ╟─94700b85-0982-47e1-9e08-8380dd585cac
# ╠═e78b5abb-3ee3-48fd-901a-40d49f48f664
# ╟─92c2600e-58b4-4ea3-8272-3fe38c0422d1
# ╟─24ccf89f-ab20-447e-9d6f-633380ee8c20
# ╠═240dc563-fd04-4c07-85ac-4e54ad016374
# ╟─265db42d-1b27-463d-b0a1-cfc62748022a
# ╟─62d657ae-bdd9-454e-8d4f-39e7d044d6dd
# ╟─dde0ed32-3b8a-40ac-ae06-16dbb7c62856
# ╠═24f92c1e-3b6c-4d0e-bfd1-6f45bc1a72c0
# ╠═47a4712f-8833-4916-b240-96f17ee6b504
# ╠═4fd60991-9844-4b1b-a756-51312dbbe33b
# ╠═98ed8227-78e6-410e-8937-3c2c18b305ef
# ╠═98858fea-5e35-4e16-bf8f-df95e95ca6db
# ╠═c5c76526-8e14-469d-ae96-bf0835864f55
# ╠═8ae03ea3-fecd-413b-b612-9201e4cab552
# ╠═59838b82-2402-4c76-811f-2d8896f10e2f
# ╠═8d9a77ff-974f-4cb1-a01a-e03b8951fb70
# ╠═3a5ae1d2-394b-410b-b3bf-0b2d90198db3
# ╠═d6098ab5-4090-4568-92e7-83e659ee17eb
# ╠═6afe6d05-c4d3-4875-b240-d9fd7ab0759e
# ╠═8bee01d3-ee5a-4bc5-beb3-00095d487a78
# ╠═736b6799-2dd7-473e-934a-2bf3c2fb6fa0
# ╠═85d2153a-8cb1-40f8-aa3d-3d3faa1fa10e
# ╠═1377c495-47e0-41c1-9ca6-eebc1a5a9dde
# ╠═3022d29f-e500-4669-ba68-b516f0c00244
# ╠═cf697ad0-ee10-4511-bfce-4459ff0540b9
# ╠═7441f85a-d8ba-47ab-983c-f32dc1761b41
# ╠═1dd607c8-b816-4995-bb63-7e6fdef68bd6
# ╠═5d5aad1f-4864-474f-ba31-34e007daceec
# ╠═1d636cfc-d163-495d-adf3-049ffee74c3b
# ╠═15a9152e-ad82-4fe1-945c-4e87c5a434ef
# ╠═67666b0f-fabd-4a8a-aca8-365c66526462
# ╠═72d108e9-5c72-4784-b31f-fd0cc34f5ff2
# ╟─032b80e1-be25-42e7-9bf6-df5219f34391
# ╟─55e74d54-a303-4285-8f28-e8b7947f9555
# ╠═427d801e-f2a1-4417-b505-14cd90968f67
# ╠═379f4f22-c716-4a24-b75f-3eb127b94334
# ╠═f6f5a8ac-e1a3-41e2-a38c-5f31d2c60505
# ╠═1e0f4ca4-0ade-433d-a4dd-13537e5fc6b4
# ╠═6c795e94-7196-4832-ab0f-aba17ea551f2
# ╠═bd790756-66ec-4b82-af61-10eaf209e1ff
# ╠═093eec7f-b57d-4e6f-a971-f1b60cde0220
# ╟─39bb5c7b-3414-49c9-824c-ca50dee45cbb
# ╟─3ee90492-bc4e-401a-94ae-7388e259f6f9
# ╟─46580846-3fa3-4311-a554-40a2c5ec3ccf
# ╠═346dd791-4b42-4f7d-a5f6-e6e1032cade9
# ╠═84edfef4-f525-452b-8f5d-d44e17b93a3b
# ╠═7cf29ef3-b311-44e8-9036-de788e85cfcc
# ╠═533a195d-746b-4f28-b811-ff1af27d29c5
# ╠═6e15e9b9-1bae-43bd-ae4c-408fac574a14
# ╟─06cec7af-094f-44b4-8e9d-cfacefed52f6
# ╟─53c68570-f342-4051-9b2d-8f3db95d4427
# ╟─90c17f22-1141-4a06-b784-cd1c582af806
# ╟─b297dea8-5af3-4005-b982-77a562255004
# ╟─04d7d7ea-306e-479a-b79d-24500bdb068c
# ╟─232c4d00-74ea-4f77-ad45-dbe3f16d6964
# ╠═9b44540c-0e14-4cde-8060-dd9595b12c5b
# ╠═242424b3-7135-4209-afe4-82429a858b66
# ╠═19ea46d1-40f2-4e4f-8c97-72db8fb86482
# ╠═a7e885ac-4b74-49ed-b395-74d99ccdbbba
# ╠═c2f393d1-b4a4-42e5-b2c4-66585b03b63d
# ╠═5d68923a-125c-4fe8-8764-7cca881862f5
# ╠═f9149118-1f58-4b32-a731-e9b5e4a3079a
# ╠═c8c20326-d661-438f-bd23-8982f1a6329e
# ╠═69df05a1-dcd2-498a-8fc4-da95b62b5add
# ╠═2b2f2f39-b861-40af-a964-d22b6f2f85ea
# ╠═5a87d233-ebc9-4158-a23e-b2b0d5cef768
# ╠═0a32b684-9569-466e-bb6b-7ea86657de65
# ╠═a4aa5d99-8764-4782-a241-9ff0dcbf3615
# ╟─53150fb9-7330-4435-97fd-8b47d10255d0
# ╠═e3b20ee0-4c21-4c76-b908-21e8476ce79b
# ╠═d001e998-6f14-4359-9dc6-186dbc95ad84
# ╠═808769bb-e655-4f7e-941a-a76724206af2
# ╠═1764be4f-79a0-4824-915f-ad4eddcfd193
# ╠═22db7508-72f0-40cc-9d8e-b7843dba4d3e
# ╠═50e88e24-daf7-4891-b09a-a8afde96bd06
# ╠═574f7bed-477e-4f06-9af0-a1efb22a0501
# ╠═8dee9c88-4a33-4fd5-9095-f4fccae6bbf7
# ╠═1d888ffa-0d14-4e71-9cb4-607836adc405
# ╠═b2952b4c-4d2e-4c65-9a0c-5b067053a89d
# ╠═63da8a2e-01e9-4ae6-aef7-1c13f743fdf0
# ╠═b0ca03a1-20f4-4ba2-b0fb-8a48fe9696b2
# ╠═028b3b1c-63cd-4c5f-a472-90f058df2a00
# ╟─0ab2174b-9e6d-431a-a9cd-10d2518a759b
# ╠═da215349-0686-405a-a7f7-8012a86382ab
# ╠═e510936c-2b38-4669-99ab-29ef6ab3ce16
# ╠═cbe39710-5202-411f-bd42-368de800e5f1
# ╠═be347bbc-1549-4dcd-add3-1ee3e91a76f7
# ╠═7002dd0e-01b8-4598-929c-dbff02fb9143
# ╠═586ba32f-79b6-4c2e-80af-ef2c1224fd31
# ╠═92e35913-4eec-47cc-ac8e-48ec0a22533e
# ╠═d864b0dd-90a3-4e91-9022-8146568575cc
# ╠═d7ae12f9-ec9d-4d07-83ab-97205ffab69b
# ╠═1a02cb1b-2495-482d-9588-2a93f3abac47
# ╠═2ae940a2-9624-4426-8bba-1e85b51db18e
# ╠═8c60f5fd-0c62-4d1a-b7d8-91b02a14b97d
# ╠═a83ddc94-9fa8-4411-a78a-fef19444096e
# ╠═29de4184-0745-4407-8bc3-5bbe1e6869f5
# ╠═64919972-1b56-4ba5-b9e5-c7aa247aed85
# ╠═17d3c458-ec9f-4dd3-afff-3b187a437c42
