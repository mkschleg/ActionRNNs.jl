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

# ╔═╡ e78b5abb-3ee3-48fd-901a-40d49f48f664
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[1],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[end-1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[end],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])


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
			push!(d_s, z * (sqrt(var(d[i].data))/length(d[i].data)))
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

# ╔═╡ b3a7b61f-e91d-492e-ad1b-bc358985e4e2
ic_dir_rpu_online, dd_dir_rpu_online = RPU.load_data("../local_data/final_ringworld_online_rmsprop_10/")

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

# ╔═╡ cf697ad0-ee10-4511-bfce-4459ff0540b9
data_rpu_online = RPU.get_line_data_for(
	ic_dir_rpu_online,
	["numhidden", "truncation", "cell"],
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
ic_dir, dd_dir = ic_dir_rpu_er, dd_dir_rpu_er

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

# ╔═╡ 379f4f22-c716-4a24-b75f-3eb127b94334
#savefig("../data/ringworld_lc_plots/final_ringworld_er_lc_plot_same_params.pdf")

# ╔═╡ f6f5a8ac-e1a3-41e2-a38c-5f31d2c60505
let 
	#args_list = [
	#	Dict("numhidden"=>15, "truncation"=>12, "cell"=>"MARNN"),
	#	Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AARNN"),
	#	Dict("numhidden"=>20, "truncation"=>12, "cell"=>"RNN")
	#]
	args_list = [
		Dict("numhidden"=>15, "truncation"=>8, "cell"=>"MARNN"),
		Dict("numhidden"=>20, "truncation"=>8, "cell"=>"AARNN"),
		Dict("numhidden"=>20, "truncation"=>8, "cell"=>"RNN")
	]
	plt = plot()
	for args ∈ args_list
		plt = plot!(data_rpu_online, args; z=1.97, lw=2, label="$(args["cell"]) (nh: $(args["numhidden"]), τ: $(args["truncation"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online, Size: 10", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"])
	end
	plt
end

# ╔═╡ 1e0f4ca4-0ade-433d-a4dd-13537e5fc6b4
data = data_rpu_online

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

# ╔═╡ bd790756-66ec-4b82-af61-10eaf209e1ff
data_rpu_online

# ╔═╡ 84edfef4-f525-452b-8f5d-d44e17b93a3b
let 
	args_list_fac = [
		Dict("numhidden_factors"=>[15, 15]),
		Dict("numhidden_factors"=>[20, 10])
	]
	plt = plot()
	for args ∈ args_list_fac
		plt = plot!(data_rpu_online_facmarnn, args; z=1.97, lw=2, label="FacMARNN (nh_fac: $(args["numhidden_factors"])", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online, Size: 10", grid=false, tickdir=:out, color=cell_colors["FacMARNN"], fillalpha=0.4)
	end
	
	args_list = [
		Dict("numhidden"=>9, "truncation"=>8, "cell"=>"MARNN"),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"AARNN"),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"RNN")
	]

	for args ∈ args_list
		plt = plot!(data_rpu_online, args; z=1.97, lw=2, label="$(args["cell"]) (nh: $(args["numhidden"]), τ: $(args["truncation"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online, Size: 10", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"], fillalpha=0.4)
	end
	plt
end

# ╔═╡ 22db7508-72f0-40cc-9d8e-b7843dba4d3e
#savefig("../data/ringworld_lc_plots/final_ringworld_online_lc_plot_same_params.pdf")

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
		plt = plot!(data_rpu_online_facmagru, args; z=1.97, lw=2, label="FacMAGRU (nh_fac: $(args["numhidden_factors"])", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online, Size: 10", grid=false, tickdir=:out, color=cell_colors["FacMAGRU"], fillalpha=0.4)
	end
	
	args_list = [
		Dict("numhidden"=>9, "truncation"=>8, "cell"=>"MAGRU"),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"AAGRU"),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"GRU")
	]

	for args ∈ args_list
		plt = plot!(data_rpu_online, args; z=1.97, lw=2, label="$(args["cell"]) (nh: $(args["numhidden"]), τ: $(args["truncation"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="RMSE", xlabel="Steps (Thousands)", ylim=(0, 0.40), title="Ringworld Online, Size: 10", grid=false, tickdir=:out, color=cell_colors["$(args["cell"])"], fillalpha=0.4)
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
		plt = plot_sensitivity_from_data_with_params!(plt, sensitivity_data, (nh_, cell), trunc; label="$(nh)", palette=color_scheme, color=nh_colors[nh_], legend=:topright, ylabel="RMSE (Final 50k steps)", xlabel="Truncation", ylim=(0, 0.35), title="Cell: $(cell), Envsize: 10", markershape=mshape[i], markersize=5, linestyle=:solid, grid=false, tickdir=:out, legendtitlefontsize=10, legendfontsize=8, legendfonthalign=:center, lw=2, z=1)
	end
	plt
end

# ╔═╡ 18b53418-b7af-48cf-99d1-08abdf9387ad
size(sensitivity_data[1].data, 1)

# ╔═╡ c0e0a43a-4bf7-416d-b412-c1ee684e1e0e
# savefig("../data/sensitivity_plots/final_ringworld_online_sensitivity_plot_$(cell).png")

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

# ╔═╡ e4f917eb-be8b-4164-a51c-5180b2e4d52c
begin
	args_list_hc = [
		Dict("numhidden"=>9, "truncation"=>8, "cell"=>"RNN", "eta"=>0.00005421),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"AARNN", "eta"=>0.003725),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"MARNN", "eta"=>0.003725), 
		Dict("numhidden"=>12, "factors"=>10, "truncation"=>8, "cell"=>"FacMARNN", "eta"=>0.003725),
		Dict("numhidden"=>9, "truncation"=>8, "cell"=>"GRU", "eta"=>0.00005421), 
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"AAGRU", "eta"=>0.003725),
		Dict("numhidden"=>12, "truncation"=>8, "cell"=>"MAGRU", "eta"=>0.003725), 
		Dict("numhidden"=>12, "factors"=>10, "truncation"=>8, "cell"=>"FacMAGRU", "eta"=>0.003725)
		]
end

# ╔═╡ 8c60f5fd-0c62-4d1a-b7d8-91b02a14b97d
f=jldopen("../final_runs/ringworld_online_10_1M.jld2", "r")

# ╔═╡ a83ddc94-9fa8-4411-a78a-fef19444096e
data_ = read(f, keys(f)[1])

# ╔═╡ b7e8ab78-9867-4b5b-9d7c-adc5af3c9ab2
FileIO.save("../final_runs/ringworld_online_10_1M.jld2", "args", args_list_hc)

# ╔═╡ Cell order:
# ╠═f7f500a8-a1e9-11eb-009b-d7afdcade891
# ╠═e0d51e67-63dc-45ea-9092-9965f97660b3
# ╠═fbdbb061-f620-4704-ba0f-9e4d00ddcc8f
# ╟─0c746c1e-ea39-4415-a1b1-d7124b886f98
# ╟─94700b85-0982-47e1-9e08-8380dd585cac
# ╟─e78b5abb-3ee3-48fd-901a-40d49f48f664
# ╟─92c2600e-58b4-4ea3-8272-3fe38c0422d1
# ╟─24ccf89f-ab20-447e-9d6f-633380ee8c20
# ╠═240dc563-fd04-4c07-85ac-4e54ad016374
# ╟─265db42d-1b27-463d-b0a1-cfc62748022a
# ╠═62d657ae-bdd9-454e-8d4f-39e7d044d6dd
# ╟─dde0ed32-3b8a-40ac-ae06-16dbb7c62856
# ╠═24f92c1e-3b6c-4d0e-bfd1-6f45bc1a72c0
# ╠═b3a7b61f-e91d-492e-ad1b-bc358985e4e2
# ╠═8d9a77ff-974f-4cb1-a01a-e03b8951fb70
# ╠═3a5ae1d2-394b-410b-b3bf-0b2d90198db3
# ╠═d6098ab5-4090-4568-92e7-83e659ee17eb
# ╠═cf697ad0-ee10-4511-bfce-4459ff0540b9
# ╠═1d636cfc-d163-495d-adf3-049ffee74c3b
# ╠═15a9152e-ad82-4fe1-945c-4e87c5a434ef
# ╠═67666b0f-fabd-4a8a-aca8-365c66526462
# ╠═72d108e9-5c72-4784-b31f-fd0cc34f5ff2
# ╠═032b80e1-be25-42e7-9bf6-df5219f34391
# ╠═55e74d54-a303-4285-8f28-e8b7947f9555
# ╠═427d801e-f2a1-4417-b505-14cd90968f67
# ╠═379f4f22-c716-4a24-b75f-3eb127b94334
# ╠═f6f5a8ac-e1a3-41e2-a38c-5f31d2c60505
# ╠═1e0f4ca4-0ade-433d-a4dd-13537e5fc6b4
# ╠═6c795e94-7196-4832-ab0f-aba17ea551f2
# ╠═bd790756-66ec-4b82-af61-10eaf209e1ff
# ╠═84edfef4-f525-452b-8f5d-d44e17b93a3b
# ╠═22db7508-72f0-40cc-9d8e-b7843dba4d3e
# ╠═50e88e24-daf7-4891-b09a-a8afde96bd06
# ╠═574f7bed-477e-4f06-9af0-a1efb22a0501
# ╠═8dee9c88-4a33-4fd5-9095-f4fccae6bbf7
# ╟─7002dd0e-01b8-4598-929c-dbff02fb9143
# ╟─586ba32f-79b6-4c2e-80af-ef2c1224fd31
# ╠═18b53418-b7af-48cf-99d1-08abdf9387ad
# ╠═c0e0a43a-4bf7-416d-b412-c1ee684e1e0e
# ╠═d7ae12f9-ec9d-4d07-83ab-97205ffab69b
# ╠═e4f917eb-be8b-4164-a51c-5180b2e4d52c
# ╠═2ae940a2-9624-4426-8bba-1e85b51db18e
# ╠═8c60f5fd-0c62-4d1a-b7d8-91b02a14b97d
# ╠═a83ddc94-9fa8-4411-a78a-fef19444096e
# ╠═b7e8ab78-9867-4b5b-9d7c-adc5af3c9ab2
