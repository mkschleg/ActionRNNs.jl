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
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 6a65de2a-2af7-4da5-9f14-d18623b3235b
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

# ╔═╡ 1886bf05-f4be-4160-b61c-edf186a7f3cb
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ fe50ffef-b691-47b5-acf8-8378fbf860a1
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[1],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[end-1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[end],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ 834b1cf3-5b22-4e0a-abe9-61e427e6cfda
# function plot_line_from_data_with_params!(
# 		plt, data_col::Vector{PU.LineData}, params; pkwargs...)
#     idx = findfirst(data_col) do (ld)
#         line_params = ld.line_params
# 	all([line_params[i] == params[i] for i ∈ 1:length(line_params)])
#     end
#     d = data_col[idx]
#     if plt isa Nothing
# 		plt = plot(d; pkwargs...)
#     else
# 		plot!(plt, d; pkwargs...)
#     end
#     plt
# end

# ╔═╡ eefd2ccc-ce9b-4cf8-991f-a58f2f932e99
ic_dir_os6_adam, dd_dir_os6_adam = RPU.load_data("../local_data/lunar_lander_er_adam_os6/")

# ╔═╡ 44f24315-a9a5-4474-8a26-0b45468fe660
ic_dir_os346_adam, dd_dir_os346_adam = RPU.load_data("../local_data/lunar_lander_er_adam_os346/")

# ╔═╡ 07e543f2-b87d-491f-982b-03e9beaf13cf
ic_dir_os346, dd_dir_os346 = RPU.load_data("../local_data/lunar_lander_er_rmsprop_os346/")

# ╔═╡ 7c90fced-9612-4fde-b59e-e48e3bb4a826
ic_dir_os346_magru, dd_dir_os346_magru = RPU.load_data("../local_data/lunar_lander_er_rmsprop_os346_magru/")

# ╔═╡ 3ff5a07f-7dea-4a26-8e22-47be6435cde9
ic_dir_os6, dd_dir_os6 = RPU.load_data("../local_data/lunar_lander_er_rmsprop_os6_150/")

# ╔═╡ 465a4284-ad67-40bc-911d-b2ea6706e4f9
ic_dir_os6_sc2, dd_dir_os6_sc2 = RPU.load_data("../local_data/lunar_lander_er_rmsprop_os6_sc2/")

# ╔═╡ 55bd927b-1b3e-49d0-8056-4c820ba60fbf
ic_dir_os6_sc2_magru, dd_dir_os6_sc2_magru = RPU.load_data("../local_data/lunar_lander_er_rmsprop_os6_sc2_magru/")

# ╔═╡ 01b9cd42-d64e-4988-a5b0-3fd5cddbf728
begin
	sub_ic_g = search(ic_dir_os6_sc2, Dict("cell"=>"GRU"))
	sub_ic_a = search(ic_dir_os6_sc2, Dict("cell"=>"AAGRU"))
	sub_ic_m = search(ic_dir_os6_sc2_magru, Dict("cell"=>"MAGRU"))
	sub_ic_g_ad = search(ic_dir_os6_adam, Dict("cell"=>"GRU"))
	sub_ic_a_ad = search(ic_dir_os6_adam, Dict("cell"=>"AAGRU"))
end

# ╔═╡ 4acf4ecb-9959-4602-b23e-0bd4ef0f4e87
begin
	sub_ic_sc2_g = search(ic_dir_os6_sc2, Dict("cell"=>"GRU"))
	sub_ic_sc2_a = search(ic_dir_os6_sc2, Dict("cell"=>"AAGRU"))
	sub_ic_sc2_m = search(ic_dir_os6_sc2_magru, Dict("cell"=>"MAGRU"))
end

# ╔═╡ 735aaf47-d8d1-46aa-b4f2-1358e4832551
begin 
	data_opt_sc2_g = RPU.get_line_data_for(
	sub_ic_sc2_g,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
	
	data_opt_sc2_a = RPU.get_line_data_for(
	sub_ic_sc2_a,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
	
	data_opt_sc2_m = RPU.get_line_data_for(
	sub_ic_sc2_m,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
end

# ╔═╡ 15693091-5dc9-4104-ae13-42f20046be19
diff(sub_ic_m)

# ╔═╡ 7eda4150-c8e1-4567-b7b9-2b41473eead9
ic_dir, dd_dir = ic_dir_os6_sc2, dd_dir_os6_sc2

# ╔═╡ eab5c7a0-8052-432d-977e-68b967baf5ca
ic_dir[1].parsed_args["steps"]

# ╔═╡ 1edc751d-0a75-459d-9ce0-4d4c8c9bc5cc
FileIO.load(joinpath(ic_dir[1].folder_str, "results.jld2"))

# ╔═╡ efede3d6-d147-47ee-9dbc-59bdc5272769
begin 
	data_opt_g = RPU.get_line_data_for(
	sub_ic_g,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
	
	data_opt_a = RPU.get_line_data_for(
	sub_ic_a,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
	
	data_opt_m = RPU.get_line_data_for(
	sub_ic_m,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
	
	data_opt_g_ad = RPU.get_line_data_for(
	sub_ic_g_ad,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
	
	data_opt_a_ad = RPU.get_line_data_for(
	sub_ic_a_ad,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
end

# ╔═╡ 8fd0bdb1-4db3-4a19-986a-e29addc1974a
begin 
	data_opt_g_mean = RPU.get_line_data_for(
	sub_ic_g,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
	
	data_opt_a_mean = RPU.get_line_data_for(
	sub_ic_a,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
	
	data_opt_m_mean = RPU.get_line_data_for(
	sub_ic_m,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
	
	data_opt_g_ad_mean = RPU.get_line_data_for(
	sub_ic_g_ad,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
	
	data_opt_a_ad_mean = RPU.get_line_data_for(
	sub_ic_a_ad,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
end

# ╔═╡ e51d91bc-c2d7-4d07-8319-803e28ff02d6
begin 
	data_opt_g_mean_el = RPU.get_line_data_for(
	sub_ic_g,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_extended_line(x, :total_rews, :total_steps, n=100))
	
	data_opt_a_mean_el = RPU.get_line_data_for(
	sub_ic_a,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_extended_line(x, :total_rews, :total_steps, n=100))
	
	data_opt_m_mean_el = RPU.get_line_data_for(
	sub_ic_m,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_extended_line(x, :total_rews, :total_steps, n=100))
	
	data_opt_g_ad_mean_el = RPU.get_line_data_for(
	sub_ic_g_ad,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_extended_line(x, :total_rews, :total_steps, n=100))
	
	data_opt_a_ad_mean_el = RPU.get_line_data_for(
	sub_ic_a_ad,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_extended_line(x, :total_rews, :total_steps, n=100))
end

# ╔═╡ ae784ee9-c937-4eda-89e8-6d1399aa7a36
begin
	data_opt_g_all = RPU.get_line_data_for(
	sub_ic_g,
	["truncation", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
	
	data_opt_a_all = RPU.get_line_data_for(
	sub_ic_a,
	["truncation", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
	
	data_opt_m_all = RPU.get_line_data_for(
	sub_ic_m,
	["truncation", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
	
	data_opt_g_ad_all = RPU.get_line_data_for(
	sub_ic_g_ad,
	["truncation", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
	
	data_opt_a_ad_all = RPU.get_line_data_for(
	sub_ic_a_ad,
	["truncation", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))
end

# ╔═╡ 032c75e5-ddd1-4b16-b524-af2d8f99d41e
data_opt = RPU.get_line_data_for(
	ic_dir,
	["cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))

# ╔═╡ e3ebad86-41aa-452c-a6b3-48c3b497c27d
md"""
Cell: $(@bind cells_dir_opt MultiSelect(dd_dir["cell"]))

"""

# ╔═╡ db53e161-68ac-421d-b433-4307da6e0442
let 
	plt = plot()
	for cell in cells_dir_opt
		plt = plot!(
		  data_opt,
		  Dict("cell"=>cell), label="cell: $(cell), opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	end
	plt
end

# ╔═╡ 503e447c-b7d0-4805-acde-0993179b9781
let 
	τ = parse(Int, τ_dir_opt)
	plt = plot()
	plt = plot!(
		  data_opt,
		  Dict("truncation"=>τ), label="cell: MAGRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	plt
end

# ╔═╡ 9aa2a048-78c6-4d07-b9c2-51815c75822d
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g,
		  Dict(), label="cell: GRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	
	plt = plot!(
		  data_opt_sc2_a,
		  Dict(), label="cell: AAGRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	
	plt = plot!(
		  data_opt_sc2_m,
		  Dict(), label="cell: MAGRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	plt
end

# ╔═╡ e2fecec9-0556-4517-b65d-8ddf81c82b0c
let
	# plt = nothing
	τ = parse(Int, τ_dir_opt)
	plt = plot()
	plt = plot!(
		  data_opt_g,
		  Dict("truncation"=>τ), label="cell: GRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	
	plt = plot!(
		  data_opt_a,
		  Dict("truncation"=>τ), label="cell: AAGRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	
	plt = plot!(
		  data_opt_m,
		  Dict("truncation"=>τ), label="cell: MAGRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	
	plt = plot!(
		  data_opt_g_ad,
		  Dict("truncation"=>τ), label="cell: GRU, opt: ADAM",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	
	plt = plot!(
		  data_opt_a_ad,
		  Dict("truncation"=>τ), label="cell: AAGRU, opt: ADAM",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", title="Lunar Lander MUE, τ:$(τ), $(ic_dir[1].parsed_args["steps"]) steps", lw=2)
	plt
end

# ╔═╡ 0bb67028-be79-45eb-9370-0f53bb6055ca
let
	# plt = nothing
	τ = parse(Int, τ_dir_opt)
	plt = plot()
	plt = plot!(
		  data_opt_g_mean,
		  Dict("truncation"=>τ), label="cell: GRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	
	plt = plot!(
		  data_opt_a_mean,
		  Dict("truncation"=>τ), label="cell: AAGRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	
	plt = plot!(
		  data_opt_m_mean,
		  Dict("truncation"=>τ), label="cell: MAGRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	
	plt = plot!(
		  data_opt_g_ad_mean,
		  Dict("truncation"=>τ), label="cell: GRU, opt: ADAM",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	
	plt = plot!(
		  data_opt_a_ad_mean,
		  Dict("truncation"=>τ), label="cell: AAGRU, opt: ADAM",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", title="Lunar Lander MEAN, τ:$(τ), $(ic_dir[1].parsed_args["steps"]) steps", lw=2)
	plt
end

# ╔═╡ 3c45d9a9-e4ab-4af2-abb3-2b6c5fd88e32
let
	# plt = nothing
	τ = parse(Int, τ_dir_opt)
	plt = plot()
	plt = plot!(
		  data_opt_g_mean_el,
		  Dict("truncation"=>τ), label="cell: GRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	
	plt = plot!(
		  data_opt_a_mean_el,
		  Dict("truncation"=>τ), label="cell: AAGRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	
	plt = plot!(
		  data_opt_m_mean_el,
		  Dict("truncation"=>τ), label="cell: MAGRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2, title="Lunar Lander MEAN, τ:$(τ), $(ic_dir[1].parsed_args["steps"]) steps")
	

end

# ╔═╡ 5e9aeddc-337c-484b-96df-cb4492111bb2
md"""
Truncation: $(@bind τ_dir_all Select(string.(dd_dir["truncation"])))
Eta: $(@bind eta_dir_all Select(string.(dd_dir["eta"])))

"""

# ╔═╡ 60d09c31-6555-476f-8c29-1c312a8e36e1
let
	# plt = nothing
	τ = parse(Int, τ_dir_all)
	eta = parse(Float64, eta_dir_all)
	
	plt = plot()
	plt = plot!(
		  data_opt_g_all,
		  Dict("truncation"=>τ, "eta"=>eta), label="cell: GRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", title="Lunar Lander, $(ic_dir[1].parsed_args["steps"]) steps", lw=2)
	
	plt = plot!(
		  data_opt_a_all,
		  Dict("truncation"=>τ, "eta"=>eta), label="cell: AAGRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", title="Lunar Lander, $(ic_dir[1].parsed_args["steps"]) steps", lw=2)
	
	plt = plot!(
		  data_opt_m_all,
		  Dict("truncation"=>τ, "eta"=>eta), label="cell: MAGRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", title="Lunar Lander, $(ic_dir[1].parsed_args["steps"]) steps", lw=2)
	plt
end

# ╔═╡ 3d8b0b2c-005e-4294-8817-e6bd075a0fb8
let
	# plt = nothing
	τ = parse(Int, τ_dir_opt)
	plt = plot()
	for cell ∈ cells_dir_opt
		plt = plot!(
			  data_opt,
			  Dict("truncation"=>τ, "cell"=>cell),
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", title="Lunar Lander, $(ic_dir[1].parsed_args["steps"]) steps")
	end
	plt
end

# ╔═╡ 55d1416b-d580-4112-827a-30504c21f397
data_all = RPU.get_line_data_for(
	ic_dir,
	["truncation", "eta", "cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 1000))

# ╔═╡ 467ed417-cf69-493d-b21c-3fc4d1fb9907
md"""
Truncation: $(@bind τ_dir Select(string.(dd_dir["truncation"])))
Eta: $(@bind eta_dir Select(string.(dd_dir["eta"])))
Cell: $(@bind cells_dir MultiSelect(dd_dir["cell"]))
"""

# ╔═╡ 2040d613-0ea8-48ce-937c-9180073812ea
let
	# plt = nothing
	τ = parse(Int, τ_dir)
	rs = parse(Int, rs_dir)
	eta = parse(Float64, eta_dir)
	plt = plot()
	for cell ∈ cells_dir
		plt = plot!(
			  data_all,
			  Dict("truncation"=>τ, "replay_size"=>rs, "eta"=>eta, "cell"=>cell),
			  palette=RPU.custom_colorant, legend=:topleft, ylim=(-500, 300))
	end
	plt
end

# ╔═╡ 7882977c-4802-4372-ab5b-1e2e45130fed
data_sens = RPU.get_line_data_for(
	ic_dir,
	["truncation", "cell", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))

# ╔═╡ 40cc0919-c56b-401b-83fa-0feb263c44b0
md"""
Truncation: $(@bind τ_dir_sens_ Select(string.(dd_dir["truncation"])))
Cell: $(@bind cells_dir_sens_ MultiSelect(dd_dir["cell"]))
"""

# ╔═╡ 5f70ab7f-f603-4c78-bddd-f60e125211ce
let
	# plt = nothing
	τ = parse(Int, τ_dir_sens_)
	plt = plot()
	for cell ∈ cells_dir_sens_
		plt = plot!(data_sens,
	 	  Dict("cell"=>cell, "truncation"=>τ);
	 	  sort_idx="eta",
		  z=1.97, lw=2, xaxis=:log,
		  palette=RPU.custom_colorant, label="cell: $(cell)",
		  color=cell_colors[cell], title="Truncation: $(τ)")
	end
	plt
end

# ╔═╡ Cell order:
# ╠═f7f500a8-a1e9-11eb-009b-d7afdcade891
# ╠═e0d51e67-63dc-45ea-9092-9965f97660b3
# ╠═6a65de2a-2af7-4da5-9f14-d18623b3235b
# ╟─0c746c1e-ea39-4415-a1b1-d7124b886f98
# ╠═1886bf05-f4be-4160-b61c-edf186a7f3cb
# ╟─fe50ffef-b691-47b5-acf8-8378fbf860a1
# ╟─834b1cf3-5b22-4e0a-abe9-61e427e6cfda
# ╠═eefd2ccc-ce9b-4cf8-991f-a58f2f932e99
# ╠═44f24315-a9a5-4474-8a26-0b45468fe660
# ╠═07e543f2-b87d-491f-982b-03e9beaf13cf
# ╠═7c90fced-9612-4fde-b59e-e48e3bb4a826
# ╠═3ff5a07f-7dea-4a26-8e22-47be6435cde9
# ╠═465a4284-ad67-40bc-911d-b2ea6706e4f9
# ╠═55bd927b-1b3e-49d0-8056-4c820ba60fbf
# ╠═eab5c7a0-8052-432d-977e-68b967baf5ca
# ╠═01b9cd42-d64e-4988-a5b0-3fd5cddbf728
# ╠═4acf4ecb-9959-4602-b23e-0bd4ef0f4e87
# ╠═735aaf47-d8d1-46aa-b4f2-1358e4832551
# ╠═15693091-5dc9-4104-ae13-42f20046be19
# ╠═7eda4150-c8e1-4567-b7b9-2b41473eead9
# ╠═1edc751d-0a75-459d-9ce0-4d4c8c9bc5cc
# ╠═efede3d6-d147-47ee-9dbc-59bdc5272769
# ╠═8fd0bdb1-4db3-4a19-986a-e29addc1974a
# ╠═e51d91bc-c2d7-4d07-8319-803e28ff02d6
# ╠═ae784ee9-c937-4eda-89e8-6d1399aa7a36
# ╠═032c75e5-ddd1-4b16-b524-af2d8f99d41e
# ╟─e3ebad86-41aa-452c-a6b3-48c3b497c27d
# ╠═db53e161-68ac-421d-b433-4307da6e0442
# ╠═503e447c-b7d0-4805-acde-0993179b9781
# ╠═9aa2a048-78c6-4d07-b9c2-51815c75822d
# ╠═e2fecec9-0556-4517-b65d-8ddf81c82b0c
# ╟─0bb67028-be79-45eb-9370-0f53bb6055ca
# ╟─3c45d9a9-e4ab-4af2-abb3-2b6c5fd88e32
# ╟─5e9aeddc-337c-484b-96df-cb4492111bb2
# ╠═60d09c31-6555-476f-8c29-1c312a8e36e1
# ╠═3d8b0b2c-005e-4294-8817-e6bd075a0fb8
# ╠═55d1416b-d580-4112-827a-30504c21f397
# ╠═467ed417-cf69-493d-b21c-3fc4d1fb9907
# ╠═2040d613-0ea8-48ce-937c-9180073812ea
# ╠═7882977c-4802-4372-ab5b-1e2e45130fed
# ╟─40cc0919-c56b-401b-83fa-0feb263c44b0
# ╠═5f70ab7f-f603-4c78-bddd-f60e125211ce
