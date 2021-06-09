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

# ╔═╡ c5cd1680-ccd5-4201-83ae-4be257507515
ic_dir_os6_sc2_4M, dd_dir_os6_sc2_4M = RPU.load_data("../local_data/lunar_lander_er_rmsprop_os6_sc2_4M/")

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

# ╔═╡ 00039914-c54f-46d8-9c54-51d8ddda236d
begin
	ic_dir_os6_sc2_gru_4M, dd_dir_os6_sc2_gru_4M = RPU.load_data("../local_data/LL_4M_8M/lunar_lander_er_rmsprop_os6_sc2_gru_4M/")
	ic_dir_os6_sc2_aagru_4M, dd_dir_os6_sc2_aagru_4M = RPU.load_data("../local_data/LL_4M_8M/lunar_lander_er_rmsprop_os6_sc2_aagru_4M/")
	ic_dir_os6_sc2_magru_4M, dd_dir_os6_sc2_magru_4M = RPU.load_data("../local_data/LL_4M_8M/lunar_lander_er_rmsprop_os6_sc2_magru_4M/")
	ic_dir_os6_sc2_facmagru152_4M, dd_dir_os6_sc2_facmagru152_4M = RPU.load_data("../local_data/LL_4M_8M/lunar_lander_er_rmsprop_os6_sc2_facmagru152_4M/")
	ic_dir_os6_sc2_facmagru100_4M, dd_dir_os6_sc2_facmagru100_4M = RPU.load_data("../local_data/LL_4M_8M/lunar_lander_er_rmsprop_os6_sc2_facmagru100_4M/")
	ic_dir_os6_sc2_facmagru64_4M, dd_dir_os6_sc2_facmagru64_4M = RPU.load_data("../local_data/LL_4M_8M/lunar_lander_er_rmsprop_os6_sc2_facmagru64_4M/")
end

# ╔═╡ 44030a7c-92ea-44ce-b221-7a0d6225d787
begin
	ic_dir_os6_sc2_gru_8M, dd_dir_os6_sc2_gru_8M = RPU.load_data("../local_data/LL_4M_8M/lunar_lander_er_rmsprop_os6_sc2_gru_8M/")
	ic_dir_os6_sc2_aagru_8M, dd_dir_os6_sc2_aagru_8M = RPU.load_data("../local_data/LL_4M_8M/lunar_lander_er_rmsprop_os6_sc2_aagru_8M/")
	ic_dir_os6_sc2_magru_8M, dd_dir_os6_sc2_magru_8M = RPU.load_data("../local_data/LL_4M_8M/lunar_lander_er_rmsprop_os6_sc2_magru_8M/")
	ic_dir_os6_sc2_facmagru152_8M, dd_dir_os6_sc2_facmagru152_8M = RPU.load_data("../local_data/LL_4M_8M/lunar_lander_er_rmsprop_os6_sc2_facmagru152_8M/")
	ic_dir_os6_sc2_facmagru100_8M, dd_dir_os6_sc2_facmagru100_8M = RPU.load_data("../local_data/LL_4M_8M/lunar_lander_er_rmsprop_os6_sc2_facmagru100_8M/")
	ic_dir_os6_sc2_facmagru64_8M, dd_dir_os6_sc2_facmagru64_8M = RPU.load_data("../local_data/LL_4M_8M/lunar_lander_er_rmsprop_os6_sc2_facmagru64_8M/")
end

# ╔═╡ 50734263-f616-4793-a580-7aa0f7a29223
begin
	ic_dir_os6_sc2_gru_4M_relu, dd_dir_os6_sc2_gru_4M_relu = RPU.load_data("../local_data/LL_relu/lunar_lander_er_relu_rmsprop_os6_sc2_gru_4M/")
	ic_dir_os6_sc2_aagru_4M_relu, dd_dir_os6_sc2_aagru_4M_relu = RPU.load_data("../local_data/LL_relu/lunar_lander_er_relu_rmsprop_os6_sc2_aagru_4M/")
	ic_dir_os6_sc2_magru_4M_relu, dd_dir_os6_sc2_magru_4M_relu = RPU.load_data("../local_data/LL_relu/lunar_lander_er_relu_rmsprop_os6_sc2_magru_4M/")
	ic_dir_os6_sc2_facmagru152_4M_relu, dd_dir_os6_sc2_facmagru152_4M_relu = RPU.load_data("../local_data/LL_relu/lunar_lander_er_relu_rmsprop_os6_sc2_facmagru152_4M/")
	ic_dir_os6_sc2_facmagru100_4M_relu, dd_dir_os6_sc2_facmagru100_4M_relu = RPU.load_data("../local_data/LL_relu/lunar_lander_er_relu_rmsprop_os6_sc2_facmagru100_4M/")
	ic_dir_os6_sc2_facmagru64_4M_relu, dd_dir_os6_sc2_facmagru64_4M_relu = RPU.load_data("../local_data/LL_relu/lunar_lander_er_relu_rmsprop_os6_sc2_facmagru64_4M/")
end

# ╔═╡ b5a22572-cc8f-4159-8f8f-246bd28d4053
begin
	ic_dir_os6_sc2_gru_4M_relu_final, dd_dir_os6_sc2_gru_4M_relu_final = RPU.load_data("../local_data/LL_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_gru_4M/")
	ic_dir_os6_sc2_aagru_4M_relu_final, dd_dir_os6_sc2_aagru_4M_relu_final = RPU.load_data("../local_data/LL_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_aagru_4M/")
	ic_dir_os6_sc2_magru_4M_relu_final, dd_dir_os6_sc2_magru_4M_relu_final = RPU.load_data("../local_data/LL_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_magru_4M/")
	ic_dir_os6_sc2_facmagru152_4M_relu_final, dd_dir_os6_sc2_facmagru152_4M_relu_final = RPU.load_data("../local_data/LL_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru152_4M/")
	ic_dir_os6_sc2_facmagru100_4M_relu_final, dd_dir_os6_sc2_facmagru100_4M_relu_final = RPU.load_data("../local_data/LL_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru100_4M/")
	ic_dir_os6_sc2_facmagru64_4M_relu_final, dd_dir_os6_sc2_facmagru64_4M_relu_final = RPU.load_data("../local_data/LL_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru64_4M/")
end

# ╔═╡ 4acf4ecb-9959-4602-b23e-0bd4ef0f4e87
begin
	sub_ic_sc2_g = search(ic_dir_os6_sc2_gru_4M, Dict("eta"=>0.00013877787807814446))
	sub_ic_sc2_a = search(ic_dir_os6_sc2_aagru_4M, Dict("eta"=>0.00013877787807814446))
	sub_ic_sc2_m = search(ic_dir_os6_sc2_magru_4M, Dict("eta"=>0.00013877787807814446))
	sub_ic_sc2_f152 = search(ic_dir_os6_sc2_facmagru152_4M, Dict("eta"=>0.00013877787807814446))
	sub_ic_sc2_f100 = search(ic_dir_os6_sc2_facmagru100_4M, Dict("eta"=>0.00013877787807814446))
	sub_ic_sc2_f64 = search(ic_dir_os6_sc2_facmagru64_4M, Dict("eta"=>0.00013877787807814446))
end

# ╔═╡ 735aaf47-d8d1-46aa-b4f2-1358e4832551
begin 
	data_opt_sc2_g_4M = RPU.get_line_data_for(
	sub_ic_sc2_g,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_a_4M = RPU.get_line_data_for(
	sub_ic_sc2_a,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_m_4M = RPU.get_line_data_for(
	sub_ic_sc2_m,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_f152_4M = RPU.get_line_data_for(
	sub_ic_sc2_f152,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_f100_4M = RPU.get_line_data_for(
	sub_ic_sc2_f100,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_f64_4M = RPU.get_line_data_for(
	sub_ic_sc2_f64,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
end

# ╔═╡ b2a18bf1-f75a-42b3-a207-78bc4d9e043f
begin 
	data_opt_sc2_g_8M = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_8M,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_a_8M = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_8M,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_m_8M = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_8M,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_f152_8M = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru152_8M,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_f100_8M = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru100_8M,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_f64_8M = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru64_8M,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
end

# ╔═╡ ae4cfd3a-084b-48f2-8a19-421853a189c9
begin 
	data_opt_sc2_g_4M_relu = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_4M_relu,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_a_4M_relu = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_relu,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_m_4M_relu = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_relu,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_f152_4M_relu = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru152_4M_relu,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_f100_4M_relu = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru100_4M_relu,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_f64_4M_relu = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru64_4M_relu,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
end

# ╔═╡ 77ded675-c775-4e17-b51f-9084f3ccbb88
begin 
	data_opt_sc2_g_4M_relu_final = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_a_4M_relu_final = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_m_4M_relu_final = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_relu_final,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_f152_4M_relu_final = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru152_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_f100_4M_relu_final = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru100_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_f64_4M_relu_final = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru64_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
end

# ╔═╡ c93e55ba-62a3-4f73-b552-7ba90cb4a8fb
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final,
		  Dict(), label="cell: GRU (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors["GRU"])
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors["MAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final,
		  Dict(), label="cell: FacMAGRU (nh: 152, fac: 170",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=2, z=1, color=cell_colors["FacMAGRU"], title="MUE, Steps: 4M", grid=false, tickdir=:out,)
	plt
end

# ╔═╡ 6e73dbc1-3615-4a78-8316-6c63aae62606
begin 
	data_opt_sc2_g_4M_relu_final_steps = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 2000))
	
	data_opt_sc2_a_4M_relu_final_steps = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 2000))
	
	data_opt_sc2_m_4M_relu_final_steps = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_relu_final,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 2000))
	
	data_opt_sc2_f152_4M_relu_final_steps = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru152_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 2000))
	
	data_opt_sc2_f100_4M_relu_final_steps = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru100_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 2000))
	
	data_opt_sc2_f64_4M_relu_final_steps = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru64_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 2000))
end

# ╔═╡ dcfd8f8c-ed8d-4b44-bacb-673ff9e05c09
function get_running_sum_line(ddict, key; n=nothing)
	if n isa Nothing
    	n = length(ddict["results"][key])
	end
    running(sum, ddict["results"][key], n)
end

# ╔═╡ 0757a4ec-aea3-4d0a-a5d7-b1ae4953db25
begin 
	data_opt_sc2_g_4M_relu_final_sum = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_running_sum_line(x, :total_rews, n=nothing))
	
	data_opt_sc2_a_4M_relu_final_sum = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_running_sum_line(x, :total_rews, n=nothing))
	
	data_opt_sc2_m_4M_relu_final_sum = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_relu_final,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_running_sum_line(x, :total_rews, n=nothing))
	
	data_opt_sc2_f152_4M_relu_final_sum = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru152_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_running_sum_line(x, :total_rews, n=nothing))
	
	data_opt_sc2_f100_4M_relu_final_sum = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru100_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_running_sum_line(x, :total_rews, n=nothing))
	
	data_opt_sc2_f64_4M_relu_final_sum = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru64_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_running_sum_line(x, :total_rews, n=nothing))
end

# ╔═╡ c92d71f5-fbd6-42dc-9938-303745e7a2f2
begin 
	data_opt_sc2_g_4M_relu_final_el = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_extended_line(x, :total_rews, :total_steps, n=100))
	
	data_opt_sc2_a_4M_relu_final_el = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_extended_line(x, :total_rews, :total_steps, n=100))
	
	data_opt_sc2_m_4M_relu_final_el = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_relu_final,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_extended_line(x, :total_rews, :total_steps, n=100))
	
	data_opt_sc2_f152_4M_relu_final_el = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru152_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_extended_line(x, :total_rews, :total_steps, n=100))
	
	data_opt_sc2_f100_4M_relu_final_el = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru100_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_extended_line(x, :total_rews, :total_steps, n=100))
	
	data_opt_sc2_f64_4M_relu_final_el = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru64_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_extended_line(x, :total_rews, :total_steps, n=100))
end

# ╔═╡ d9857f89-ed14-4dfb-8ba8-ad003ef72dfb
function get_extended_line(ddict, key1, key2; n=0)
    ret = zeros(eltype(ddict["results"][key1]), sum(ddict["results"][key2]))
    cur_idx = 1
    for i in 1:length(ddict["results"][key1])
        ret[cur_idx:(cur_idx + ddict["results"][key2][i] - 1)] .= ddict["results"][key1][i]
        cur_idx += ddict["results"][key2][i]
    end

    if n == 0
        ret
    else
		return_arr = zeros(eltype(ddict["results"][key1]), Int(floor(length(ret)/n)))
		for i in 1:Int(floor(length(ret)/n))
			return_arr[i] = mean(ret[i*n - n + 1: i*n])
		end
		return_arr
    end
end

# ╔═╡ d64d298b-1d16-4f27-9d50-b78c70e77e58
function get_extended_line_ep(ddict, key1, key2; n=0)
    ret = zeros(eltype(ddict["results"][key1]), sum(ddict["results"][key2]))
    cur_idx = 1
    for i in 1:length(ddict["results"][key1])
        ret[cur_idx:(cur_idx + ddict["results"][key2][i] - 1)] .= ddict["results"][key1][i]./ddict["results"][key2][i]
        cur_idx += ddict["results"][key2][i]
    end

    if n == 0
        ret
    else
		return_arr = zeros(eltype(ddict["results"][key1]), Int(floor(length(ret)/n)))
		for i in 1:Int(floor(length(ret)/n))
			return_arr[i] = mean(ret[i*n - n + 1: i*n])
		end
		return_arr
    end
end

# ╔═╡ fcfe0f74-958a-4bd7-9163-4165116b5129
begin 
	data_opt_sc2_g_4M_relu_final_el_50k_fast = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_a_4M_relu_final_el_50k_fast = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_m_4M_relu_final_el_50k_fast = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_relu_final,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_f152_4M_relu_final_el_50k_fast = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru152_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_f100_4M_relu_final_el_50k_fast = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru100_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_f64_4M_relu_final_el_50k_fast = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru64_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
end

# ╔═╡ 7a631e59-16cf-429e-a6d0-c2ff30ef1245
begin 
	data_opt_sc2_g_4M_relu_final_el_50k_fast_ep = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line_ep(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_a_4M_relu_final_el_50k_fast_ep = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line_ep(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_m_4M_relu_final_el_50k_fast_ep = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_relu_final,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line_ep(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_f152_4M_relu_final_el_50k_fast_ep = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru152_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line_ep(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_f100_4M_relu_final_el_50k_fast_ep = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru100_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line_ep(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_f64_4M_relu_final_el_50k_fast_ep = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru64_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line_ep(x, :total_rews, :total_steps, n=100000))
end

# ╔═╡ cb0072a4-6460-4dd1-8063-981ff79996ac
function get_cumulative_line(ddict, key1, key2; n=0)
    ret = zeros(eltype(ddict["results"][key1]), sum(ddict["results"][key2]))
    cur_idx = 1
    for i in 1:length(ddict["results"][key1])
        ret[cur_idx:(cur_idx + ddict["results"][key2][i] - 1)] .= ddict["results"][key1][i]
        cur_idx += ddict["results"][key2][i]
    end

    if n == 0
        ret
    else
		return_arr = zeros(eltype(ddict["results"][key1]), Int(floor(length(ret)/n)))
		for i in 1:Int(floor(length(ret)/n))
			return_arr[i] = sum(ret[1: i*n])
		end
		return_arr
    end
end

# ╔═╡ 7f6d42ac-d9ae-40d7-be67-ef8ef235d372
begin 
	data_opt_sc2_g_4M_relu_final_cl_50k_fast = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_cumulative_line(x, :total_rews, :total_steps, n=50000))
	
	data_opt_sc2_a_4M_relu_final_cl_50k_fast = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_cumulative_line(x, :total_rews, :total_steps, n=50000))
	
	data_opt_sc2_m_4M_relu_final_cl_50k_fast = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_relu_final,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_cumulative_line(x, :total_rews, :total_steps, n=50000))
	
	data_opt_sc2_f152_4M_relu_final_cl_50k_fast = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru152_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_cumulative_line(x, :total_rews, :total_steps, n=50000))
	
	data_opt_sc2_f100_4M_relu_final_cl_50k_fast = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru100_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_cumulative_line(x, :total_rews, :total_steps, n=50000))
	
	data_opt_sc2_f64_4M_relu_final_cl_50k_fast = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru64_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_cumulative_line(x, :total_rews, :total_steps, n=50000))
end

# ╔═╡ ef0208aa-bc71-44f8-ae33-d1c7c42e22da
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

# ╔═╡ 15693091-5dc9-4104-ae13-42f20046be19
diff(sub_ic_m)

# ╔═╡ 7eda4150-c8e1-4567-b7b9-2b41473eead9
ic_dir, dd_dir = ic_dir_os6_sc2, dd_dir_os6_sc2

# ╔═╡ eab5c7a0-8052-432d-977e-68b967baf5ca
ic_dir[1].parsed_args["steps"]

# ╔═╡ ca81ca07-cd6e-4b0b-93fa-2cdcc3b10178
let
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final_cl_50k_fast,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=2, color=cell_colors["GRU"])
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_cl_50k_fast,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=2, color=cell_colors["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final_cl_50k_fast,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=2, title="Lunar Lander MUE, $(ic_dir[1].parsed_args["steps"]) steps", color=cell_colors["MAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final_cl_50k_fast,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylabel="Total Reward", xlabel="Steps (50 Thousand)", lw=2, title="Lunar Lander MUE, 4M steps", color=cell_colors["FacMAGRU"])
	

end

# ╔═╡ 7252105e-2ffd-4ddd-b1b1-fe05dcf3b0d2
let
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final_sum,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=2, color=cell_colors["GRU"])
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_sum,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=2, color=cell_colors["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final_sum,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=2, title="Lunar Lander MUE, $(ic_dir[1].parsed_args["steps"]) steps", color=cell_colors["MAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final_sum,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylabel="Total Reward", xlabel="Steps (50 Thousand)", lw=2, title="Lunar Lander MUE, 4M steps", color=cell_colors["FacMAGRU"])
	

end

# ╔═╡ 1edc751d-0a75-459d-9ce0-4d4c8c9bc5cc
FileIO.load(joinpath(ic_dir[2].folder_str, "settings.jld2"))

# ╔═╡ 54d7c9dd-a442-4f26-97ab-e7d2549a1f64
sum(FileIO.load(joinpath(ic_dir[5].folder_str, "results.jld2"))["results"][:total_steps])

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

# ╔═╡ 07aab6c1-1462-415e-8468-ae8a20c6c910
data_opt_sc2_m_eta = RPU.get_line_data_for(
	sub_ic_sc2_m,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 500))

# ╔═╡ bf7d0904-94db-4e83-af9b-022944a52de9
data_opt_sc2_f_eta = RPU.get_line_data_for(
	sub_ic_sc2_g,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))

# ╔═╡ 3233dad1-9e7d-4745-b99b-68c09396679c
data_opt_sc2_g

# ╔═╡ a53426cd-2424-46f0-b8a5-5285c7a72fc3
md"""
Eta: $(@bind eta_dir_opt MultiSelect(string.(dd_dir_os6_sc2_facmagru_4M["eta"])))

"""

# ╔═╡ 1dbf4d41-546f-45e1-9cad-d740b3f9080d
let 
	plt = plot()
	for eta in eta_dir_opt
		eta_ = parse(Float64, eta)
		plt = plot!(
		  data_opt_sc2_f_eta,
		  Dict("eta"=>eta_), label="eta: $(eta), opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Episode", lw=2)
	end
	plt
end

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

# ╔═╡ e656cc0a-d3a2-427e-a17d-b332808c1cea
let
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final_sum,
		  Dict(), label="cell: GRU (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors["GRU"])
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_sum,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final_sum,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors["MAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final_sum,
		  Dict(), label="cell: FacMAGRU (nh: 152, fac: 170",
			  palette=RPU.custom_colorant, legend=:bottomleft, ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors["FacMAGRU"], title="Lunar Lander Cumulative Reward, Steps: 4M")
	plt
end

# ╔═╡ 315d6f3a-19ee-4a8b-800b-873379a59485
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final,
		  Dict(), label="cell: GRU (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors["GRU"])
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors["MAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final,
		  Dict(), label="cell: FacMAGRU (nh: 152, fac: 170",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors["FacMAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f100_4M_relu_final,
		  Dict(), label="cell: FacMAGRU (nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors["FacMARNN"])
	
	plt = plot!(
		  data_opt_sc2_f64_4M_relu_final,
		  Dict(), label="cell: FacMAGRU (nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors["MARNN"], title="Lunar Lander 20 Independent Runs, Steps: 4M", tickfontsize=11)
	plt
end

# ╔═╡ 1c97ac4a-654b-4dc9-9553-eab35d7430ab
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu,
		  Dict(), label="cell: GRU (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["GRU"])
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["MAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu,
		  Dict(), label="cell: FacMAGRU (nh: 152, fac: 170",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["FacMAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f100_4M_relu,
		  Dict(), label="cell: FacMAGRU (nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["RNN"])
	
	plt = plot!(
		  data_opt_sc2_f64_4M_relu,
		  Dict(), label="cell: FacMAGRU (nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["MARNN"], title="Lunar Lander ReLU, Steps: 4M")
	plt
end

# ╔═╡ 9aa2a048-78c6-4d07-b9c2-51815c75822d
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M,
		  Dict(), label="cell: GRU (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["GRU"])
	
	plt = plot!(
		  data_opt_sc2_a_4M,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_4M,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["MAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f152_4M,
		  Dict(), label="cell: FacMAGRU (nh: 152, τ: 170",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["FacMAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f100_4M,
		  Dict(), label="cell: FacMAGRU (nh: 100, τ: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["RNN"])
	
	plt = plot!(
		  data_opt_sc2_f64_4M,
		  Dict(), label="cell: FacMAGRU (nh: 64, τ: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["MARNN"], title="Lunar Lander, Steps: 4M")
	plt
end

# ╔═╡ 21705c72-53ec-48c5-8b20-94d83344b30b
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_8M,
		  Dict(), label="cell: GRU (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["GRU"])
	
	plt = plot!(
		  data_opt_sc2_a_8M,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_8M,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["MAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f152_8M,
		  Dict(), label="cell: FacMAGRU (nh: 152, τ: 170",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["FacMAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f100_8M,
		  Dict(), label="cell: FacMAGRU (nh: 100, τ: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["RNN"])
	
	plt = plot!(
		  data_opt_sc2_f64_8M,
		  Dict(), label="cell: FacMAGRU (nh: 64, τ: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, color=cell_colors["MARNN"], title="Lunar Lander, Steps: 8M")
	plt
end

# ╔═╡ b1f2943c-202b-485f-a9f5-7310749c07a8
data = RPU.get_line_data_for(
	ic,
	["numhidden", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))

# ╔═╡ c2e23373-a28e-4a7f-adc9-722da1c476c8
data_ = RPU.get_line_data_for(
	ic_,
	["numhidden", "factors"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))

# ╔═╡ cb355796-cf8b-4aef-baa1-797871aeaacb
let
	boxplot(data_opt_sc2_a, Dict();
		color=cell_colors["AAGRU"],
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="MUE")
	dotplot!(data_10_dist, Dict();
		color=cell_colors["AAGRU"])
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ 46d10bea-5573-41a2-bd18-917ee9ca5d6f
diff(ic)

# ╔═╡ 6183ae39-7277-419e-9e4d-e7b21d63c47e
ic.items

# ╔═╡ 0ae07bd4-f051-4071-bf47-0b26215f6dee
diff(ic_)

# ╔═╡ a9b56696-2d66-4b20-a5a9-a25ff52ba5bf
boxplot(data_opt_sc2_a, Dict("cell"=>"GRU"), label_idx="cell", legend=nothing, color=cell_colors["GRU"])

# ╔═╡ c3a60b0c-63e7-4799-ba9d-05ff9ed11e96
let
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: GRU, (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=2, color=cell_colors["GRU"])
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: AAGRU, (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=2, color=cell_colors["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: MAGRU, (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=2, title="Lunar Lander MUE, $(ic_dir[1].parsed_args["steps"]) steps", color=cell_colors["MAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: FacMAGRU, (nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 170), ylabel="Total Reward", xlabel="Steps (50 Thousand)", lw=2, title="Lunar Lander MUE, $(ic_dir[1].parsed_args["steps"]) steps", color=cell_colors["FacMAGRU"])
	

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

# ╔═╡ 5e9aeddc-337c-484b-96df-cb4492111bb2
md"""
Truncation: $(@bind τ_dir_all Select(string.(dd_dir["truncation"])))
Eta: $(@bind eta_dir_all Select(string.(dd_dir["eta"])))

"""

# ╔═╡ 3c45d9a9-e4ab-4af2-abb3-2b6c5fd88e32
let
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final_el_50k,
		  Dict(), label="cell: GRU, (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=2)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_el_50k,
		  Dict(), label="cell: AAGRU, (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=2)
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final_el_50k,
		  Dict(), label="cell: MAGRU, (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=2, title="Lunar Lander MUE, $(ic_dir[1].parsed_args["steps"]) steps")
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final_el_50k,
		  Dict(), label="cell: FacMAGRU, (nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (50 Thousand)", lw=2, title="Lunar Lander MUE, $(ic_dir[1].parsed_args["steps"]) steps")
	

end

# ╔═╡ 60d09c31-6555-476f-8c29-1c312a8e36e1
let
	# plt = nothing
	τ = parse(Int, τ_dir_all)
	eta = parse(Float64, eta_dir_all)
	
	plt = plot()
	plt = plot!(
		  data_opt_g_all,
		  Dict("truncation"=>τ, "eta"=>eta), label="cell: GRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (10 Thousand)", title="Lunar Lander, $(ic_dir[1].parsed_args["steps"]) steps", lw=2)
	
	plt = plot!(
		  data_opt_a_all,
		  Dict("truncation"=>τ, "eta"=>eta), label="cell: AAGRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (10 Thousand)", title="Lunar Lander, $(ic_dir[1].parsed_args["steps"]) steps", lw=2)
	
	plt = plot!(
		  data_opt_m_all,
		  Dict("truncation"=>τ, "eta"=>eta), label="cell: MAGRU, opt: RMSProp",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (10 Thousand)", title="Lunar Lander, $(ic_dir[1].parsed_args["steps"]) steps", lw=2)
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

# ╔═╡ 90bf6c60-0a64-47d7-b63a-de4fbd82b5e9
begin 
	data_sens_g = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_4M_relu,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_sens_a = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_relu,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_sens_m = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_relu,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_sens_f152 = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru152_4M_relu,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_sens_f100 = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru100_4M_relu,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_sens_f64 = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru64_4M_relu,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
end

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

# ╔═╡ 699c81ce-9043-42f4-900f-953128ac1265
cell_colors_ = Dict(
	"RNN" => color_scheme_[3],
	"AARNN" => color_scheme_[end],
	"MARNN" => color_scheme_[5],
	"FacMARNN" => color_scheme_[1],
	"GRU" => color_scheme_[4],
	"AAGRU" => color_scheme_[2],
	"MAGRU" => color_scheme_[6],
	"FacMAGRU" => color_scheme_[end-2])

# ╔═╡ 930746a7-ce5a-45fd-8963-08c9302a9685
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final_steps,
		  Dict(), label="cell: GRU (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors_["GRU"])
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_steps,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors_["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final_steps,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors_["MAGRU"], grid=false, tickdir=:out, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, fillalpha=0.3, title="Lunar Lander", ylabel="Steps", xlabel="Episode")
	
	plt

	savefig("../data/paper_plots/lunar_lander_er_step_curves_4M_nonfac_cells.pdf")
end

# ╔═╡ 2f992ed9-e6f1-4b1b-907b-9f2bd0361b04
let
	# plt = nothing
	plt = plot()	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final_steps,
		  Dict(), label="cell: FacGRU (nh: 152, fac: 170",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(100, 450), lw=2, z=1, color=cell_colors_["FacMAGRU"], grid=false, tickdir=:out, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, fillalpha=0.3, title="Lunar Lander", ylabel="Steps", xlabel="Episode")
	
	plt = plot!(
		  data_opt_sc2_f100_4M_relu_final_steps,
		  Dict(), label="cell: FacGRU (nh: 100, fac: 265",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(100, 450), lw=2, z=1, color=cell_colors_["FacMAGRU"], grid=false, tickdir=:out, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, fillalpha=0.3, linestyle=:dot, title="Lunar Lander", ylabel="Steps", xlabel="Episode")
	
	plt = plot!(
		  data_opt_sc2_f64_4M_relu_final_steps,
		  Dict(), label="cell: FacGRU (nh: 64, fac: 380",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(100, 450), lw=2, z=1, color=cell_colors_["FacMAGRU"], grid=false, tickdir=:out, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, fillalpha=0.3, linestyle=:dash, title="Lunar Lander", ylabel="Steps", xlabel="Episode")
	plt
	
	savefig("../data/paper_plots/lunar_lander_er_step_curves_4M_fac_cells.pdf")
end

# ╔═╡ 8128d88a-d4f9-405d-a668-5dba0ebfd576
let
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: GRU, (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (100 Thousand)", lw=2, color=cell_colors_["GRU"], fillalpha=0.5, z=1)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: AAGRU, (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), lw=2, color=cell_colors_["AAGRU"], fillalpha=0.4, z=1)
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: MAGRU, (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), lw=2, color=cell_colors_["MAGRU"], fillalpha=0.5, z=1)
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: FacMAGRU, (nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-270, 170), lw=2, title="Lunar Lander MUE, 4M steps", color=cell_colors_["FacMAGRU"], fillalpha=0.4, tickfontsize=10, grid=false, tickdir=:out, z=1)
	

end

# ╔═╡ c132cec6-4a6a-470c-9a90-1d40eeaadcbe
let
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final_el_50k_fast,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylim=(-500, 300), lw=3, color=cell_colors_["GRU"], fillalpha=0.4, z=1)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_el_50k_fast,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylim=(-500, 300), lw=3, color=cell_colors_["AAGRU"], fillalpha=0.3, z=1)
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final_el_50k_fast,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylim=(-500, 300), lw=3, color=cell_colors_["MAGRU"], fillalpha=0.3, z=1)
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final_el_50k_fast,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylim=(-250, 170), lw=3, color=cell_colors_["FacMAGRU"], fillalpha=0.4, tickfontsize=12, grid=false, tickdir=:out, z=1)
	
#savefig("../data/paper_plots/lunar_lander_er_learning_curves_per_ep_per_step_4M_steps.pdf")
end

# ╔═╡ 5ba3ec72-5c7f-4f31-b947-d50f9a8056f0
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final,
		  Dict(), label="cell: GRU (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors_["GRU"])
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors_["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors_["MAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final,
		  Dict(), label="cell: FacMAGRU (nh: 152, fac: 170",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-250, 150), ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors_["FacMAGRU"], title="Lunar Lander Final Runs, Steps: 4M")
	plt
end

# ╔═╡ 5f70ab7f-f603-4c78-bddd-f60e125211ce
let
	# plt = nothing
	plt = plot()
	plt = plot!(data_sens_g,
	  Dict();
	  sort_idx="eta",
	  z=1.97, lw=2, xaxis=:log,
	  palette=RPU.custom_colorant, label="cell: GRU, (nh: 154)", xlabel="Learning Rate", ylabel="Mean Return (last 10% of episodes)", title="Lunar Lander",
	  color=cell_colors_["GRU"])
	
	plt = plot!(data_sens_a,
	  Dict();
	  sort_idx="eta",
	  z=1.97, lw=2, xaxis=:log,
	  palette=RPU.custom_colorant, label="cell: AAGRU, (nh: 152)", xlabel="Learning Rate", ylabel="Mean Return (last 10% of episodes)", title="Lunar Lander",
	  color=cell_colors_["AAGRU"])
	
	plt = plot!(data_sens_m,
	  Dict();
	  sort_idx="eta",
	  z=1.97, lw=2, xaxis=:log,
	  palette=RPU.custom_colorant, label="cell: MAGRU, (nh: 64)", xlabel="Learning Rate", ylabel="Mean Return (last 10% of episodes)", title="Lunar Lander",
	  color=cell_colors_["MAGRU"])
	
	plt = plot!(data_sens_f152,
	  Dict();
	  sort_idx="eta",
	  z=1.97, lw=2, xaxis=:log,
	  palette=RPU.custom_colorant, label="cell: FacGRU, (nh: 152, fac: 170)", xlabel="Learning Rate", ylabel="Mean Return (last 10% of episodes)", title="Lunar Lander",
	  color=cell_colors_["FacMAGRU"])
	
		plt = plot!(data_sens_f100,
	  Dict();
	  sort_idx="eta",
	  z=1.97, lw=2, xaxis=:log,
	  palette=RPU.custom_colorant, label="cell: FacGRU, (nh: 100, fac: 265)", xlabel="Learning Rate", ylabel="Mean Return (last 10% of episodes)", title="Lunar Lander", linestyle=:dot,
	  color=cell_colors_["FacMAGRU"])
	
		plt = plot!(data_sens_f64,
	  Dict();
	  sort_idx="eta",
	  z=1.97, lw=2, xaxis=:log,
	  palette=RPU.custom_colorant, label="cell: FacGRU, (nh: 64, fac: 380)", xlabel="Learning Rate", ylabel="Mean Return (last 10% of episodes)", title="Lunar Lander", linestyle=:dash, ylim=(-400, 550), grid=false, tickdir=:out,
	  color=cell_colors_["FacMAGRU"], legend=:topright, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15,)
	
	plt
#	savefig("../data/paper_plots/lunar_lander_er_lr_sensitivity_curves_4M_steps.pdf")
end

# ╔═╡ 84db962e-1192-4a16-8684-5263c0d675c6
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, lw=3, z=1, color=cell_colors_["GRU"], fillalpha=0.4)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, lw=3, z=1, color=cell_colors_["AAGRU"], fillalpha=0.4)
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, lw=3, z=1, color=cell_colors_["MAGRU"], fillalpha=0.4)
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylim=(-250, 150), lw=3, z=1, color=cell_colors_["FacMAGRU"], grid=false, tickdir=:out, tickfontsize=12, fillalpha=0.4, title="Lunar Lander")
	plt
	
#	savefig("../data/paper_plots/lunar_lander_er_learning_curves_4M_steps.pdf")
end

# ╔═╡ 42995a23-e5be-49b6-a4ea-28eee592ebd0
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final,
		  Dict(), label="cell: GRU (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["GRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["AAGRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["MAGRU"], ylim=(-300, 150), grid=false, tickdir=:out, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, fillalpha=0.3, title="Lunar Lander Non-Factored Cells")
	

	plt
	
	savefig("../data/paper_plots/lunar_lander_er_learning_curves_4M_eps_nonfac_cells.pdf")
end

# ╔═╡ c9e6f13c-0ab8-405f-a39d-656627384c67
let
	# plt = nothing
	plt = plot()
	
	plt = plot(
		  data_opt_sc2_f152_4M_relu_final,
		  Dict(), label="cell: FacGRU (nh: 152, fac: 170",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["FacMAGRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_f100_4M_relu_final,
		  Dict(), label="cell: FacGRU (nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["FacMAGRU"], fillalpha=0.3, linestyle=:dot)
	
	plt = plot!(
		  data_opt_sc2_f64_4M_relu_final,
		  Dict(), label="cell: FacGRU (nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=3, z=1, color=cell_colors_["FacMAGRU"],  grid=false, tickdir=:out, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, fillalpha=0.3, linestyle=:dash, title="Lunar Lander Factored Cells")
	plt
	
	savefig("../data/paper_plots/lunar_lander_er_learning_curves_4M_eps_fac_cells.pdf")
end

# ╔═╡ e3f6be18-cae5-4048-a635-f3ab049b8e79
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final,
		  Dict(), label="cell: GRU (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["GRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["AAGRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["MAGRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final,
		  Dict(), label="cell: FacGRU (nh: 152, fac: 170",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["FacMAGRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_f100_4M_relu_final,
		  Dict(), label="cell: FacGRU (nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["FacMAGRU"], fillalpha=0.3, linestyle=:dot)
	
	plt = plot!(
		  data_opt_sc2_f64_4M_relu_final,
		  Dict(), label="cell: FacGRU (nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=3, z=1, color=cell_colors_["FacMAGRU"],  grid=false, tickdir=:out, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, fillalpha=0.3, linestyle=:dash, title="Lunar Lander")
	plt
	
#	savefig("../data/paper_plots/lunar_lander_er_learning_curves_4M_eps_6_cells.pdf")
end

# ╔═╡ 4581ec85-5931-47ed-92c3-30d69c7ea635
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["GRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["AAGRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["MAGRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["FacMAGRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_f100_4M_relu_final,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors_["FacMAGRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_f64_4M_relu_final,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylim=(-250, 150), lw=3, z=1, color=cell_colors_["FacMAGRU"],  grid=false, tickdir=:out, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, fillalpha=0.3)
	plt
	
	#savefig("../data/paper_plots/lunar_lander_er_learning_curves_4M_steps_6_cells.pdf")
end

# ╔═╡ afe13e9f-ddd4-4b19-8b29-a0f41ea6f291
let
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: GRU, (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=3, color=cell_colors_["GRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: AAGRU, (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=3, color=cell_colors_["AAGRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: MAGRU, (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=3, title="Lunar Lander MUE, $(ic_dir[1].parsed_args["steps"]) steps", color=cell_colors_["MAGRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: FacGRU, (nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 170), ylabel="Total Reward", xlabel="Steps (100 Thousand)", lw=3, title="Lunar Lander MUE, 4M steps", color=cell_colors_["FacMAGRU"], fillalpha=0.3)
	
		plt = plot!(
		  data_opt_sc2_f100_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: FacGRU, (nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 170), ylabel="Total Reward", xlabel="Steps (100 Thousand)", lw=3, title="Lunar Lander MUE, 4M steps", color=cell_colors_["FacMAGRU"], fillalpha=0.3, linestyle=:dot)
	
		plt = plot!(
		  data_opt_sc2_f64_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: FacGRU, (nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 170), ylabel="Total Reward", xlabel="Steps (100 Thousand)", lw=3, title="Lunar Lander", color=cell_colors_["FacMAGRU"], fillalpha=0.3, linestyle=:dash, grid=false, tickdir=:out, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15,)
	
	plt
	
savefig("../data/paper_plots/lunar_lander_er_learning_curves_4M_steps_6_cells.pdf")
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
# ╠═c5cd1680-ccd5-4201-83ae-4be257507515
# ╠═55bd927b-1b3e-49d0-8056-4c820ba60fbf
# ╠═eab5c7a0-8052-432d-977e-68b967baf5ca
# ╠═01b9cd42-d64e-4988-a5b0-3fd5cddbf728
# ╠═00039914-c54f-46d8-9c54-51d8ddda236d
# ╠═44030a7c-92ea-44ce-b221-7a0d6225d787
# ╠═50734263-f616-4793-a580-7aa0f7a29223
# ╠═b5a22572-cc8f-4159-8f8f-246bd28d4053
# ╠═4acf4ecb-9959-4602-b23e-0bd4ef0f4e87
# ╠═735aaf47-d8d1-46aa-b4f2-1358e4832551
# ╠═b2a18bf1-f75a-42b3-a207-78bc4d9e043f
# ╠═ae4cfd3a-084b-48f2-8a19-421853a189c9
# ╠═77ded675-c775-4e17-b51f-9084f3ccbb88
# ╠═c93e55ba-62a3-4f73-b552-7ba90cb4a8fb
# ╠═6e73dbc1-3615-4a78-8316-6c63aae62606
# ╠═930746a7-ce5a-45fd-8963-08c9302a9685
# ╠═2f992ed9-e6f1-4b1b-907b-9f2bd0361b04
# ╠═dcfd8f8c-ed8d-4b44-bacb-673ff9e05c09
# ╠═0757a4ec-aea3-4d0a-a5d7-b1ae4953db25
# ╠═c92d71f5-fbd6-42dc-9938-303745e7a2f2
# ╠═d9857f89-ed14-4dfb-8ba8-ad003ef72dfb
# ╠═d64d298b-1d16-4f27-9d50-b78c70e77e58
# ╠═fcfe0f74-958a-4bd7-9163-4165116b5129
# ╠═7a631e59-16cf-429e-a6d0-c2ff30ef1245
# ╠═7f6d42ac-d9ae-40d7-be67-ef8ef235d372
# ╠═cb0072a4-6460-4dd1-8063-981ff79996ac
# ╟─ca81ca07-cd6e-4b0b-93fa-2cdcc3b10178
# ╠═7252105e-2ffd-4ddd-b1b1-fe05dcf3b0d2
# ╠═ef0208aa-bc71-44f8-ae33-d1c7c42e22da
# ╠═8128d88a-d4f9-405d-a668-5dba0ebfd576
# ╠═c132cec6-4a6a-470c-9a90-1d40eeaadcbe
# ╠═15693091-5dc9-4104-ae13-42f20046be19
# ╠═7eda4150-c8e1-4567-b7b9-2b41473eead9
# ╠═1edc751d-0a75-459d-9ce0-4d4c8c9bc5cc
# ╠═54d7c9dd-a442-4f26-97ab-e7d2549a1f64
# ╠═efede3d6-d147-47ee-9dbc-59bdc5272769
# ╠═8fd0bdb1-4db3-4a19-986a-e29addc1974a
# ╠═e51d91bc-c2d7-4d07-8319-803e28ff02d6
# ╠═ae784ee9-c937-4eda-89e8-6d1399aa7a36
# ╠═032c75e5-ddd1-4b16-b524-af2d8f99d41e
# ╠═07aab6c1-1462-415e-8468-ae8a20c6c910
# ╠═bf7d0904-94db-4e83-af9b-022944a52de9
# ╠═3233dad1-9e7d-4745-b99b-68c09396679c
# ╠═a53426cd-2424-46f0-b8a5-5285c7a72fc3
# ╟─1dbf4d41-546f-45e1-9cad-d740b3f9080d
# ╟─e3ebad86-41aa-452c-a6b3-48c3b497c27d
# ╟─db53e161-68ac-421d-b433-4307da6e0442
# ╠═503e447c-b7d0-4805-acde-0993179b9781
# ╟─e656cc0a-d3a2-427e-a17d-b332808c1cea
# ╠═5ba3ec72-5c7f-4f31-b947-d50f9a8056f0
# ╠═315d6f3a-19ee-4a8b-800b-873379a59485
# ╟─1c97ac4a-654b-4dc9-9553-eab35d7430ab
# ╟─9aa2a048-78c6-4d07-b9c2-51815c75822d
# ╟─21705c72-53ec-48c5-8b20-94d83344b30b
# ╠═b1f2943c-202b-485f-a9f5-7310749c07a8
# ╠═c2e23373-a28e-4a7f-adc9-722da1c476c8
# ╠═cb355796-cf8b-4aef-baa1-797871aeaacb
# ╠═46d10bea-5573-41a2-bd18-917ee9ca5d6f
# ╠═6183ae39-7277-419e-9e4d-e7b21d63c47e
# ╠═0ae07bd4-f051-4071-bf47-0b26215f6dee
# ╠═a9b56696-2d66-4b20-a5a9-a25ff52ba5bf
# ╠═c3a60b0c-63e7-4799-ba9d-05ff9ed11e96
# ╠═e2fecec9-0556-4517-b65d-8ddf81c82b0c
# ╟─0bb67028-be79-45eb-9370-0f53bb6055ca
# ╟─5e9aeddc-337c-484b-96df-cb4492111bb2
# ╠═3c45d9a9-e4ab-4af2-abb3-2b6c5fd88e32
# ╠═60d09c31-6555-476f-8c29-1c312a8e36e1
# ╠═3d8b0b2c-005e-4294-8817-e6bd075a0fb8
# ╠═55d1416b-d580-4112-827a-30504c21f397
# ╠═467ed417-cf69-493d-b21c-3fc4d1fb9907
# ╠═90bf6c60-0a64-47d7-b63a-de4fbd82b5e9
# ╠═2040d613-0ea8-48ce-937c-9180073812ea
# ╠═7882977c-4802-4372-ab5b-1e2e45130fed
# ╟─40cc0919-c56b-401b-83fa-0feb263c44b0
# ╠═5f70ab7f-f603-4c78-bddd-f60e125211ce
# ╠═84db962e-1192-4a16-8684-5263c0d675c6
# ╠═42995a23-e5be-49b6-a4ea-28eee592ebd0
# ╠═c9e6f13c-0ab8-405f-a39d-656627384c67
# ╠═e3f6be18-cae5-4048-a635-f3ab049b8e79
# ╠═699c81ce-9043-42f4-900f-953128ac1265
# ╠═4581ec85-5931-47ed-92c3-30d69c7ea635
# ╠═afe13e9f-ddd4-4b19-8b29-a0f41ea6f291
