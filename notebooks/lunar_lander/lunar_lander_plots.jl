### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ ca083fbd-7625-4cf6-9064-f98a486433ad
let
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ f7f500a8-a1e9-11eb-009b-d7afdcade891
using Revise

# ╔═╡ e0d51e67-63dc-45ea-9092-9965f97660b3
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 62c28887-331d-4e53-865d-8e71685c9e91
at(dir) = joinpath("../../local_data/lunar_lander/", dir) # was ../local_data/LL_relu/

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
	colorant"#1E90FF",
]

# ╔═╡ 1886bf05-f4be-4160-b61c-edf186a7f3cb
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ fe50ffef-b691-47b5-acf8-8378fbf860a1
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
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

# ╔═╡ 50734263-f616-4793-a580-7aa0f7a29223
begin
	ic_dir_os6_sc2_gru_4M_relu, dd_dir_os6_sc2_gru_4M_relu = RPU.load_data(at("lunar_lander_er_relu_rmsprop_os6_sc2_gru_4M/"))
	ic_dir_os6_sc2_aagru_4M_relu, dd_dir_os6_sc2_aagru_4M_relu = RPU.load_data(at("lunar_lander_er_relu_rmsprop_os6_sc2_aagru_4M/"))
	ic_dir_os6_sc2_magru_4M_relu, dd_dir_os6_sc2_magru_4M_relu = RPU.load_data(at("lunar_lander_er_relu_rmsprop_os6_sc2_magru_4M/"))
	ic_dir_os6_sc2_facmagru152_4M_relu, dd_dir_os6_sc2_facmagru152_4M_relu = RPU.load_data(at("lunar_lander_er_relu_rmsprop_os6_sc2_facmagru152_4M/"))
	ic_dir_os6_sc2_facmagru100_4M_relu, dd_dir_os6_sc2_facmagru100_4M_relu = RPU.load_data(at("lunar_lander_er_relu_rmsprop_os6_sc2_facmagru100_4M/"))
	ic_dir_os6_sc2_facmagru64_4M_relu, dd_dir_os6_sc2_facmagru64_4M_relu = RPU.load_data(at("lunar_lander_er_relu_rmsprop_os6_sc2_facmagru64_4M/"))
end

# ╔═╡ a3370aae-bd38-42d6-8653-ba4299903c71
begin
	ic_dir_os6_sc2_aagru_4M_tau_sweep, dd_dir_os6_sc2_aagru_4M_tau_sweep = RPU.load_data("../local_data/lunar_lander/lunar_lander_er_rmsprop_os6_sc2_aagru_4M/")
	ic_dir_os6_sc2_magru_4M_tau_sweep, dd_dir_os6_sc2_magru_4M_tau_sweep = RPU.load_data("../local_data/lunar_lander/lunar_lander_er_rmsprop_os6_sc2_magru_4M/")
end

# ╔═╡ b5a22572-cc8f-4159-8f8f-246bd28d4053
begin
	ic_dir_os6_sc2_gru_4M_relu_final, dd_dir_os6_sc2_gru_4M_relu_final = RPU.load_data(at("final_lunar_lander_er_relu_rmsprop_os6_sc2_gru_4M/"))
	
	ic_dir_os6_sc2_aagru_4M_relu_final, dd_dir_os6_sc2_aagru_4M_relu_final = RPU.load_data(at("final_lunar_lander_er_relu_rmsprop_os6_sc2_aagru_4M/"))
	
	ic_dir_os6_sc2_magru_4M_relu_final, dd_dir_os6_sc2_magru_4M_relu_final = RPU.load_data(at("final_lunar_lander_er_relu_rmsprop_os6_sc2_magru_4M/"))
	
	ic_dir_os6_sc2_facmagru152_4M_relu_final, dd_dir_os6_sc2_facmagru152_4M_relu_final = RPU.load_data(at("final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru152_4M/"))
	
	ic_dir_os6_sc2_facmagru100_4M_relu_final, dd_dir_os6_sc2_facmagru100_4M_relu_final = RPU.load_data(at("final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru100_4M/"))
	
	ic_dir_os6_sc2_facmagru64_4M_relu_final, dd_dir_os6_sc2_facmagru64_4M_relu_final = RPU.load_data(at("final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru64_4M/"))
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

# ╔═╡ ebcd4a27-8e86-44b3-aa9d-833f8cbf2e21
begin 
	data_opt_sc2_a_4M_tau_sweep = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_tau_sweep,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_opt_sc2_m_4M_tau_sweep = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_tau_sweep,
	["truncation"],
	["eta"];
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

# ╔═╡ c9fe540f-592d-4085-8732-88d1c5e035d6
let
	# plt = nothing
	plt = plot()
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final,
		  Dict(), label="cell: AAGRU (nh: 152, tau: 16)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_a_4M_tau_sweep,
		  Dict("truncation"=>12), label="cell: AAGRU (nh: 152, tau: 12)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-250, 120), lw=2, z=1, grid=false, tickdir=:out, color=cell_colors["AAGRU"], linestyle=:dot)
	
	plt = plot!(
		  data_opt_sc2_a_4M_tau_sweep,
		  Dict("truncation"=>8), label="cell: AAGRU (nh: 152, tau: 8)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-250, 120), lw=2, z=1, title="MUE", xlabel="Episode", grid=false, tickdir=:out, color=cell_colors["AAGRU"], linestyle=:dash)
	
	plt
end

# ╔═╡ 7cf8378d-6197-4ea6-a536-beb1ef4d311a
let
	# plt = nothing
	plt = plot()
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final,
		  Dict(), label="cell: MAGRU (nh: 64, tau: 16)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors["MAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_4M_tau_sweep,
		  Dict("truncation"=>12), label="cell: MAGRU (nh: 64, tau: 12)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-250, 120), lw=2, z=1, grid=false, tickdir=:out, color=cell_colors["MAGRU"], linestyle=:dot)
	
	plt = plot!(
		  data_opt_sc2_m_4M_tau_sweep,
		  Dict("truncation"=>8), label="cell: MAGRU (nh: 64, tau: 8)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-250, 120), lw=2, z=1, title="MUE", xlabel="Episode", grid=false, tickdir=:out, color=cell_colors["MAGRU"], linestyle=:dash)
	
	plt
end

# ╔═╡ 71571300-e381-41eb-ab76-f868b1b80f1c
begin 
	plt = plot()
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final,
		  Dict(), label="cell: MAGRU (nh: 64, tau: 16)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors["MAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_4M_tau_sweep,
		  Dict("truncation"=>12), label="cell: MAGRU (nh: 64, tau: 12)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-250, 120), lw=2, z=1, grid=false, tickdir=:out, color=cell_colors["MAGRU"], linestyle=:dot)
	
	plt = plot!(
		  data_opt_sc2_m_4M_tau_sweep,
		  Dict("truncation"=>8), label="cell: MAGRU (nh: 64, tau: 8)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-250, 120), lw=2, z=1, title="MUE", xlabel="Episode", grid=false, tickdir=:out, color=cell_colors["MAGRU"], linestyle=:dash)
	

	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final,
		  Dict(), label="cell: AAGRU (nh: 152, tau: 16)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_a_4M_tau_sweep,
		  Dict("truncation"=>12), label="cell: AAGRU (nh: 152, tau: 12)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-250, 120), lw=2, z=1, grid=false, tickdir=:out, color=cell_colors["AAGRU"], linestyle=:dot)
	
	plt = plot!(
		  data_opt_sc2_a_4M_tau_sweep,
		  Dict("truncation"=>8), label="cell: AAGRU (nh: 152, tau: 8)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-250, 120), lw=2, z=1, title="MUE", xlabel="Episode", grid=false, tickdir=:out, color=cell_colors["AAGRU"], linestyle=:dash)
	
		plt
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

# ╔═╡ 930746a7-ce5a-45fd-8963-08c9302a9685
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final_steps,
		  Dict(), label="cell: GRU (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors["GRU"])
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_steps,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final_steps,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors["MAGRU"], grid=false, tickdir=:out, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, fillalpha=0.3, title="Lunar Lander", ylabel="Steps", xlabel="Episode")
	
	plt

#	savefig("../data/paper_plots/lunar_lander_er_step_curves_4M_nonfac_cells.pdf")
end

# ╔═╡ 2f992ed9-e6f1-4b1b-907b-9f2bd0361b04
let
	# plt = nothing
	plt = plot()	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final_steps,
		  Dict(), label="cell: FacGRU (nh: 152, fac: 170",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(100, 450), lw=2, z=1, color=cell_colors["FacMAGRU"], grid=false, tickdir=:out, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, fillalpha=0.3, title="Lunar Lander", ylabel="Steps", xlabel="Episode")
	
	plt = plot!(
		  data_opt_sc2_f100_4M_relu_final_steps,
		  Dict(), label="cell: FacGRU (nh: 100, fac: 265",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(100, 450), lw=2, z=1, color=cell_colors["FacMAGRU"], grid=false, tickdir=:out, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, fillalpha=0.3, linestyle=:dot, title="Lunar Lander", ylabel="Steps", xlabel="Episode")
	
	plt = plot!(
		  data_opt_sc2_f64_4M_relu_final_steps,
		  Dict(), label="cell: FacGRU (nh: 64, fac: 380",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(100, 450), lw=2, z=1, color=cell_colors["FacMAGRU"], grid=false, tickdir=:out, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, fillalpha=0.3, linestyle=:dash, title="Lunar Lander", ylabel="Steps", xlabel="Episode")
	plt
	
#	savefig("../data/paper_plots/lunar_lander_er_step_curves_4M_fac_cells.pdf")
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

# ╔═╡ d1512e4b-33bb-4c36-9745-6d5f1224b6b4
begin 
	data_opt_sc2_a_4M_tau_sweep_el = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_tau_sweep,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_m_4M_tau_sweep_el = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_tau_sweep,
	["truncation"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
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

# ╔═╡ e90d3154-b408-40e5-821c-7c63e29828e7
let
	# plt = nothing
	plt = plot()
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: AAGRU (nh: 152, tau: 16)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors["AAGRU"])
	
	plt = plot!(
		  data_opt_sc2_a_4M_tau_sweep_el,
		  Dict("truncation"=>12), label="cell: AAGRU (nh: 152, tau: 12)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, grid=false, tickdir=:out, color=cell_colors["AAGRU"], linestyle=:dot)
	
	plt = plot!(
		  data_opt_sc2_a_4M_tau_sweep_el,
		  Dict("truncation"=>8), label="cell: AAGRU (nh: 152, tau: 8)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-250, 200), lw=2, z=1, title="MUE", xlabel="Steps (100k)", grid=false, tickdir=:out, color=cell_colors["AAGRU"], linestyle=:dash)
	
		plt = plot!(
		  data_opt_sc2_a_4M_tau_sweep_el,
		  Dict("truncation"=>1), label="cell: AAGRU (nh: 152, tau: 1)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-250, 200), lw=2, z=1, title="MUE", xlabel="Steps (100k)", grid=false, tickdir=:out, color=cell_colors["AAGRU"], linestyle=:dashdot)
	
	plt
end

# ╔═╡ 8a2a366f-55a3-4708-b663-d4037b04a65e
let
	# plt = nothing
	plt = plot()
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: MAGRU (nh: 64, tau: 16)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, color=cell_colors["MAGRU"])
	
	plt = plot!(
		  data_opt_sc2_m_4M_tau_sweep_el,
		  Dict("truncation"=>12), label="cell: AAGRU (nh: 64, tau: 12)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=2, z=1, grid=false, tickdir=:out, color=cell_colors["MAGRU"], linestyle=:dot)
	
	plt = plot!(
		  data_opt_sc2_m_4M_tau_sweep_el,
		  Dict("truncation"=>8), label="cell: AAGRU (nh: 64, tau: 8)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-250, 200), lw=2, z=1, title="MUE", xlabel="Steps (100k)", grid=false, tickdir=:out, color=cell_colors["MAGRU"], linestyle=:dash)
	
	plt = plot!(
		  data_opt_sc2_m_4M_tau_sweep_el,
		  Dict("truncation"=>1), label="cell: AAGRU (nh: 64, tau: 1)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-250, 200), lw=2, z=1, title="MUE", xlabel="Steps (100k)", grid=false, tickdir=:out, color=cell_colors["MAGRU"], linestyle=:dashdot)
	
	plt
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
			  palette=RPU.custom_colorant, legend=nothing, ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=2, title="Lunar Lander MUE", color=cell_colors["MAGRU"])
	
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
			  palette=RPU.custom_colorant, legend=nothing, ylabel="Total Reward", xlabel="Steps (10 Thousand)", lw=2, title="Lunar Lander MUE", color=cell_colors["MAGRU"])
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final_sum,
		  Dict(), label=nothing,
			  palette=RPU.custom_colorant, legend=nothing, ylabel="Total Reward", xlabel="Episodes", lw=2, title="Lunar Lander MUE, 4M steps", color=cell_colors["FacMAGRU"])
	

end

# ╔═╡ 8128d88a-d4f9-405d-a668-5dba0ebfd576
let
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: GRU, (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), ylabel="Total Reward", xlabel="Steps (100 Thousand)", lw=2, color=cell_colors["GRU"], fillalpha=0.5, z=1)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: AAGRU, (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), lw=2, color=cell_colors["AAGRU"], fillalpha=0.4, z=1)
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: MAGRU, (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-500, 300), lw=2, color=cell_colors["MAGRU"], fillalpha=0.5, z=1)
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final_el_50k_fast,
		  Dict(), label="cell: FacMAGRU, (nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-270, 170), lw=2, title="Lunar Lander MUE, 4M steps", color=cell_colors["FacMAGRU"], fillalpha=0.4, tickfontsize=10, grid=false, tickdir=:out, z=1)
	

end

# ╔═╡ 0ba0ae82-b568-4204-86e3-c4150aac6818
begin 
	data_opt_sc2_a_4M_tau_sweep_sens = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_tau_sweep,
	["truncation", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_opt_sc2_m_4M_tau_sweep_sens = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_tau_sweep,
	["truncation", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
end

# ╔═╡ 5f70ab7f-f603-4c78-bddd-f60e125211ce
let
	# plt = nothing
	plt = plot()
	
	plt = plot!(data_opt_sc2_a_4M_tau_sweep_sens,
	  Dict("truncation"=>12);
	  sort_idx="eta",
	  z=1.97, lw=2, xaxis=:log,
	  palette=RPU.custom_colorant, label="cell: AAGRU, (nh: 152, tau: 12)", xlabel="Learning Rate", ylabel="Mean Return (last 10% of episodes)", title="Lunar Lander",
	  color=cell_colors["AAGRU"])
	
	plt = plot!(data_opt_sc2_m_4M_tau_sweep_sens,
	  Dict("truncation"=>12);
	  sort_idx="eta",
	  z=1.97, lw=2, xaxis=:log,
	  palette=RPU.custom_colorant, label="cell: MAGRU, (nh: 64, tau: 12)", xlabel="Learning Rate", ylabel="Mean Return (last 10% of episodes)", title="Lunar Lander",
	  color=cell_colors["MAGRU"], ylim=(-400, 550), grid=false, tickdir=:out, legend=:topright, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15,)
	
		plt = plot!(data_opt_sc2_a_4M_tau_sweep_sens,
	  Dict("truncation"=>8);
	  sort_idx="eta",
	  z=1.97, lw=2, xaxis=:log,
	  palette=RPU.custom_colorant, label="cell: AAGRU, (nh: 152, tau: 8)", xlabel="Learning Rate", ylabel="Mean Return (last 10% of episodes)", title="Lunar Lander",
	  color=cell_colors["AAGRU"])
	
	plt = plot!(data_opt_sc2_m_4M_tau_sweep_sens,
	  Dict("truncation"=>8);
	  sort_idx="eta",
	  z=1.97, lw=2, xaxis=:log,
	  palette=RPU.custom_colorant, label="cell: MAGRU, (nh: 64, tau: 8)", xlabel="Learning Rate", ylabel="Mean Return (last 10% of episodes)", title="Lunar Lander",
	  color=cell_colors["MAGRU"], ylim=(-400, 550), grid=false, tickdir=:out, legend=:topright, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15,)
	

	
	plt
#	savefig("../data/paper_plots/lunar_lander_er_lr_sensitivity_curves_4M_steps.pdf")
end

# ╔═╡ 42995a23-e5be-49b6-a4ea-28eee592ebd0
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final,
		  Dict(), label="cell: GRU (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors["GRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-400, 150), ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors["AAGRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylabel="Total Reward", xlabel="Episode", lw=3, z=1, color=cell_colors["MAGRU"], ylim=(-300, 150), grid=false, tickdir=:out, tickfontsize=12, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, fillalpha=0.3, title="Lunar Lander Non-Factored Cells")
	

	plt
	
	#savefig("../data/paper_plots/lunar_lander_er_learning_curves_4M_eps_nonfac_cells.pdf")
end

# ╔═╡ Cell order:
# ╠═ca083fbd-7625-4cf6-9064-f98a486433ad
# ╠═f7f500a8-a1e9-11eb-009b-d7afdcade891
# ╠═e0d51e67-63dc-45ea-9092-9965f97660b3
# ╠═62c28887-331d-4e53-865d-8e71685c9e91
# ╠═6a65de2a-2af7-4da5-9f14-d18623b3235b
# ╠═0c746c1e-ea39-4415-a1b1-d7124b886f98
# ╠═1886bf05-f4be-4160-b61c-edf186a7f3cb
# ╠═fe50ffef-b691-47b5-acf8-8378fbf860a1
# ╟─834b1cf3-5b22-4e0a-abe9-61e427e6cfda
# ╠═50734263-f616-4793-a580-7aa0f7a29223
# ╠═a3370aae-bd38-42d6-8653-ba4299903c71
# ╠═b5a22572-cc8f-4159-8f8f-246bd28d4053
# ╠═4acf4ecb-9959-4602-b23e-0bd4ef0f4e87
# ╠═735aaf47-d8d1-46aa-b4f2-1358e4832551
# ╠═ebcd4a27-8e86-44b3-aa9d-833f8cbf2e21
# ╟─c9fe540f-592d-4085-8732-88d1c5e035d6
# ╟─7cf8378d-6197-4ea6-a536-beb1ef4d311a
# ╟─71571300-e381-41eb-ab76-f868b1b80f1c
# ╟─d1512e4b-33bb-4c36-9745-6d5f1224b6b4
# ╟─e90d3154-b408-40e5-821c-7c63e29828e7
# ╟─8a2a366f-55a3-4708-b663-d4037b04a65e
# ╠═ae4cfd3a-084b-48f2-8a19-421853a189c9
# ╠═77ded675-c775-4e17-b51f-9084f3ccbb88
# ╠═c93e55ba-62a3-4f73-b552-7ba90cb4a8fb
# ╠═6e73dbc1-3615-4a78-8316-6c63aae62606
# ╠═930746a7-ce5a-45fd-8963-08c9302a9685
# ╠═2f992ed9-e6f1-4b1b-907b-9f2bd0361b04
# ╠═dcfd8f8c-ed8d-4b44-bacb-673ff9e05c09
# ╠═0757a4ec-aea3-4d0a-a5d7-b1ae4953db25
# ╠═d9857f89-ed14-4dfb-8ba8-ad003ef72dfb
# ╠═d64d298b-1d16-4f27-9d50-b78c70e77e58
# ╠═fcfe0f74-958a-4bd7-9163-4165116b5129
# ╠═7f6d42ac-d9ae-40d7-be67-ef8ef235d372
# ╠═cb0072a4-6460-4dd1-8063-981ff79996ac
# ╟─ca81ca07-cd6e-4b0b-93fa-2cdcc3b10178
# ╟─7252105e-2ffd-4ddd-b1b1-fe05dcf3b0d2
# ╟─8128d88a-d4f9-405d-a668-5dba0ebfd576
# ╠═0ba0ae82-b568-4204-86e3-c4150aac6818
# ╠═5f70ab7f-f603-4c78-bddd-f60e125211ce
# ╠═42995a23-e5be-49b6-a4ea-28eee592ebd0
