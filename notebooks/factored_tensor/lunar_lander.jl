### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ ae353392-e006-11eb-0aad-03cc4a895756
using Revise

# ╔═╡ 3ae22ee3-2773-4748-a154-d5da599b7428
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ dc5aaa9e-bedf-4ce7-8dee-ddeafd101831
const RPU = ReproducePlotUtils

# ╔═╡ fee73b2f-0fbd-4314-91a3-783c63c5a321
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

# ╔═╡ b229fe0f-f8aa-4dd8-a19f-68a9e8cb83e0
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ 5b94a3d6-d829-462b-a48e-e13ef754d8cd
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ 2a0b7edb-57b5-4dca-bb74-472b9592af97
begin
	ic_magru, dd_magru = RPU.load_data("../../local_data/lunar_lander_er_relu_rmsprop_os6_sc2_magru_4M_wr/")
	ic_facmagru64, dd_facmagru64 = RPU.load_data("../../local_data/factored_tensor/lunar_lander_er_relu_rmsprop_os6_sc2_facmagru64_4M_wr/")
	ic_facmagru100, dd_facmagru100 = RPU.load_data("../../local_data/factored_tensor/lunar_lander_er_relu_rmsprop_os6_sc2_facmagru100_4M_wr/")
	ic_facmagru152, dd_facmagru152 = RPU.load_data("../../local_data/factored_tensor/lunar_lander_er_relu_rmsprop_os6_sc2_facmagru152_4M_wr/")
end

# ╔═╡ 476cfeaa-864d-4213-ace4-b82a6a8fde59
begin
	ic_facmagru64_f, dd_facmagru64_f = RPU.load_data("../../local_data/factored_tensor/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru64_4M_wr/")
	ic_facmagru100_f, dd_facmagru100_f = RPU.load_data("../../local_data/factored_tensor/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru100_4M_wr/")
	ic_facmagru152_f, dd_facmagru152_f = RPU.load_data("../../local_data/factored_tensor/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru152_4M_wr/")
end

# ╔═╡ 313041e3-05da-49e2-96d7-1aa080b22606
begin 
	data_magru = RPU.get_line_data_for(
	ic_magru,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_facmagru64 = RPU.get_line_data_for(
	ic_facmagru64,
	["init_style"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_facmagru100 = RPU.get_line_data_for(
	ic_facmagru100,
	["init_style"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_facmagru152 = RPU.get_line_data_for(
	ic_facmagru152,
	["init_style"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
end

# ╔═╡ 6b5557a1-6b5f-4c59-80d7-7003541a934d
begin 
	data_facmagru64_f = RPU.get_line_data_for(
	ic_facmagru64_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_facmagru100_f = RPU.get_line_data_for(
	ic_facmagru100_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_facmagru152_f = RPU.get_line_data_for(
	ic_facmagru152_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
end

# ╔═╡ 18f8651e-8686-4d71-99aa-3be8db6303a2
begin 
	data_facmagru64_f_steps = RPU.get_line_data_for(
	ic_facmagru64_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 2000))
	
	data_facmagru100_f_steps = RPU.get_line_data_for(
	ic_facmagru100_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 2000))
	
	data_facmagru152_f_steps = RPU.get_line_data_for(
	ic_facmagru152_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 2000))
end

# ╔═╡ 85803271-69dc-483e-8115-dffb07910066
begin 
	data_facmagru64_fm = RPU.get_line_data_for(
	ic_facmagru64_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_MEAN(x, :total_rews))
	
	data_facmagru100_fm = RPU.get_line_data_for(
	ic_facmagru100_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_MEAN(x, :total_rews))
	
	data_facmagru152_fm = RPU.get_line_data_for(
	ic_facmagru152_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_MEAN(x, :total_rews))
end

# ╔═╡ 00d5ab33-7583-40d0-ad61-4fce6a289bee
begin 
	data_facmagru64_fmue = RPU.get_line_data_for(
	ic_facmagru64_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_facmagru100_fmue = RPU.get_line_data_for(
	ic_facmagru100_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_facmagru152_fmue = RPU.get_line_data_for(
	ic_facmagru152_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
end

# ╔═╡ 037436b3-37a6-4734-9aea-cbb7185274c4
begin 
	data_magru_sens = RPU.get_line_data_for(
	ic_magru,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_facmagru64_sens = RPU.get_line_data_for(
	ic_facmagru64,
	["init_style", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_facmagru100_sens = RPU.get_line_data_for(
	ic_facmagru100,
	["init_style", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_facmagru152_sens = RPU.get_line_data_for(
	ic_facmagru152,
	["init_style", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
end

# ╔═╡ aad533aa-4b7b-40e3-8ebb-320eb3108426
let
	plt = plot()
	
	plt = plot!(
		  data_magru_sens,
		  Dict(), sort_idx="eta", xaxis=:log, label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["MAGRU"], fillalpha=0.2)
	plt = plot!(
		  data_facmagru64_sens,
		  Dict("init_style"=>"tensor"), sort_idx="eta", xaxis=:log, label="cell: FacMAGRU (init: tnsr, nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru100_sens,
		  Dict("init_style"=>"tensor"), sort_idx="eta", xaxis=:log, label="cell: FacMAGRU (init: tnsr, nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru152_sens,
		  Dict("init_style"=>"tensor"), sort_idx="eta", xaxis=:log, label="cell: FacMAGRU (init: tnsr, nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=3, z=1, color=cell_colors["FacMAGRU"], title="MUE, Episodes", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	plt
end

# ╔═╡ 2bdc8606-ab44-47fe-a722-c10d966e4a1c
let
	plt = plot()
	plt = plot!(
		  data_facmagru64,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["GRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru100,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["GRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru152,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=3, z=1, color=cell_colors["GRU"], title="MUE, Steps: 4M", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru64,
		  Dict("init_style"=>"tensor"), label="cell: FacMAGRU (init: tnsr, nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru100,
		  Dict("init_style"=>"tensor"), label="cell: FacMAGRU (init: tnsr, nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru152,
		  Dict("init_style"=>"tensor"), label="cell: FacMAGRU (init: tnsr, nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=3, z=1, color=cell_colors["FacMAGRU"], title="MUE, Episodes", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	plt
end

# ╔═╡ 078b38a3-d94c-431b-86c2-2f67c46e6d28
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

# ╔═╡ 60cf1891-d047-42f5-b88d-10179df74087
begin 
	data_magru_steps = RPU.get_line_data_for(
	ic_magru,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_facmagru64_steps = RPU.get_line_data_for(
	ic_facmagru64,
	["init_style"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_facmagru100_steps = RPU.get_line_data_for(
	ic_facmagru100,
	["init_style"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_facmagru152_steps = RPU.get_line_data_for(
	ic_facmagru152,
	["init_style"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
end

# ╔═╡ 81a19e1d-4162-4c6a-80d9-523a163abeba
begin 
	data_facmagru64_steps_f = RPU.get_line_data_for(
	ic_facmagru64_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_facmagru100_steps_f = RPU.get_line_data_for(
	ic_facmagru100_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_facmagru152_steps_f = RPU.get_line_data_for(
	ic_facmagru152_f,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
end

# ╔═╡ 8b3e6f97-8fb5-463b-b43a-9b837ac4fac6
let
	plt = plot()

	plt = plot!(
		  data_facmagru64_steps,
		  Dict("init_style"=>"tensor"), label="cell: FacMAGRU (init: tnsr, nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru100_steps,
		  Dict("init_style"=>"tensor"), label="cell: FacMAGRU (init: tnsr, nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru152_steps,
		  Dict("init_style"=>"tensor"), label="cell: FacMAGRU (init: tnsr, nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 200), lw=3, z=1, color=cell_colors["FacMAGRU"], title="MUE, Steps: 4M", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru64_steps,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["AAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru100_steps,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["AAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru152_steps,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 200), lw=3, z=1, color=cell_colors["AAGRU"], title="MUE, Steps", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	plt
end

# ╔═╡ 7f16c499-6bee-4f5f-87fc-7a6f6ec5b7f6
begin
	ic_dir_os6_sc2_gru_4M_relu_final, dd_dir_os6_sc2_gru_4M_relu_final = RPU.load_data("../../local_data/LL_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_gru_4M/")
	ic_dir_os6_sc2_aagru_4M_relu_final, dd_dir_os6_sc2_aagru_4M_relu_final = RPU.load_data("../../local_data/LL_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_aagru_4M/")
	ic_dir_os6_sc2_magru_4M_relu_final, dd_dir_os6_sc2_magru_4M_relu_final = RPU.load_data("../../local_data/LL_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_magru_4M/")
	ic_dir_os6_sc2_facmagru152_4M_relu_final, dd_dir_os6_sc2_facmagru152_4M_relu_final = RPU.load_data("../../local_data/LL_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru152_4M/")
	ic_dir_os6_sc2_facmagru100_4M_relu_final, dd_dir_os6_sc2_facmagru100_4M_relu_final = RPU.load_data("../../local_data/LL_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru100_4M/")
	ic_dir_os6_sc2_facmagru64_4M_relu_final, dd_dir_os6_sc2_facmagru64_4M_relu_final = RPU.load_data("../../local_data/LL_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru64_4M/")
end

# ╔═╡ 021b3edc-6926-44af-a47f-d3c43a8c4219
begin
	ic_dir_os6_sc2_gru_4M_relu, dd_dir_os6_sc2_gru_4M_relu = RPU.load_data("../../local_data/LL_relu/lunar_lander_er_relu_rmsprop_os6_sc2_gru_4M/")
	ic_dir_os6_sc2_aagru_4M_relu, dd_dir_os6_sc2_aagru_4M_relu = RPU.load_data("../../local_data/LL_relu/lunar_lander_er_relu_rmsprop_os6_sc2_aagru_4M/")
	ic_dir_os6_sc2_magru_4M_relu, dd_dir_os6_sc2_magru_4M_relu = RPU.load_data("../../local_data/LL_relu/lunar_lander_er_relu_rmsprop_os6_sc2_magru_4M/")
	ic_dir_os6_sc2_facmagru152_4M_relu, dd_dir_os6_sc2_facmagru152_4M_relu = RPU.load_data("../../local_data/LL_relu/lunar_lander_er_relu_rmsprop_os6_sc2_facmagru152_4M/")
	ic_dir_os6_sc2_facmagru100_4M_relu, dd_dir_os6_sc2_facmagru100_4M_relu = RPU.load_data("../../local_data/LL_relu/lunar_lander_er_relu_rmsprop_os6_sc2_facmagru100_4M/")
	ic_dir_os6_sc2_facmagru64_4M_relu, dd_dir_os6_sc2_facmagru64_4M_relu = RPU.load_data("../../local_data/LL_relu/lunar_lander_er_relu_rmsprop_os6_sc2_facmagru64_4M/")
end

# ╔═╡ 6c672630-ccae-4934-a9fd-3139ac7740f4
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

# ╔═╡ 1607dbff-3b54-4963-aa2d-a72f769f261b
let
	plt = plot()
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["MAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["AAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_f64_4M_relu_final,
		  Dict(), label="cell: FacMAGRU (init: paper, nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_f100_4M_relu_final,
		  Dict(), label="cell: FacMAGRU (init: paper, nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final,
		  Dict(), label="cell: FacMAGRU (init: paper, nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=3, z=1, color=cell_colors["FacMAGRU"], title="MUE, Episodes", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	plt
end

# ╔═╡ f7291063-8ec1-4c78-8a3e-9223adeb3945
let
	plt = plot()
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["MAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["AAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru64,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru100,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru152,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=3, z=1, color=cell_colors["FacMAGRU"], title="MUE, Episodes", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	plt
end

# ╔═╡ f0987fbd-5afe-4298-997b-38646fbea411
let
	plt = plot()
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["MAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["AAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru64,
		  Dict("init_style"=>"tensor"), label="cell: FacMAGRU (init: tnsr, nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru100,
		  Dict("init_style"=>"tensor"), label="cell: FacMAGRU (init: tnsr, nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru152,
		  Dict("init_style"=>"tensor"), label="cell: FacMAGRU (init: tnsr, nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=3, z=1, color=cell_colors["FacMAGRU"], title="MUE, Episodes", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	plt
end

# ╔═╡ 6058ef88-6f34-4a35-b548-b9e3b9044ddb
begin 
	data_opt_sc2_g_4M_relu_final_steps_ = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 2000))
	
	data_opt_sc2_a_4M_relu_final_steps_ = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 2000))
	
	data_opt_sc2_m_4M_relu_final_steps_ = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_relu_final,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 2000))
end

# ╔═╡ 279930b5-ad7e-45c9-a8f7-3cc595a95dbb
begin 
	data_opt_sc2_g_4M_relu_final_steps = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_a_4M_relu_final_steps = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_m_4M_relu_final_steps = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_relu_final,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_f152_4M_relu_final_steps = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru152_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_f100_4M_relu_final_steps = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru100_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
	
	data_opt_sc2_f64_4M_relu_final_steps = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru64_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->get_extended_line(x, :total_rews, :total_steps, n=100000))
end

# ╔═╡ e37e6c4d-c6c9-4b34-8cf3-fd5e80e15584
let
	plt = plot()
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final_steps,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["MAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_steps,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["AAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_f64_4M_relu_final_steps,
		  Dict(), label="cell: FacMAGRU (init: paper, nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_f100_4M_relu_final_steps,
		  Dict(), label="cell: FacMAGRU (init: paper, nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final_steps,
		  Dict(), label="cell: FacMAGRU (init: paper, nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=3, z=1, color=cell_colors["FacMAGRU"], title="MUE, Steps", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	plt
end

# ╔═╡ b22e68a4-b53a-4a3e-bebd-9799b723d24d
let
	plt = plot()
	plt = plot!(
		  data_magru_steps,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["MAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_steps,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["AAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru64_steps,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru100_steps,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru152_steps,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 200), lw=3, z=1, color=cell_colors["FacMAGRU"], title="MUE, Steps", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	plt
end

# ╔═╡ 7fde25bc-537b-4650-81e4-00acc17e6a34
let
	plt = plot()
	plt = plot!(
		  data_magru_steps,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["MAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_steps,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["AAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru64_steps,
		  Dict("init_style"=>"tensor"), label="cell: FacMAGRU (init: tnsr, nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru100_steps,
		  Dict("init_style"=>"tensor"), label="cell: FacMAGRU (init: tnsr, nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru152_steps,
		  Dict("init_style"=>"tensor"), label="cell: FacMAGRU (init: tnsr, nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 200), lw=3, z=1, color=cell_colors["FacMAGRU"], title="MUE, Steps", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	plt
end

# ╔═╡ e7ba7431-5a2f-4c83-b70d-333858b1870c
begin 
	data_opt_sc2_g_4M_relu_final_mean = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_MEAN(x, :total_rews))
	
	data_opt_sc2_a_4M_relu_final_mean = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_MEAN(x, :total_rews))
	
	data_opt_sc2_m_4M_relu_final_mean = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_relu_final,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_MEAN(x, :total_rews))
	
	data_opt_sc2_f152_4M_relu_final_mean = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru152_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_MEAN(x, :total_rews))
	
	data_opt_sc2_f100_4M_relu_final_mean = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru100_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_MEAN(x, :total_rews))
	
	data_opt_sc2_f64_4M_relu_final_mean = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru64_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :total_rews),
	get_data=(x)->RPU.get_MEAN(x, :total_rews))
end

# ╔═╡ d09c4e81-8dfe-47cc-9348-56c8d03609d9
begin 
	data_opt_sc2_g_4M_relu_final_mue = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_opt_sc2_a_4M_relu_final_mue = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_opt_sc2_m_4M_relu_final_mue = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_relu_final,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_opt_sc2_f152_4M_relu_final_mue = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru152_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_opt_sc2_f100_4M_relu_final_mue = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru100_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_opt_sc2_f64_4M_relu_final_mue = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru64_4M_relu_final,
	[],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
end

# ╔═╡ 104a0852-eadb-4436-9a0c-c8147e47796c
begin 
	data_opt_sc2_g_4M_relu_mue = RPU.get_line_data_for(
	ic_dir_os6_sc2_gru_4M_relu,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_opt_sc2_a_4M_relu_mue = RPU.get_line_data_for(
	ic_dir_os6_sc2_aagru_4M_relu,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_opt_sc2_m_4M_relu_mue = RPU.get_line_data_for(
	ic_dir_os6_sc2_magru_4M_relu,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_opt_sc2_f152_4M_relu_mue = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru152_4M_relu,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_opt_sc2_f100_4M_relu_mue = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru100_4M_relu,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_opt_sc2_f64_4M_relu_mue = RPU.get_line_data_for(
	ic_dir_os6_sc2_facmagru64_4M_relu,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
end

# ╔═╡ ca69f424-032a-4ec5-a4e9-32ecc76cf287
let
	plt = plot()

	plt = plot!(
		  data_opt_sc2_f64_4M_relu_final,
		  Dict(), label="cell: FacMAGRU (init: paper, nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_f100_4M_relu_final,
		  Dict(), label="cell: FacMAGRU (init: paper, nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_f152_4M_relu_final,
		  Dict(), label="cell: FacMAGRU (init: paper, nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 200), lw=3, z=1, color=cell_colors["FacMAGRU"], title="MUE, Steps: 4M", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru64,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["AAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru100,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["AAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru152,
		  Dict("init_style"=>"standard"), label="cell: FacMAGRU (init: std, nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=3, z=1, color=cell_colors["AAGRU"], title="MUE, Episodes", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	plt
end

# ╔═╡ 15d3bb02-1156-45a1-9a83-0ce163ac182d
let
	plt = plot()

	plt = plot!(
		  data_opt_sc2_m_4M_relu_final,
		  Dict(), label="cell: FacMAGRU (paper, nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["MAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_magru,
		  Dict(), label="cell: FacMAGRU (new, nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["AAGRU"], fillalpha=0.2)
	
	plt
end

# ╔═╡ 7e9060d3-43f4-401d-ba6f-ab84e251ed33
let
	plt = plot()
	plt = plot!(
		  data_opt_sc2_g_4M_relu_final_steps_,
		  Dict(), label="cell: GRU (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["GRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_steps_,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["AAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_m_4M_relu_final_steps_,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(150, 500), lw=3, z=1, color=cell_colors["MAGRU"], grid=false, tickdir=:out, fillalpha=0.2, ylabel="Steps", xlabel="Episode")
	plt
	
	#savefig("../../data/paper_plots/factored_tensor/lunar_lander_steps_vs_eps_reg.pdf")
end

# ╔═╡ ae9147b0-12ba-4242-b9f3-3ed5005d07c8
let
	plt = plot()
	plt = plot!(
		  data_facmagru64_f_steps,
		  Dict(), label="cell: FacMAGRU (nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru100_f_steps,
		  Dict(), label="cell: FacMAGRU (nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru152_f_steps,
		  Dict(), label="cell: FacMAGRU (nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(150, 500), lw=3, z=1, color=cell_colors["FacMAGRU"], grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2, ylabel="Steps", xlabel="Episode")
	plt
	
	#savefig("../../data/paper_plots/factored_tensor/lunar_lander_steps_vs_eps_fac.pdf")
end

# ╔═╡ 1d176c89-6526-430c-bbbe-a302f0435e42
let
	plt = plot()
	plt = plot!(
		  data_facmagru64_f,
		  Dict(), label="cell: FacMAGRU (nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru100_f,
		  Dict(), label="cell: FacMAGRU (nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru152_f,
		  Dict(), label="cell: FacMAGRU (nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=3, z=1, color=cell_colors["FacMAGRU"], grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2, ylabel="Return", xlabel="Episode")
	plt
	
	#savefig("../../data/paper_plots/factored_tensor/lunar_lander_return_vs_eps_fac.pdf")
end

# ╔═╡ df91ee5e-e6a2-4f34-8beb-fe3b4c01cb04
let	
	plt = violin(data_opt_sc2_g_4M_relu_final_mean, Dict(), label="GRU", legend=false, color=cell_colors["GRU"], lw=2, linecolor=cell_colors["GRU"])
		
	plt = boxplot!(data_opt_sc2_g_4M_relu_final_mean, Dict(), label="GRU", color=color=cell_colors["GRU"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Average Return", ylims=(-150, 50))
		
	#plt = dotplot!(data_opt_sc2_g_4M_relu_final_mean, Dict(), label="GRU", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(-150, 50))
	
	plt = violin!(data_opt_sc2_a_4M_relu_final_mean, Dict(), label="AAGRU", legend=false, color=cell_colors["AAGRU"], lw=2, linecolor=cell_colors["AAGRU"])
		
	plt = boxplot!(data_opt_sc2_a_4M_relu_final_mean, Dict(), label="AAGRU", color=color=cell_colors["AAGRU"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Average Return", ylims=(-150, 50))
		
	#plt = dotplot!(data_opt_sc2_a_4M_relu_final_mean, Dict(), label="AAGRU", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(-150, 50))
	
	plt = violin!(data_opt_sc2_m_4M_relu_final_mean, Dict(), label="MAGRU", legend=false, color=cell_colors["MAGRU"], lw=2, linecolor=cell_colors["MAGRU"])
		
	plt = boxplot!(data_opt_sc2_m_4M_relu_final_mean, Dict(), label="MAGRU", color=color=cell_colors["MAGRU"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Average Return", ylims=(-150, 50))
		
	#plt = dotplot!(data_opt_sc2_m_4M_relu_final_mean, Dict(), label="MAGRU", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(-150, 50))
	
	plt = violin!(data_facmagru64_fm, Dict(), label="FacGRU 64", legend=false, color=cell_colors["FacMAGRU"], lw=2, linecolor=cell_colors["FacMAGRU"])
		
	plt = boxplot!(data_facmagru64_fm, Dict(), label="FacGRU 64", color=color=cell_colors["FacMAGRU"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Average Return", ylims=(-150, 50))
		
	#plt = dotplot!(data_facmagru64_fm, Dict(), label="FacGRU 64", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(-150, 50))
	
	plt = violin!(data_facmagru100_fm, Dict(), label="FacGRU 100", legend=false, color=cell_colors["FacMAGRU"], lw=2, linecolor=cell_colors["FacMAGRU"])
		
	plt = boxplot!(data_facmagru100_fm, Dict(), label="FacGRU 100", color=color=cell_colors["FacMAGRU"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Average Return", ylims=(-150, 50))
		
	#plt = dotplot!(data_facmagru100_fm, Dict(), label="FacGRU 64", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(-150, 50))
	
	plt = violin!(data_facmagru152_fm, Dict(), label="FacGRU 152", legend=false, color=cell_colors["FacMAGRU"], lw=2, linecolor=cell_colors["FacMAGRU"])
		
	plt = boxplot!(data_facmagru152_fm, Dict(), label="FacGRU 152", color=color=cell_colors["FacMAGRU"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=9, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Average Return", ylims=(-150, 50))
		
	#plt = dotplot!(data_facmagru152_fm, Dict(), label="FacGRU 64", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(-150, 50))
	
	plt
	
	#savefig("../../data/paper_plots/factored_tensor/lunar_lander_box_plot_mean.pdf")
end

# ╔═╡ 2fb8b94a-b53d-4c9f-85fa-3170dfd9198a
let	
	plt = violin(data_opt_sc2_g_4M_relu_final_mue, Dict(), label="GRU", legend=false, color=cell_colors["GRU"], lw=2, linecolor=cell_colors["GRU"])
		
	plt = boxplot!(data_opt_sc2_g_4M_relu_final_mue, Dict(), label="GRU", color=color=cell_colors["GRU"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Average Return", ylims=(-150, 50))
		
	#plt = dotplot!(data_opt_sc2_g_4M_relu_final_mue, Dict(), label="GRU", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(-150, 50))
	
	plt = violin!(data_opt_sc2_a_4M_relu_final_mue, Dict(), label="AAGRU", legend=false, color=cell_colors["AAGRU"], lw=2, linecolor=cell_colors["AAGRU"])
		
	plt = boxplot!(data_opt_sc2_a_4M_relu_final_mue, Dict(), label="AAGRU", color=color=cell_colors["AAGRU"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Average Return", ylims=(-150, 50))
		
	#plt = dotplot!(data_opt_sc2_a_4M_relu_final_mue, Dict(), label="AAGRU", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(-150, 50))
	
	plt = violin!(data_opt_sc2_m_4M_relu_final_mue, Dict(), label="MAGRU", legend=false, color=cell_colors["MAGRU"], lw=2, linecolor=cell_colors["MAGRU"])
		
	plt = boxplot!(data_opt_sc2_m_4M_relu_final_mue, Dict(), label="MAGRU", color=color=cell_colors["MAGRU"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Average Return", ylims=(-150, 50))
		
	#plt = dotplot!(data_opt_sc2_m_4M_relu_final_mue, Dict(), label="MAGRU", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(-150, 50))
	
	plt = violin!(data_facmagru64_fmue, Dict(), label="FacGRU 64", legend=false, color=cell_colors["FacMAGRU"], lw=2, linecolor=cell_colors["FacMAGRU"])
		
	plt = boxplot!(data_facmagru64_fmue, Dict(), label="FacGRU 64", color=color=cell_colors["FacMAGRU"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Average Return", ylims=(-150, 50))
		
	#plt = dotplot!(data_facmagru64_fmue, Dict(), label="FacGRU 64", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(-150, 50))
	
	plt = violin!(data_facmagru100_fmue, Dict(), label="FacGRU 100", legend=false, color=cell_colors["FacMAGRU"], lw=2, linecolor=cell_colors["FacMAGRU"])
		
	plt = boxplot!(data_facmagru100_fmue, Dict(), label="FacGRU 100", color=color=cell_colors["FacMAGRU"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Average Return", ylims=(-150, 50))
		
	#plt = dotplot!(data_facmagru100_fmue, Dict(), label="FacGRU 64", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(-150, 50))
	
	plt = violin!(data_facmagru152_fmue, Dict(), label="FacGRU 152", legend=false, color=cell_colors["FacMAGRU"], lw=2, linecolor=cell_colors["FacMAGRU"])
		
	plt = boxplot!(data_facmagru152_fmue, Dict(), label="FacGRU 152", color=color=cell_colors["FacMAGRU"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=9, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Avg Return (last 10% episodes)", ylims=(-75, 175))
		
	#plt = dotplot!(data_facmagru152_fmue, Dict(), label="FacGRU 64", color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(-150, 50))
	
	plt
	
	#savefig("../../data/paper_plots/factored_tensor/lunar_lander_box_plot_mue.pdf")
end

# ╔═╡ acdb2475-acdb-4078-9997-e611e021456d
let
	plt = plot()

	plt = plot!(
		  data_opt_sc2_g_4M_relu_mue,
		  Dict(), sort_idx="eta", xaxis=:log, label="cell: GRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["GRU"], fillalpha=0.2)
	plt = plot!(
		  data_opt_sc2_a_4M_relu_mue,
		  Dict(), sort_idx="eta", xaxis=:log, label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["AAGRU"], fillalpha=0.2)
	plt = plot!(
		  data_opt_sc2_m_4M_relu_mue,
		  Dict(), sort_idx="eta", xaxis=:log, label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["MAGRU"], fillalpha=0.2)
	plt = plot!(
		  data_facmagru64_sens,
		  Dict("init_style"=>"tensor"), sort_idx="eta", xaxis=:log, label="cell: FacMAGRU (init: tnsr, nh: 64, fac: 380)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru100_sens,
		  Dict("init_style"=>"tensor"), sort_idx="eta", xaxis=:log, label="cell: FacMAGRU (init: tnsr, nh: 100, fac: 265)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru152_sens,
		  Dict("init_style"=>"tensor"), sort_idx="eta", xaxis=:log, label="cell: FacMAGRU (init: tnsr, nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:topright, ylim=(-350, 400), lw=3, z=1, color=cell_colors["FacMAGRU"], grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2, xlabel="Learning Rate", ylabel="Avg Return (last 10% episodes)")
	plt
	
	#savefig("../../data/paper_plots/factored_tensor/lunar_lander_lr_sensitivity.pdf")
end

# ╔═╡ 81bfea3a-28d0-4002-b040-d3e0474944f6
let
	plt = plot()

	plt = plot!(
		  data_opt_sc2_g_4M_relu_final_steps,
		  Dict(), label="cell: GRU (nh: 154)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["GRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_opt_sc2_a_4M_relu_final_steps,
		  Dict(), label="cell: AAGRU (nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["AAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_magru_steps,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["MAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_facmagru152_steps_f,
		  Dict(), label="cell: FacMAGRU (nh: 152, fac: 170)",
			  palette=RPU.custom_colorant, legend=:none, ylim=(-300, 175), lw=3, z=1, color=cell_colors["FacMAGRU"], grid=false, tickdir=:out, fillalpha=0.2)
	plt
	
	#savefig("../../data/paper_plots/factored_tensor/lunar_lander_return_vs_steps.pdf")
end

# ╔═╡ Cell order:
# ╠═ae353392-e006-11eb-0aad-03cc4a895756
# ╠═3ae22ee3-2773-4748-a154-d5da599b7428
# ╠═dc5aaa9e-bedf-4ce7-8dee-ddeafd101831
# ╟─fee73b2f-0fbd-4314-91a3-783c63c5a321
# ╟─b229fe0f-f8aa-4dd8-a19f-68a9e8cb83e0
# ╠═5b94a3d6-d829-462b-a48e-e13ef754d8cd
# ╠═2a0b7edb-57b5-4dca-bb74-472b9592af97
# ╠═476cfeaa-864d-4213-ace4-b82a6a8fde59
# ╠═313041e3-05da-49e2-96d7-1aa080b22606
# ╠═6b5557a1-6b5f-4c59-80d7-7003541a934d
# ╠═18f8651e-8686-4d71-99aa-3be8db6303a2
# ╠═85803271-69dc-483e-8115-dffb07910066
# ╠═00d5ab33-7583-40d0-ad61-4fce6a289bee
# ╟─037436b3-37a6-4734-9aea-cbb7185274c4
# ╠═aad533aa-4b7b-40e3-8ebb-320eb3108426
# ╟─1607dbff-3b54-4963-aa2d-a72f769f261b
# ╟─f7291063-8ec1-4c78-8a3e-9223adeb3945
# ╠═f0987fbd-5afe-4298-997b-38646fbea411
# ╟─2bdc8606-ab44-47fe-a722-c10d966e4a1c
# ╟─078b38a3-d94c-431b-86c2-2f67c46e6d28
# ╠═60cf1891-d047-42f5-b88d-10179df74087
# ╠═81a19e1d-4162-4c6a-80d9-523a163abeba
# ╠═e37e6c4d-c6c9-4b34-8cf3-fd5e80e15584
# ╟─b22e68a4-b53a-4a3e-bebd-9799b723d24d
# ╠═7fde25bc-537b-4650-81e4-00acc17e6a34
# ╟─8b3e6f97-8fb5-463b-b43a-9b837ac4fac6
# ╠═7f16c499-6bee-4f5f-87fc-7a6f6ec5b7f6
# ╠═021b3edc-6926-44af-a47f-d3c43a8c4219
# ╠═6c672630-ccae-4934-a9fd-3139ac7740f4
# ╠═6058ef88-6f34-4a35-b548-b9e3b9044ddb
# ╠═279930b5-ad7e-45c9-a8f7-3cc595a95dbb
# ╠═e7ba7431-5a2f-4c83-b70d-333858b1870c
# ╠═d09c4e81-8dfe-47cc-9348-56c8d03609d9
# ╠═104a0852-eadb-4436-9a0c-c8147e47796c
# ╟─ca69f424-032a-4ec5-a4e9-32ecc76cf287
# ╟─15d3bb02-1156-45a1-9a83-0ce163ac182d
# ╠═7e9060d3-43f4-401d-ba6f-ab84e251ed33
# ╠═ae9147b0-12ba-4242-b9f3-3ed5005d07c8
# ╠═1d176c89-6526-430c-bbbe-a302f0435e42
# ╠═df91ee5e-e6a2-4f34-8beb-fe3b4c01cb04
# ╠═2fb8b94a-b53d-4c9f-85fa-3170dfd9198a
# ╠═acdb2475-acdb-4078-9997-e611e021456d
# ╠═81bfea3a-28d0-4002-b040-d3e0474944f6
