### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 1fc0b3d6-e766-11eb-04fc-4925cf93f79c
using Revise

# ╔═╡ 4629ef23-fd40-45e7-bdd8-bbc49da085c5
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 3fc6d692-0358-4de6-ad85-57e1d9caa7c6
const RPU = ReproducePlotUtils

# ╔═╡ 368d475a-0815-4539-b799-177db29569e3
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

# ╔═╡ e6ee8774-a638-4df2-a900-54710a9e0c0c
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2],
	"FacTucMAGRU" => color_scheme[4])

# ╔═╡ c8f07f8a-c67f-42d3-b1b2-1c48b0567b77
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ 46751cfb-e2c1-4d03-bfb6-6a130855aa01
begin
	ic_magru, dd_magru = RPU.load_data("../../local_data/lunar_lander_er_relu_rmsprop_os6_sc2_magru_4M_wr/")
	ic_facmagru64, dd_facmagru64 = RPU.load_data("../../local_data/factored_tensor/lunar_lander_er_relu_rmsprop_os6_sc2_facmagru64_4M_wr/")
	ic_facmagru100, dd_facmagru100 = RPU.load_data("../../local_data/factored_tensor/lunar_lander_er_relu_rmsprop_os6_sc2_facmagru100_4M_wr/")
	ic_facmagru152, dd_facmagru152 = RPU.load_data("../../local_data/factored_tensor/lunar_lander_er_relu_rmsprop_os6_sc2_facmagru152_4M_wr/")
end

# ╔═╡ 3f79981b-0081-4068-abd9-f5c4d2bfcc9e
begin
	ic_factucmagru64, dd_factucmagru64 = RPU.load_data("../../local_data/tucker_decomp/lunar_lander_er_relu_rmsprop_os6_sc2_factucmagru64_4M/")
	ic_factucmagru100, dd_factucmagru100 = RPU.load_data("../../local_data/tucker_decomp/lunar_lander_er_relu_rmsprop_os6_sc2_factucmagru100_4M/")
	ic_factucmagru152, dd_factucmagru152 = RPU.load_data("../../local_data/tucker_decomp/lunar_lander_er_relu_rmsprop_os6_sc2_factucmagru152_4M/")
end

# ╔═╡ c0ef08db-c2c6-43d6-9fb9-1a6dc36665ef
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

# ╔═╡ b9db33cb-afde-4fd9-97e1-39c53b3241d8
begin 
	data_factucmagru64 = RPU.get_line_data_for(
	ic_factucmagru64,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_factucmagru100 = RPU.get_line_data_for(
	ic_factucmagru100,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
	
	data_factucmagru152 = RPU.get_line_data_for(
	ic_factucmagru152,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_rews, 2000))
end

# ╔═╡ e1fe424c-89f2-4638-9d90-34c228c28ca9
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

# ╔═╡ 37a7992a-225d-44ac-8553-09277b88b544
begin 
	data_factucmagru64_sens = RPU.get_line_data_for(
	ic_factucmagru64,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_factucmagru100_sens = RPU.get_line_data_for(
	ic_factucmagru100,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
	
	data_factucmagru152_sens = RPU.get_line_data_for(
	ic_factucmagru152,
	["eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews),
	get_data=(x)->RPU.get_MUE(x, :total_rews))
end

# ╔═╡ 9615ed3f-1fb4-45a5-8e2a-a02e507acee8
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

# ╔═╡ 9e35387a-49e0-4373-922e-9b72c556b043
let
	plt = plot()
	
	plt = plot!(
		  data_factucmagru64_sens,
		  Dict(), sort_idx="eta", xaxis=:log, label="cell: FacTucMAGRU (init: tnsr, nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacTucMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_factucmagru100_sens,
		  Dict(), sort_idx="eta", xaxis=:log, label="cell: FacTucMAGRU (init: tnsr, nh: 100)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacTucMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_factucmagru152_sens,
		  Dict(), sort_idx="eta", xaxis=:log, label="cell: FacTucMAGRU (init: tnsr, nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=3, z=1, color=cell_colors["FacTucMAGRU"], title="MUE, Episodes", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	plt
end

# ╔═╡ 922e82ef-02b3-4ab2-8f99-be5a90270a7b
let
	plt = plot()
	plt = plot!(
		  data_magru,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["MAGRU"], fillalpha=0.2)
	
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
	
	plt = plot!(
		  data_factucmagru64,
		  Dict(), label="cell: FacTucMAGRU (init: tnsr, nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacTucMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_factucmagru100,
		  Dict(), label="cell: FacTucMAGRU (init: tnsr, nh: 100)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacTucMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_factucmagru152,
		  Dict(), label="cell: FacTucMAGRU (init: tnsr, nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=3, z=1, color=cell_colors["FacTucMAGRU"], title="LunarLander MUE, Episodes", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	
	plt
	
	savefig("../../data/paper_plots/tucker_decomp/lunar_lander_tuc.pdf")
end

# ╔═╡ c18b3edb-7a41-4c5a-8d76-52507d3dee36
let
	plt = plot()
	plt = plot!(
		  data_magru,
		  Dict(), label="cell: MAGRU (nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["MAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_factucmagru64,
		  Dict(), label="cell: FacTucMAGRU (init: tnsr, nh: 64)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacTucMAGRU"], fillalpha=0.2)
	
	plt = plot!(
		  data_factucmagru100,
		  Dict(), label="cell: FacTucMAGRU (init: tnsr, nh: 100)",
			  palette=RPU.custom_colorant, legend=:bottomright, lw=3, z=1, color=cell_colors["FacTucMAGRU"], linestyle=:dot, fillalpha=0.2)
	
	plt = plot!(
		  data_factucmagru152,
		  Dict(), label="cell: FacTucMAGRU (init: tnsr, nh: 152)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(-300, 150), lw=3, z=1, color=cell_colors["FacTucMAGRU"], title="Lunar Lander MUE, Episodes", grid=false, tickdir=:out, linestyle=:dash, fillalpha=0.2)
	plt
end

# ╔═╡ Cell order:
# ╠═1fc0b3d6-e766-11eb-04fc-4925cf93f79c
# ╠═4629ef23-fd40-45e7-bdd8-bbc49da085c5
# ╠═3fc6d692-0358-4de6-ad85-57e1d9caa7c6
# ╟─368d475a-0815-4539-b799-177db29569e3
# ╠═e6ee8774-a638-4df2-a900-54710a9e0c0c
# ╠═c8f07f8a-c67f-42d3-b1b2-1c48b0567b77
# ╠═46751cfb-e2c1-4d03-bfb6-6a130855aa01
# ╠═3f79981b-0081-4068-abd9-f5c4d2bfcc9e
# ╠═c0ef08db-c2c6-43d6-9fb9-1a6dc36665ef
# ╠═b9db33cb-afde-4fd9-97e1-39c53b3241d8
# ╠═e1fe424c-89f2-4638-9d90-34c228c28ca9
# ╠═37a7992a-225d-44ac-8553-09277b88b544
# ╠═9615ed3f-1fb4-45a5-8e2a-a02e507acee8
# ╠═9e35387a-49e0-4373-922e-9b72c556b043
# ╠═922e82ef-02b3-4ab2-8f99-be5a90270a7b
# ╠═c18b3edb-7a41-4c5a-8d76-52507d3dee36
