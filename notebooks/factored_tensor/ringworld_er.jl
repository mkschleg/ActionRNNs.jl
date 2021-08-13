### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ f31161ce-ee85-11eb-0585-b1d35cd8086b
using Revise, Reproduce, ReproducePlotUtils, Plots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto, JLD2

# ╔═╡ 873c321d-cccc-4f9b-92fb-87164bb71e6f
const RPU = ReproducePlotUtils

# ╔═╡ 7d59ca83-6196-427c-a61f-1b78cb93f979
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

# ╔═╡ ef324f02-7ee3-46a0-b68e-7e707c1698f3
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ 62061b03-3181-43d4-8cae-1b5ec0b0c6a6
ic, dd = RPU.load_data("../../local_data/factored_tensor/ringworld_er_rmsprop_10/")

# ╔═╡ ffa355d4-e5d5-4fca-8c23-c7d9854df8da
data = RPU.get_line_data_for(
	ic,
	["cell", "numhidden", "factors", "truncation"],
	["eta"];
	comp=findmin,
	get_comp_data=(x)->RPU.get_AUC(x, "end"),
	get_data=(x)->RPU.get_AUC(x, "end"))

# ╔═╡ 9c6abca3-5692-45cb-b63f-b3f8a7aaaa40
let
	plt = plot()
	
	args_list = [
		Dict("cell"=>"RNN", "numhidden"=> 3),
		Dict("cell"=>"RNN", "numhidden"=> 6),
		Dict("cell"=>"RNN", "numhidden"=> 9),
		Dict("cell"=>"RNN", "numhidden"=> 12),
		Dict("cell"=>"RNN", "numhidden"=> 15),
		Dict("cell"=>"RNN", "numhidden"=> 17),
		Dict("cell"=>"RNN", "numhidden"=> 20),
	]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	for i in 1:length(args_list)
		plt = plot!(
			  data,
			  args_list[i], sort_idx="truncation", legendtitle="nh", label="$(args_list[i]["numhidden"])",
				  palette=RPU.custom_colorant, legend=:topright, lw=3, z=1, color=cell_colors[args_list[i]["cell"]], fillalpha=0.2, ylim=(0, 0.4), markershape=marker_shapes[i], markerstrokewidth=2, title="RNN", xtickfontsize=12, ytickfontsize=12)
	end
	plot!(size=(400,400))
	
	plt
	#savefig("../../data/paper_plots/factored_tensor/ringworld_er_sens_rnn.pdf")
end

# ╔═╡ d39338ae-8fd8-4dd7-bb07-a9e82960f9a3
let
	plt = plot()
	
	args_list = [
		Dict("cell"=>"AARNN", "numhidden"=> 3),
		Dict("cell"=>"AARNN", "numhidden"=> 6),
		Dict("cell"=>"AARNN", "numhidden"=> 9),
		Dict("cell"=>"AARNN", "numhidden"=> 12),
		Dict("cell"=>"AARNN", "numhidden"=> 15),
		Dict("cell"=>"AARNN", "numhidden"=> 17),
		Dict("cell"=>"AARNN", "numhidden"=> 20),
	]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	for i in 1:length(args_list)
		plt = plot!(
			  data,
			  args_list[i], sort_idx="truncation", legendtitle="nh", label="$(args_list[i]["numhidden"])",
				  palette=RPU.custom_colorant, legend=:topright, lw=3, z=1, color=cell_colors[args_list[i]["cell"]], fillalpha=0.2, ylim=(0, 0.4), markershape=marker_shapes[i], markerstrokewidth=2, title="AARNN", xtickfontsize=12, ytickfontsize=12)
	end
	plot!(size=(400,400))
	
	plt
	#savefig("../../data/paper_plots/factored_tensor/ringworld_er_sens_aarnn.pdf")
end

# ╔═╡ c5c25442-9ba4-47a1-aad1-1ca9b4540e60
let
	plt = plot()
	
	args_list = [
		Dict("cell"=>"MARNN", "numhidden"=> 3),
		Dict("cell"=>"MARNN", "numhidden"=> 6),
		Dict("cell"=>"MARNN", "numhidden"=> 9),
		Dict("cell"=>"MARNN", "numhidden"=> 12),
		Dict("cell"=>"MARNN", "numhidden"=> 15),
		Dict("cell"=>"MARNN", "numhidden"=> 17),
		Dict("cell"=>"MARNN", "numhidden"=> 20),
	]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	for i in 1:length(args_list)
		plt = plot!(
			  data,
			  args_list[i], sort_idx="truncation", legendtitle="nh", label="$(args_list[i]["numhidden"])",
				  palette=RPU.custom_colorant, legend=:topright, lw=3, z=1, color=cell_colors[args_list[i]["cell"]], fillalpha=0.2, ylim=(0, 0.4), markershape=marker_shapes[i], markerstrokewidth=2, title="MARNN", xtickfontsize=12, ytickfontsize=12)
	end
	plot!(size=(400,400))
	
	plt
	#savefig("../../data/paper_plots/factored_tensor/ringworld_er_sens_marnn.pdf")
end

# ╔═╡ 07f590a4-97a4-4090-9839-4891bc2a8fb0
let
	plt = plot()
	
	args_list = [
		Dict("cell"=>"FacMARNN", "numhidden"=> 3, "factors"=>12),
		Dict("cell"=>"FacMARNN", "numhidden"=> 6, "factors"=>12),
		Dict("cell"=>"FacMARNN", "numhidden"=> 9, "factors"=>12),
		Dict("cell"=>"FacMARNN", "numhidden"=> 12, "factors"=>12),
		Dict("cell"=>"FacMARNN", "numhidden"=> 15, "factors"=>12),
		Dict("cell"=>"FacMARNN", "numhidden"=> 17, "factors"=>12),
		Dict("cell"=>"FacMARNN", "numhidden"=> 20, "factors"=>12),
	]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	for i in 1:length(args_list)
		plt = plot!(
			  data,
			  args_list[i], sort_idx="truncation", legendtitle="nh", label="$(args_list[i]["numhidden"])",
				  palette=RPU.custom_colorant, legend=:topright, lw=3, z=1, color=cell_colors[args_list[i]["cell"]], fillalpha=0.2, ylim=(0, 0.4), markershape=marker_shapes[i], markerstrokewidth=2, title="FacMARNN: Params = AARNN", xtickfontsize=12, ytickfontsize=12)
	end
	plot!(size=(400,400))
	
	plt
	#savefig("../../data/paper_plots/factored_tensor/ringworld_er_sens_facmarnn_aarnn.pdf")
end

# ╔═╡ 9a6a656c-a09a-497e-9611-b66d4236b370
let
	plt = plot()
	
	args_list = [
		Dict("cell"=>"FacMARNN", "numhidden"=> 3, "factors"=>14),
		Dict("cell"=>"FacMARNN", "numhidden"=> 6, "factors"=>14),
		Dict("cell"=>"FacMARNN", "numhidden"=> 9, "factors"=>14),
		Dict("cell"=>"FacMARNN", "numhidden"=> 12, "factors"=>14),
		Dict("cell"=>"FacMARNN", "numhidden"=> 15, "factors"=>14),
		Dict("cell"=>"FacMARNN", "numhidden"=> 17, "factors"=>14),
		Dict("cell"=>"FacMARNN", "numhidden"=> 20, "factors"=>14),
	]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	for i in 1:length(args_list)
		plt = plot!(
			  data,
			  args_list[i], sort_idx="truncation", legendtitle="nh", label="$(args_list[i]["numhidden"])",
				  palette=RPU.custom_colorant, legend=:topright, lw=3, z=1, color=cell_colors[args_list[i]["cell"]], fillalpha=0.2, ylim=(0, 0.4), markershape=marker_shapes[i], markerstrokewidth=2, title="FacMARNN: Params = MARNN", xtickfontsize=12, ytickfontsize=12)
	end
	plot!(size=(400,400))
	
	plt
	#savefig("../../data/paper_plots/factored_tensor/ringworld_er_sens_facmarnn_marnn.pdf")
end

# ╔═╡ f9027e4d-fae2-439c-911e-2c39e21b2e6f
let
	plt = plot()
	
	args_list = [
		Dict("cell"=>"GRU", "numhidden"=> 3),
		Dict("cell"=>"GRU", "numhidden"=> 6),
		Dict("cell"=>"GRU", "numhidden"=> 9),
		Dict("cell"=>"GRU", "numhidden"=> 12),
		Dict("cell"=>"GRU", "numhidden"=> 15),
		Dict("cell"=>"GRU", "numhidden"=> 17),
		Dict("cell"=>"GRU", "numhidden"=> 20),
	]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	for i in 1:length(args_list)
		plt = plot!(
			  data,
			  args_list[i], sort_idx="truncation", legendtitle="nh", label="$(args_list[i]["numhidden"])",
				  palette=RPU.custom_colorant, legend=:topright, lw=3, z=1, color=cell_colors[args_list[i]["cell"]], fillalpha=0.2, ylim=(0, 0.4), markershape=marker_shapes[i], markerstrokewidth=2, title="GRU", xtickfontsize=12, ytickfontsize=12)
	end
	plot!(size=(400,400))
	
	plt
	#savefig("../../data/paper_plots/factored_tensor/ringworld_er_sens_gru.pdf")
end

# ╔═╡ 9ddd35ad-cc68-4448-bc6e-baf13d7048c4
let
	plt = plot()
	
	args_list = [
		Dict("cell"=>"AAGRU", "numhidden"=> 3),
		Dict("cell"=>"AAGRU", "numhidden"=> 6),
		Dict("cell"=>"AAGRU", "numhidden"=> 9),
		Dict("cell"=>"AAGRU", "numhidden"=> 12),
		Dict("cell"=>"AAGRU", "numhidden"=> 15),
		Dict("cell"=>"AAGRU", "numhidden"=> 17),
		Dict("cell"=>"AAGRU", "numhidden"=> 20),
	]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	for i in 1:length(args_list)
		plt = plot!(
			  data,
			  args_list[i], sort_idx="truncation", legendtitle="nh", label="$(args_list[i]["numhidden"])",
				  palette=RPU.custom_colorant, legend=:topright, lw=3, z=1, color=cell_colors[args_list[i]["cell"]], fillalpha=0.2, ylim=(0, 0.4), markershape=marker_shapes[i], markerstrokewidth=2, title="AAGRU", xtickfontsize=12, ytickfontsize=12)
	end
	plot!(size=(400,400))
	
	plt
	#savefig("../../data/paper_plots/factored_tensor/ringworld_er_sens_aagru.pdf")
end

# ╔═╡ 2af2a195-e1cf-43b7-810a-06dbfcc17061
let
	plt = plot()
	
	args_list = [
		Dict("cell"=>"MAGRU", "numhidden"=> 3),
		Dict("cell"=>"MAGRU", "numhidden"=> 6),
		Dict("cell"=>"MAGRU", "numhidden"=> 9),
		Dict("cell"=>"MAGRU", "numhidden"=> 12),
		Dict("cell"=>"MAGRU", "numhidden"=> 15),
		Dict("cell"=>"MAGRU", "numhidden"=> 17),
		Dict("cell"=>"MAGRU", "numhidden"=> 20),
	]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	for i in 1:length(args_list)
		plt = plot!(
			  data,
			  args_list[i], sort_idx="truncation", legendtitle="nh", label="$(args_list[i]["numhidden"])",
				  palette=RPU.custom_colorant, legend=:topright, lw=3, z=1, color=cell_colors[args_list[i]["cell"]], fillalpha=0.2, ylim=(0, 0.4), markershape=marker_shapes[i], markerstrokewidth=2, title="MAGRU", xtickfontsize=12, ytickfontsize=12)
	end
	plot!(size=(400,400))
	
	plt
	#savefig("../../data/paper_plots/factored_tensor/ringworld_er_sens_magru.pdf")
end

# ╔═╡ dee15259-5402-47c2-9fa2-6072f140c930
let
	plt = plot()
	
	args_list = [
		Dict("cell"=>"FacMAGRU", "numhidden"=> 3, "factors"=>8),
		Dict("cell"=>"FacMAGRU", "numhidden"=> 6, "factors"=>8),
		Dict("cell"=>"FacMAGRU", "numhidden"=> 9, "factors"=>8),
		Dict("cell"=>"FacMAGRU", "numhidden"=> 12, "factors"=>8),
		Dict("cell"=>"FacMAGRU", "numhidden"=> 15, "factors"=>8),
		Dict("cell"=>"FacMAGRU", "numhidden"=> 17, "factors"=>8),
		Dict("cell"=>"FacMAGRU", "numhidden"=> 20, "factors"=>8),
	]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	for i in 1:length(args_list)
		plt = plot!(
			  data,
			  args_list[i], sort_idx="truncation", legendtitle="nh", label="$(args_list[i]["numhidden"])",
				  palette=RPU.custom_colorant, legend=:topright, lw=3, z=1, color=cell_colors[args_list[i]["cell"]], fillalpha=0.2, ylim=(0, 0.4), markershape=marker_shapes[i], markerstrokewidth=2, title="FacMAGRU: Params = AAGRU", xtickfontsize=12, ytickfontsize=12)
	end
	plot!(size=(400,400))
	
	plt
	#savefig("../../data/paper_plots/factored_tensor/ringworld_er_sens_facmagru_aagru.pdf")
end

# ╔═╡ 4ed743e9-59aa-4d43-bdb8-24d7ea6e8201
let
	plt = plot()
	
	args_list = [
		Dict("cell"=>"FacMAGRU", "numhidden"=> 3, "factors"=>12),
		Dict("cell"=>"FacMAGRU", "numhidden"=> 6, "factors"=>12),
		Dict("cell"=>"FacMAGRU", "numhidden"=> 9, "factors"=>12),
		Dict("cell"=>"FacMAGRU", "numhidden"=> 12, "factors"=>12),
		Dict("cell"=>"FacMAGRU", "numhidden"=> 15, "factors"=>12),
		Dict("cell"=>"FacMAGRU", "numhidden"=> 17, "factors"=>12),
		Dict("cell"=>"FacMAGRU", "numhidden"=> 20, "factors"=>12),
	]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	for i in 1:length(args_list)
		plt = plot!(
			  data,
			  args_list[i], sort_idx="truncation", legendtitle="nh", label="$(args_list[i]["numhidden"])",
				  palette=RPU.custom_colorant, legend=:topright, lw=3, z=1, color=cell_colors[args_list[i]["cell"]], fillalpha=0.2, ylim=(0, 0.4), markershape=marker_shapes[i], markerstrokewidth=2, title="FacMAGRU: Params = MAGRU", xtickfontsize=12, ytickfontsize=12)
	end
	plot!(size=(400,400))
	
	plt
	#savefig("../../data/paper_plots/factored_tensor/ringworld_er_sens_facmagru_magru.pdf")
end

# ╔═╡ d7647b91-f708-4260-a1e1-b4916e87717f
let
	plt = plot()
	
	args_list = [
		Dict("cell"=>"RNN", "numhidden"=> 15),
		Dict("cell"=>"AARNN", "numhidden"=> 15),
		Dict("cell"=>"MARNN", "numhidden"=> 12),
		Dict("cell"=>"FacMARNN", "numhidden"=> 12, "factors"=> 14),
		Dict("cell"=>"FacMARNN", "numhidden"=> 15, "factors"=> 12),
	]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	line_styles = [:solid, :solid, :solid, :solid, :dash]
	for i in 1:length(args_list)
		plt = plot!(
			  data,
			  args_list[i], sort_idx="truncation", legendtitle="nh", label="$(args_list[i]["numhidden"])",
				  palette=RPU.custom_colorant, legend=:none, lw=3, z=1, color=cell_colors[args_list[i]["cell"]], fillalpha=0.2, ylim=(0, 0.4), markershape=marker_shapes[i], markerstrokewidth=2, title="RNN Cells", xtickfontsize=12, ytickfontsize=12, linestyle=line_styles[i], grid=false, tickdir=:out)
	end
	
	plt
	#savefig("../../data/paper_plots/factored_tensor/ringworld_er_sens_cells_rnn.pdf")
end

# ╔═╡ 2a2c2f16-db82-4f77-a179-b3e89bd855a3
let
	plt = plot()
	
	args_list = [
		Dict("cell"=>"GRU", "numhidden"=> 12),
		Dict("cell"=>"AAGRU", "numhidden"=> 12),
		Dict("cell"=>"MAGRU", "numhidden"=> 9),
		Dict("cell"=>"FacMAGRU", "numhidden"=> 9, "factors"=> 12),
		Dict("cell"=>"FacMAGRU", "numhidden"=> 12, "factors"=> 8),
	]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	line_styles = [:solid, :solid, :solid, :solid, :dash]
	for i in 1:length(args_list)
		plt = plot!(
			  data,
			  args_list[i], sort_idx="truncation", legendtitle="nh", label="$(args_list[i]["numhidden"])",
				  palette=RPU.custom_colorant, legend=:none, lw=3, z=1, color=cell_colors[args_list[i]["cell"]], fillalpha=0.2, ylim=(0, 0.4), markershape=marker_shapes[i], markerstrokewidth=2, title="GRU Cells", xtickfontsize=12, ytickfontsize=12, linestyle=line_styles[i], grid=false, tickdir=:out)
	end
	
	plt
	#savefig("../../data/paper_plots/factored_tensor/ringworld_er_sens_cells_gru.pdf")
end

# ╔═╡ Cell order:
# ╠═f31161ce-ee85-11eb-0585-b1d35cd8086b
# ╠═873c321d-cccc-4f9b-92fb-87164bb71e6f
# ╟─7d59ca83-6196-427c-a61f-1b78cb93f979
# ╟─ef324f02-7ee3-46a0-b68e-7e707c1698f3
# ╠═62061b03-3181-43d4-8cae-1b5ec0b0c6a6
# ╠═ffa355d4-e5d5-4fca-8c23-c7d9854df8da
# ╟─9c6abca3-5692-45cb-b63f-b3f8a7aaaa40
# ╟─d39338ae-8fd8-4dd7-bb07-a9e82960f9a3
# ╟─c5c25442-9ba4-47a1-aad1-1ca9b4540e60
# ╟─07f590a4-97a4-4090-9839-4891bc2a8fb0
# ╟─9a6a656c-a09a-497e-9611-b66d4236b370
# ╟─f9027e4d-fae2-439c-911e-2c39e21b2e6f
# ╟─9ddd35ad-cc68-4448-bc6e-baf13d7048c4
# ╟─2af2a195-e1cf-43b7-810a-06dbfcc17061
# ╟─dee15259-5402-47c2-9fa2-6072f140c930
# ╟─4ed743e9-59aa-4d43-bdb8-24d7ea6e8201
# ╠═d7647b91-f708-4260-a1e1-b4916e87717f
# ╠═2a2c2f16-db82-4f77-a179-b3e89bd855a3
