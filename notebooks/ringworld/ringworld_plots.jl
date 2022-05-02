### A Pluto.jl notebook ###
# v0.19.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ f3cc35f4-1788-420e-bb20-4658eaec2eda
let
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ cf756452-bb40-11eb-1b8f-a1c2376756ed
using Revise

# ╔═╡ a14f2ee3-b61f-4d5f-8819-9a71f977c375
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 6607a779-d1b0-41b7-8b4f-69a83ef8d543
let
	using Random

	plt = plot(legend=nothing)
	for seed ∈ [1, 2, 3, 4]
		Random.seed!(seed)	
		x = rand(100)
		violin!(["$(seed)"], x)	
		boxplot!(["$(seed)"], x, fillalpha=0.75)
	end
	plt
end

# ╔═╡ 3052ee11-68ec-42ff-9177-5f70bce043b6
const RPU = ReproducePlotUtils

# ╔═╡ e1817bb9-5918-4ed4-a73d-8c650e5399bc
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

# ╔═╡ 4c5a2db1-1ed2-4b86-8917-79b55727d45d
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ 5062d5b8-3141-4c05-82d3-1bfba1951e68
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ ad0b6a6c-a062-4952-8632-616b2086bb5f
at(dir) = joinpath("../../local_data/ringworld/", dir)

# ╔═╡ 2148c874-e915-47af-9f67-6a565c23e9ff
ic_test, dd_test = RPU.load_data(at("ringworld_fac_er_rmsprop_10_marnn/"))

# ╔═╡ 4561c8c6-81a1-4966-915e-7b524135ae92
ic, dd = let
	ic1, dd = RPU.load_data(at("ringworld_fac_er_rmsprop_10/"))
	ic2, dd = RPU.load_data(at("ringworld_fac_er_rmsprop_10_marnn/"))
	ic3, dd = RPU.load_data(at("ringworld_fac_er_rmsprop_10_aagru/"))
	ic4, dd = RPU.load_data(at("ringworld_fac_er_rmsprop_10_magru/"))
	ic = ItemCollection([ic1.items; ic2.items; ic3.items; ic4.items])
	ic, diff(ic)
end

# ╔═╡ 34658923-9629-44c4-adc3-4694adebac1d
data_fac = RPU.get_line_data_for(
		ic,
		["numhidden", "truncation", "save_dir"],
		["eta"];
		comp=:min,
		get_comp_data=(x)->x["results"]["end"],
		get_data=(x)->x["results"]["end"])

# ╔═╡ 2f13e071-94ad-4023-8ea3-b0a14f4b885b
FileIO.load(joinpath(ic[1].folder_str, "results.jld2"))["results"]

# ╔═╡ a089dc38-fb29-4bc1-b012-76889709a991
# data = let
# 	sub_ic = search(ic, Dict("save_dir"=>"/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10/data"))
# 	RPU.get_line_data_for(
# 		sub_ic,
# 		["numhidden", "truncation"],
# 		["eta"];
# 		comp=:min,
# 		get_comp_data=(x)->x["results"]["end"],
# 		get_data=(x)->x["results"]["end"])
# end

# ╔═╡ 2a883704-29d7-4e43-894b-b0530cbbb6fc
# let
# 	plt = plot()
# 	# plot(data, Dict("numhidden"=>6))
# 	for τ = dd["numhidden"]
# 		plot!(data, Dict("numhidden"=>τ); label_idx="truncation", sort_idx="truncation", lw=2, palette=color_scheme, label=τ, labeltitle="Hidden", ylims=(0.0, 0.3))
# 	end
# 	plt
# end

# ╔═╡ 66796dd0-f0ad-4df8-a9c6-b7d810734e05
let
	dir = "/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10/data"
	# sub_ic = search(ic, Dict("save_dir"=>dir))
	# data = RPU.get_line_data_for(
	# 	sub_ic,
	# 	["numhidden", "truncation"],
	# 	["eta"];
	# 	comp=:min,
	# 	get_comp_data=(x)->x["results"]["end"],
	# 	get_data=(x)->x["results"]["end"])
	plt = plot()
	for τ = dd["numhidden"]
		plot!(data_fac, Dict("numhidden"=>τ, "save_dir"=>dir); label_idx="truncation", sort_idx="truncation", lw=2, palette=color_scheme, label=τ, labeltitle="Hidden", xlabel="truncation", ylims=(0.0, 0.3))
	end
	plt
end

# ╔═╡ cfa3f621-3d81-4e0b-903c-f78b67d3a3ff
let
	dir = "/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_marnn/data"
	plt = plot()
	for τ = dd["numhidden"]
		plot!(data_fac, Dict("numhidden"=>τ, "save_dir"=>dir); label_idx="truncation", sort_idx="truncation", lw=2, palette=color_scheme, label=τ, labeltitle="Hidden", xlabel="truncation", ylims=(0.0, 0.3))
	end
	plt
end

# ╔═╡ d0eebe70-fb0e-4e82-aa7d-4b0fadb8dbbf
let
	dir = "/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_aagru/data"
	plt = plot()
	for τ = dd["numhidden"]
		plot!(data_fac, Dict("numhidden"=>τ, "save_dir"=>dir); label_idx="truncation", sort_idx="truncation", lw=2, palette=color_scheme, label=τ, labeltitle="Hidden", xlabel="truncation", ylims=(0.0, 0.3))
	end
	plt
end

# ╔═╡ 355816e8-5031-4d06-a73b-2331617a0eb6
let
	dir = "/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_magru/data"
	plt = plot()
	for τ = dd["numhidden"]
		plot!(data_fac, Dict("numhidden"=>τ, "save_dir"=>dir); label_idx="truncation", sort_idx="truncation", lw=2, palette=color_scheme, label=τ, labeltitle="Hidden", xlabel="truncation", ylims=(0.0, 0.3))
	end
	plt
end

# ╔═╡ 1cdb0d15-f25c-4147-8055-458f3f5f06b2
let
	args = Dict{String, Any}[]
	for dir ∈ dd["save_dir"]
		for nh_f ∈ dd["numhidden"]
			for τ ∈ dd["truncation"]
				params = Dict(
					"numhidden"=>nh_f,
					"save_dir"=>dir,
					"truncation"=>τ)
				idx = findall(data_fac.data) do ld
					all([params[k] == ld.line_params[k] for k in keys(params)])
				end
				if length(idx) == 1
					params["eta"] = data_fac[idx][1].swept_params[1]
					delete!(params, "save_dir")
					push!(args, params)
				end
			end
		end
	end
	args
end


# ╔═╡ 1f57edbe-10b9-4857-a7ab-6f867090f159
ic_ring, dd_ring = RPU.load_data(at("final_ringworld_er_rmsprop_10/"))

# ╔═╡ 78581ee5-b4e9-4e4b-baae-2aa778854a02
save_at(plt_file) = joinpath("../../plots/ringworld/", plt_file)

# ╔═╡ 6760c7fd-244f-4265-9fae-987fa9bae1f3
md"""
Save Figures: $(@bind save_figs PlutoUI.CheckBox())
"""

# ╔═╡ 3fea2f09-2333-4bfd-8273-90368fa824ae
data_ring = RPU.get_line_data_for(
		ic_ring,
		["cell", "numhidden", "truncation"],
		[];
		comp=:min,
		get_comp_data=(x)->x["results"]["end"],
		get_data=(x)->x["results"]["end"])

# ╔═╡ 75478ea0-136f-48b9-854d-ebb73efd303a
let
	nh = 12
	dir = "/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10/data"
	sub_ic = search(ic, Dict("save_dir"=>dir, "numhidden"=>nh))
	data = RPU.get_line_data_for(
		sub_ic,
		["truncation"],
		["eta"];
		comp=:min,
		get_comp_data=(x)->x["results"]["end"],
		get_data=(x)->x["results"]["end"])
	plt = plot(legend=nothing, tickdir=:out, grid=false, tickfontsize=18, yticks=(0.0:0.05:0.3), ylims=(0.0,0.3), markersize=5)
	# for τ = dd["numhidden"]
	plot!(data; label_idx="truncation", sort_idx="truncation", lw=3, palette=color_scheme, label="FacMARNN, $(nh)", labeltitle="Hidden", color=cell_colors["FacMARNN"], line=:dash, marker=:auto, markersize=5)
	# end
	params = Dict("cell"=>"AARNN", "numhidden"=>nh)
	plot!(data_ring, params; label_idx="truncation", sort_idx="truncation", lw=3, palette=color_scheme, label="AARNN, $(nh)", labeltitle="Hidden", color=cell_colors["AARNN"], marker=:auto, markersize=5)
	params = Dict("cell"=>"RNN", "numhidden"=>nh)
	plot!(data_ring, params; label_idx="truncation", sort_idx="truncation", lw=3, palette=color_scheme, label="RNN, $(nh)", labeltitle="Hidden", color=cell_colors["RNN"], marker=:auto, markersize=5)
	
	
	nh = 9
	dir = "/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_marnn/data"
	sub_ic = search(ic, Dict("save_dir"=>dir, "numhidden"=>nh))
	data = RPU.get_line_data_for(
		sub_ic,
		["truncation"],
		["eta"];
		comp=:min,
		get_comp_data=(x)->x["results"]["end"],
		get_data=(x)->x["results"]["end"])

	plot!(data; label_idx="truncation", sort_idx="truncation", lw=3, palette=color_scheme, label="FacMARNN, $(nh)", labeltitle="Hidden", color=cell_colors["FacMARNN"], marker=:auto, markersize=5)
	
	params = Dict("cell"=>"MARNN", "numhidden"=>nh)
	plot!(data_ring, params; label_idx="truncation", sort_idx="truncation", lw=3, palette=color_scheme, label="MARNN, $(nh)", labeltitle="Hidden",color=cell_colors["MARNN"], marker=:auto, markersize=5)
	

	plt
end

# ╔═╡ 810aed19-9833-4b05-82c7-5b00e8c4d7be
let
	plt = plot(legend=nothing, tickdir=:out, grid=false, tickfontsize=18, yticks=(0.0:0.05:0.3), ylims=(0.0,0.3), markersize=5)
	
	nh = 15
	dir = "/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10/data"
	sub_ic = search(ic, Dict("save_dir"=>dir, "numhidden"=>nh))
	data = RPU.get_line_data_for(
		sub_ic,
		["truncation"],
		["eta"];
		comp=:min,
		get_comp_data=(x)->x["results"]["end"],
		get_data=(x)->x["results"]["end"])
	plot!(data; label_idx="truncation", sort_idx="truncation", lw=3, palette=color_scheme, label="FacMARNN, $(nh)", labeltitle="Hidden", color=cell_colors["FacMARNN"], line=:dash, marker=:auto, markersize=5)
	# end
	nh = 12
	dir = "/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_marnn/data"
	sub_ic = search(ic, Dict("save_dir"=>dir, "numhidden"=>nh))
	data = RPU.get_line_data_for(
		sub_ic,
		["truncation"],
		["eta"];
		comp=:min,
		get_comp_data=(x)->x["results"]["end"],
		get_data=(x)->x["results"]["end"])

	plot!(data; label_idx="truncation", sort_idx="truncation", lw=3, palette=color_scheme, label="FacMARNN, $(nh)", labeltitle="Hidden", color=cell_colors["FacMARNN"], marker=:auto, markersize=5)
	
	args = [
		Dict("cell"=>"MARNN", "numhidden"=>12),
		Dict("cell"=>"RNN", "numhidden"=>15),
		Dict("cell"=>"AARNN", "numhidden"=>15)
	]
	colors = [cell_colors["MARNN"] cell_colors["RNN"] cell_colors["AARNN"]]
	params = Dict("cell"=>"MARNN", "numhidden"=>nh)
	plot!(data_ring, args; label_idx="truncation", sort_idx="truncation", lw=3, palette=color_scheme, color=colors, marker=:auto, markersize=5)
	if save_figs
		savefig(save_at("ringworld_rnn_trunc.pdf"))
	end
	plt
end

# ╔═╡ 2db65ac7-acb5-4375-b456-8ba196a8850a
let
	nh = 12
	dir = "/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_aagru/data"
	sub_ic = search(ic, Dict("save_dir"=>dir, "numhidden"=>nh))
	data = RPU.get_line_data_for(
		sub_ic,
		["truncation"],
		["eta"];
		comp=:min,
		get_comp_data=(x)->x["results"]["end"],
		get_data=(x)->x["results"]["end"])

	plt = plot(legend=nothing, tickdir=:out, grid=false, tickfontsize=18, yticks=(0.0:0.05:0.3), ylims=(0.0,0.3), marker=:auto, markersize=5)
	# for τ = dd["numhidden"]
	plot!(data; label_idx="truncation", sort_idx="truncation", lw=3, palette=color_scheme, label="FacMAGRU, $(nh)", labeltitle="Hidden", color=cell_colors["FacMAGRU"], line=:dash, marker=:auto, markersize=5)
	# end
	params = Dict("cell"=>"AAGRU", "numhidden"=>nh)
	plot!(data_ring, params; label_idx="truncation", sort_idx="truncation", lw=3, palette=color_scheme, label="AAGRU, $(nh)", labeltitle="Hidden", color=cell_colors["AAGRU"], marker=:auto, markersize=5)
	params = Dict("cell"=>"GRU", "numhidden"=>nh)
	plot!(data_ring, params; label_idx="truncation", sort_idx="truncation", lw=3, palette=color_scheme, label="GRU, $(nh)", labeltitle="Hidden", color=cell_colors["GRU"], marker=:auto, markersize=5)
	
	nh = 9
	dir = "/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_magru/data"
	sub_ic = search(ic, Dict("save_dir"=>dir, "numhidden"=>nh))
	data = RPU.get_line_data_for(
		sub_ic,
		["truncation"],
		["eta"];
		comp=:min,
		get_comp_data=(x)->x["results"]["end"],
		get_data=(x)->x["results"]["end"])

	plot!(data; label_idx="truncation", sort_idx="truncation", lw=3, palette=color_scheme, label="FacMAGRU, $(nh)", labeltitle="Hidden", color=cell_colors["FacMAGRU"], marker=:auto, markersize=5)
	
	params = Dict("cell"=>"MAGRU", "numhidden"=>nh)
	plot!(data_ring, params; label_idx="truncation", sort_idx="truncation", lw=3, palette=color_scheme, label="MAGRU, $(nh)", labeltitle="Hidden", color=cell_colors["MAGRU"], marker=:auto, markersize=5)

	if save_figs
		savefig(save_at("ringworld_gru_trunc.pdf"))
	end
	plt
end

# ╔═╡ 5f7a1ab9-bad4-4f44-b0b6-d968dc53bb91
markers = Plots._allMarkers[Plots.is_marker_supported.(Plots._allMarkers)]

# ╔═╡ 3152373c-3567-47a0-9198-e2b3767f473f
let
	cell = "MARNN"
	plts = []
	for cell ∈ ["RNN",  "AARNN", "MARNN", "GRU",  "AAGRU", "MAGRU",]
		legend = nothing
		if cell ∈ ["MARNN", "MAGRU"]
			legend = :topright
		end
		plt = plot(legend=legend, legendfontsize=10, tickfontsize=14, grid=false, tickdir=:out, ylims=(0.0,0.3))
		for (nh_idx, nh) in enumerate(dd_ring["numhidden"])

			plot!(plt, data_ring, Dict("cell"=>cell, "numhidden"=>nh), sort_idx="truncation", marker=:auto, markersize=5, lw=3, palette=color_scheme, label=nh,title=cell, color=cell_colors[cell])
		end
		push!(plts, plt)
	end
	plt = plot(plts..., size=(1200,800))
	if save_figs
		savefig(save_at("ringworld_sens.pdf"))
	end
	plt
end

# ╔═╡ 3cbc1a74-e952-4fe6-94a8-d010509dba84
let
	plts = []
	sd_title = Dict(		"/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10/data"=>"FacMARNN: Params = AARNN",
		"/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_aagru/data"=>"FacMAGRU: Params = AAGRU",
"/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_magru/data"=>"FacMAGRU: Params = MAGRU",
"/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_marnn/data"=>"FacMARNN: Params = MARNN"
	)
	sd_cell = Dict(
"/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10/data"=>"FacMARNN",
		"/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_aagru/data"=>"FacMAGRU",
"/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_magru/data"=>"FacMAGRU",
"/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_marnn/data"=>"FacMARNN")
	save_dir_order = ["/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10/data",
"/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_marnn/data",
"/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_aagru/data",
"/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_magru/data",
]
	for dir ∈ save_dir_order
		plt = plot(palette=color_scheme, legendfontsize=10, tickfontsize=14)
		for nh ∈ dd["numhidden"]
		# dir = "/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10_magru/data"
			sub_ic = search(ic, Dict("save_dir"=>dir, "numhidden"=>nh))
			data = RPU.get_line_data_for(
				sub_ic,
				["truncation"],
				["eta"];
				comp=:min,
				get_comp_data=(x)->x["results"]["end"],
				get_data=(x)->x["results"]["end"])
			plot!(data, sort_idx="truncation", marker=:auto, 
				lw=2, markersize=5, label=nh, title=sd_title[dir], color=cell_colors[sd_cell[dir]], tickdir=:out, grid=false, ylims=(0.0,0.3))
		end
		push!(plts, plt)
	end
	plt = plot(plts..., size=(800,800))
	# savefig("../plots/factored_sens_rw.pdf")
	plt
end

# ╔═╡ 3bfcf70a-90c3-4a0f-9341-ebf54555fe45
let
	plt = plot(palette=color_scheme)
	for nh ∈ dd["numhidden"]
		dir = "/home/mkschleg/scratch/ActionRNNs/ringworld_fac_er_rmsprop_10/data"
		sub_ic = search(ic, Dict("save_dir"=>dir, "numhidden"=>nh))
		data = RPU.get_line_data_for(
			sub_ic,
			["truncation"],
			["eta"];
			comp=:min,
			get_comp_data=(x)->x["results"]["end"],
			get_data=(x)->x["results"]["end"])
		plot!(data, sort_idx="truncation", marker=:auto, 
			lw=2, markersize=5, label=nh)
	end
	plt
end

# ╔═╡ 3a50e605-9b46-4bef-b9b9-9b8f82a0139d
data_ring_lc = RPU.get_line_data_for(
		ic_ring,
		["cell", "numhidden", "truncation"],
		[];
		comp=:min,
		get_comp_data=(x)->x["results"]["end"],
		get_data=(x)->rollmean(x["results"]["lc"], 10))

# ╔═╡ 7b9b5614-f876-439f-ac2b-9e9995a95e2e
let
	plt = plot(ylims=(0.0,0.35), legend=nothing, grid=false, tickdir=:out, tickfontsize=11)
	plot!(data_ring_lc, Dict("cell"=>"RNN", "numhidden"=>15, "truncation"=>6), lw = 2, color=cell_colors["RNN"], z=1.97)
	plot!(data_ring_lc, Dict("cell"=>"MARNN", "numhidden"=>12, "truncation"=>6), lw=2, color=cell_colors["MARNN"], z=1.97)
	plot!(data_ring_lc, Dict("cell"=>"AARNN", "numhidden"=>15, "truncation"=>6), lw=2, color=cell_colors["AARNN"], z=1.97)
	if save_figs
		savefig(save_at("ringworld_example.pdf"))
	end
	plt
end

# ╔═╡ Cell order:
# ╠═f3cc35f4-1788-420e-bb20-4658eaec2eda
# ╠═cf756452-bb40-11eb-1b8f-a1c2376756ed
# ╠═a14f2ee3-b61f-4d5f-8819-9a71f977c375
# ╠═3052ee11-68ec-42ff-9177-5f70bce043b6
# ╠═e1817bb9-5918-4ed4-a73d-8c650e5399bc
# ╠═4c5a2db1-1ed2-4b86-8917-79b55727d45d
# ╠═5062d5b8-3141-4c05-82d3-1bfba1951e68
# ╠═ad0b6a6c-a062-4952-8632-616b2086bb5f
# ╠═2148c874-e915-47af-9f67-6a565c23e9ff
# ╠═4561c8c6-81a1-4966-915e-7b524135ae92
# ╠═34658923-9629-44c4-adc3-4694adebac1d
# ╠═2f13e071-94ad-4023-8ea3-b0a14f4b885b
# ╠═a089dc38-fb29-4bc1-b012-76889709a991
# ╠═2a883704-29d7-4e43-894b-b0530cbbb6fc
# ╠═66796dd0-f0ad-4df8-a9c6-b7d810734e05
# ╠═cfa3f621-3d81-4e0b-903c-f78b67d3a3ff
# ╠═d0eebe70-fb0e-4e82-aa7d-4b0fadb8dbbf
# ╠═355816e8-5031-4d06-a73b-2331617a0eb6
# ╠═1cdb0d15-f25c-4147-8055-458f3f5f06b2
# ╠═1f57edbe-10b9-4857-a7ab-6f867090f159
# ╠═78581ee5-b4e9-4e4b-baae-2aa778854a02
# ╠═6760c7fd-244f-4265-9fae-987fa9bae1f3
# ╠═3fea2f09-2333-4bfd-8273-90368fa824ae
# ╠═75478ea0-136f-48b9-854d-ebb73efd303a
# ╠═810aed19-9833-4b05-82c7-5b00e8c4d7be
# ╠═2db65ac7-acb5-4375-b456-8ba196a8850a
# ╠═5f7a1ab9-bad4-4f44-b0b6-d968dc53bb91
# ╠═3152373c-3567-47a0-9198-e2b3767f473f
# ╠═3cbc1a74-e952-4fe6-94a8-d010509dba84
# ╠═3bfcf70a-90c3-4a0f-9341-ebf54555fe45
# ╠═3a50e605-9b46-4bef-b9b9-9b8f82a0139d
# ╠═7b9b5614-f876-439f-ac2b-9e9995a95e2e
# ╠═6607a779-d1b0-41b7-8b4f-69a83ef8d543
