### A Pluto.jl notebook ###
# v0.19.3

using Markdown
using InteractiveUtils

# ╔═╡ 7525a21d-d7b4-4799-baaa-87fe30a1307d
let
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ 936c3912-bca1-11eb-3195-47018b99d2ce
using Revise

# ╔═╡ cef11852-f81b-47db-ba65-7bc097493c70
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 98ec85a9-1f76-46d3-8c86-6fed39cffc57
const RPU = ReproducePlotUtils

# ╔═╡ 71d8dbae-27c7-4403-b46c-7ebfcdd8d362
RPU.stats_plot_types

# ╔═╡ db4df601-6c3e-4c6f-81c9-59972e7fda21
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

# ╔═╡ ba710c0c-4967-4524-b802-d737998cd537
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ d43d0180-0efc-48f7-970c-3098d05f0b69
begin
	dataloc = "../local_data/image_tmaze/"
	at(folder) = joinpath(dataloc, folder)
end

# ╔═╡ cf46b03d-ad07-454e-9481-5a6ce725b980
ic, dd = let
	ic = ItemCollection(at("image_dir_tmaze_adam_6/"))
	subic = search(ic) do ld
		ld.parsed_args["cell"] !== "FacMAGRU"
	end
	subic, diff(subic)
end

# ╔═╡ 197d1f95-a93f-45fc-8afe-b4174fbc4011
ic_fac, dd_fac = let
	ic = ItemCollection(at("image_dir_tmaze_adam_6_init/"))
	subic = search(ic, Dict("cell"=>"FacMAGRU"))
	subic, diff(subic)
end

# ╔═╡ 7d800cd0-5f42-4039-b050-8989bd26c226
data = RPU.get_line_data_for(
	ic,
	["cell", "numhidden", "truncation"],
	["eta"];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ b84e3eac-582b-45bd-8852-162c2039c63a
let
	τ = 12
	boxplot(data, Dict("cell"=>"MAGRU", "numhidden"=>32, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"], legend=nothing)
	boxplot!(data, Dict("cell"=>"AAGRU", "numhidden"=>70, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"])
	boxplot!(data, Dict("cell"=>"MAGRU", "numhidden"=>64, "truncation"=>τ), label="MAGRU_64", color=cell_colors["MAGRU"])
	boxplot!(data, Dict("cell"=>"AAGRU", "numhidden"=>132, "truncation"=>τ), label="AAGRU_132", color=cell_colors["AAGRU"])
	
	# dotplot!(data, Dict("cell"=>"MAGRU", "numhidden"=>32, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"])
	# dotplot!(data, Dict("cell"=>"AAGRU", "numhidden"=>70, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"])
	# dotplot!(data, Dict("cell"=>"MAGRU", "numhidden"=>64, "truncation"=>τ), label="MAGRU_64", color=cell_colors["MAGRU"])
	# dotplot!(data, Dict("cell"=>"AAGRU", "numhidden"=>132, "truncation"=>τ), label="AAGRU_132", color=cell_colors["AAGRU"])
end

# ╔═╡ 721c1ef2-e402-4aaa-b686-81e6f8c0ebdd
data_fac = RPU.get_line_data_for(
	ic_fac,
	["numhidden", "factors", "truncation", "init_style"],
	["eta"];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ e33f5c8a-b5d9-46b5-b94b-633fcbcc6951
let
	data_plot(nh, τ, init_style) = begin
		args = Dict("numhidden"=>nh, "truncation"=>τ, "init_style"=>init_style)
		boxplot!(data_fac, args, label_idx="numhidden", color=cell_colors["FacMAGRU"], legend=nothing, make_label_string=true)
		dotplot!(data_fac, args, label_idx="numhidden", color=cell_colors["FacMAGRU"], make_label_string=true)
	end
	plot(title="Standard")
	plt1 = data_plot.([32, 64, 70, 132], 12, "standard")[end]
	plt2 = plot(title="Tensor")
	data_plot.([32, 64, 70, 132], 12, "tensor")
	plot(plt1, plt2)
end

# ╔═╡ d8e7ac8a-c2c7-4b51-9a78-2a63a8b02ace
md"""
# Learning Rate sensitivity
"""

# ╔═╡ 0ce9b0f8-0e10-4f39-9aa5-3670e598e8cd
data_sens = RPU.get_line_data_for(
	ic,
	["cell", "numhidden", "truncation", "eta"],
	[];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ dccacee5-b3e0-4405-a262-21a10a39dc2c
let
	args = [
		Dict("cell"=>"MAGRU", "numhidden"=>32, "truncation"=>20),
		Dict("cell"=>"MAGRU", "numhidden"=>64, "truncation"=>20),
		Dict("cell"=>"AAGRU", "numhidden"=>70, "truncation"=>20),
		Dict("cell"=>"AAGRU", "numhidden"=>132, "truncation"=>20)
	]
	labels = ["MAGRU_32" "MAGRU_64" "AAGRU_70" "AAGRU_132"]
	plot(data_sens, args, xscale=:log, sort_idx="eta", lw=2, label=labels, palette=color_scheme)
	
end

# ╔═╡ 1733c0a0-3572-42fa-b618-171cb9659a7b
let
	params = [
		Dict("cell"=>"MAGRU", "numhidden"=>32),
		Dict("cell"=>"MAGRU", "numhidden"=>64),
		Dict("cell"=>"AAGRU", "numhidden"=>70),
		Dict("cell"=>"AAGRU", "numhidden"=>132)
	]
	args = Dict{String, Any}[]
	for τ ∈ dd["truncation"]
		for pms ∈ params
			pms["truncation"] = τ
			idx = findfirst(data.data) do ld
				all([pms[k] == ld.line_params[k] for k in keys(pms)])
			end
			pms_copy = copy(pms)
			pms_copy["eta"] = data[idx].swept_params[1]
			push!(args, pms_copy)
		end
	end
	# FileIO.save("../final_runs/viz_dir_tmaze.jld2", "args", args)
	args
end

# ╔═╡ d671e923-a428-4103-95b1-77e2d835c755
data_fac_sens = RPU.get_line_data_for(
	ic_fac,
	["numhidden", "factors", "truncation", "eta", "init_style"],
	[];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 7fb294f5-8db0-4c11-875a-e74d4340de9b
let
	args = [
		Dict("numhidden"=>32, "truncation"=>12, "init_style"=>"tensor"),
		Dict("numhidden"=>64, "truncation"=>12, "init_style"=>"tensor"),
		Dict("numhidden"=>70, "truncation"=>12, "init_style"=>"tensor"),
		Dict("numhidden"=>132, "truncation"=>12, "init_style"=>"tensor")
	]
	labels = ["32" "64" "70" "132"]
	plot(data_fac_sens, args, xscale=:log, sort_idx="eta", lw=2, label=labels, palette=color_scheme)
end

# ╔═╡ 71979e6f-f13f-440f-b787-844970148628
let
	args = [
		Dict("numhidden"=>32, "truncation"=>12, "init_style"=>"standard"),
		Dict("numhidden"=>64, "truncation"=>12, "init_style"=>"standard"),
		Dict("numhidden"=>70, "truncation"=>12, "init_style"=>"standard"),
		Dict("numhidden"=>132, "truncation"=>12, "init_style"=>"standard")
	]
	labels = ["32" "64" "70" "132"]
	plot(data_fac_sens, args, xscale=:log, sort_idx="eta", lw=2, label=labels, palette=color_scheme)
end

# ╔═╡ c75da967-a8d0-49f0-98cf-087f07823cb7
let
	params = Dict{String, Any}[
		Dict("numhidden"=>32, "truncation"=>12,),
		Dict("numhidden"=>64, "truncation"=>12,),
		Dict("numhidden"=>70, "truncation"=>12,),
		Dict("numhidden"=>132, "truncation"=>12,)
	]
	args = Dict{String, Any}[]
	for (τ, init_style) ∈ Iterators.product(dd_fac["truncation"], dd_fac["init_style"])
		for pms_org ∈ params
			pms = copy(pms_org)
			pms["truncation"] = τ
			pms["init_style"] = init_style
			idx = findfirst(data_fac.data) do ld
				all([pms[k] == ld.line_params[k] for k in keys(pms)])
			end
			pms_copy = convert(Dict{String,Any}, copy(data_fac[idx].line_params))
			pms_copy["eta"] = data_fac[idx].swept_params[1]
			pms_copy["cell"] = "FacMAGRU"
			push!(args, pms_copy)
		end
	end
	FileIO.save("../final_runs/viz_fac_dir_tmaze.jld2", "args", args)
	args
end

# ╔═╡ 8cbbc5fe-0daa-4168-85fa-2a4beab15240
let
	τ = 20
	plt = plot(grid=false, tickdir=:out, tickfontsize=12)
	
	boxplot!(data, Dict("cell"=>"MAGRU", "numhidden"=>32, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"], legend=nothing)
	boxplot!(data, Dict("cell"=>"AAGRU", "numhidden"=>70, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"])
	dotplot!(data, Dict("cell"=>"MAGRU", "numhidden"=>32, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"])
	dotplot!(data, Dict("cell"=>"AAGRU", "numhidden"=>70, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"])
	
	for nh ∈ dd_fac["numhidden"][[1, 3]]
		boxplot!(data_fac, Dict("numhidden"=>nh, "truncation"=>τ), label="FacGRU $(nh)", color=cell_colors["FacMAGRU"], legend=nothing, make_label_string=true)
		dotplot!(data_fac, Dict("numhidden"=>nh, "truncation"=>τ),  color=cell_colors["FacMAGRU"], make_label_string=true, label="FacGRU $(nh)")
	end
	
# 	boxplot!(data, Dict("cell"=>"MAGRU", "numhidden"=>64, "truncation"=>τ), label="MAGRU_64", color=cell_colors["MAGRU"])
# 	boxplot!(data, Dict("cell"=>"AAGRU", "numhidden"=>132, "truncation"=>τ), label="AAGRU_132", color=cell_colors["AAGRU"])
# 	dotplot!(data, Dict("cell"=>"MAGRU", "numhidden"=>64, "truncation"=>τ), label="MAGRU_64", color=cell_colors["MAGRU"])
# 	dotplot!(data, Dict("cell"=>"AAGRU", "numhidden"=>132, "truncation"=>τ), label="AAGRU_132", color=cell_colors["AAGRU"])

# 	for nh ∈ dd_fac["numhidden"][[2, 4]]
# 		boxplot!(data_fac, Dict("numhidden"=>nh, "truncation"=>τ),  color=cell_colors["FacMAGRU"], legend=nothing, make_label_string=true, label="FacRNN $(nh)")
# 		dotplot!(data_fac, Dict("numhidden"=>nh, "truncation"=>τ),  color=cell_colors["FacMAGRU"], make_label_string=true, label="FacRNN $(nh)")
# 	end
	savefig("../plots/image_dir_tmaze_32.pdf")
	plt
end

# ╔═╡ 84d7b94a-18da-4176-87ba-4962ea3a06da
let
	τ = 20
	plt = plot(grid=false, tickdir=:out, tickfontsize=12)
	
# 	boxplot!(data, Dict("cell"=>"MAGRU", "numhidden"=>32, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"], legend=nothing)
# 	boxplot!(data, Dict("cell"=>"AAGRU", "numhidden"=>70, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"])
# 	dotplot!(data, Dict("cell"=>"MAGRU", "numhidden"=>32, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"])
# 	dotplot!(data, Dict("cell"=>"AAGRU", "numhidden"=>70, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"])
	
# 	for nh ∈ dd_fac["numhidden"][[1, 3]]
# 		boxplot!(data_fac, Dict("numhidden"=>nh, "truncation"=>τ), label="FacGRU $(nh)", color=cell_colors["FacMAGRU"], legend=nothing, make_label_string=true)
# 		dotplot!(data_fac, Dict("numhidden"=>nh, "truncation"=>τ),  color=cell_colors["FacMAGRU"], make_label_string=true, label="FacGRU $(nh)")
# 	end
	
	boxplot!(data, Dict("cell"=>"MAGRU", "numhidden"=>64, "truncation"=>τ), label="MAGRU_64", color=cell_colors["MAGRU"])
	boxplot!(data, Dict("cell"=>"AAGRU", "numhidden"=>132, "truncation"=>τ), label="AAGRU_132", color=cell_colors["AAGRU"])
	dotplot!(data, Dict("cell"=>"MAGRU", "numhidden"=>64, "truncation"=>τ), label="MAGRU_64", color=cell_colors["MAGRU"])
	dotplot!(data, Dict("cell"=>"AAGRU", "numhidden"=>132, "truncation"=>τ), label="AAGRU_132", color=cell_colors["AAGRU"])

	for nh ∈ dd_fac["numhidden"][[2, 4]]
		boxplot!(data_fac, Dict("numhidden"=>nh, "truncation"=>τ),  color=cell_colors["FacMAGRU"], legend=nothing, make_label_string=true, label="FacRNN $(nh)")
		dotplot!(data_fac, Dict("numhidden"=>nh, "truncation"=>τ),  color=cell_colors["FacMAGRU"], make_label_string=true, label="FacRNN $(nh)")
	end
	savefig("../plots/image_dir_tmaze_64.pdf")
	plt
end

# ╔═╡ e94b8cf0-30ab-46ba-bdab-34d8ac36ae1f
begin
	ic_final, dd_final = 
		RPU.load_data(at("final_image_dir_tmaze_adam_6/"))
	ic_fac_final, dd_fac_final = 
		RPU.load_data(at("final_image_fac_dir_tmaze_adam_6/"))
end

# ╔═╡ 9441e2ee-8d19-4128-bf39-098e81cf792b
final_run_args = FileIO.load("../final_runs/viz_dir_tmaze.jld2")

# ╔═╡ 43f20292-f128-471e-88ec-5ebcb429edb6
length(ic_final)

# ╔═╡ e98a6bce-649f-4eb0-9986-0fd47c53c08f
ic_final[1].parsed_args

# ╔═╡ 54d147f2-0574-4f45-a230-073264c87093
begin
	data_final = RPU.get_line_data_for(
		ic_final,
		["cell", "numhidden", "truncation"],
		[];
		comp=:max,
		get_comp_data=(x)->RPU.get_MUE(x, :successes),
		get_data=(x)->RPU.get_MUE(x, :successes))
	data_fac_final = RPU.get_line_data_for(
		ic_fac_final,
		["numhidden", "truncation"],
		[];
		comp=:max,
		get_comp_data=(x)->RPU.get_MUE(x, :successes),
		get_data=(x)->RPU.get_MUE(x, :successes))
end

# ╔═╡ 0c988e39-6731-4849-9df7-bf3479e7a2c0


# ╔═╡ 3d34794d-b601-4eb8-ac67-284b4dbf04ff
md"""
# Working Final Plots
"""

# ╔═╡ ee067d4a-6502-43d8-9b17-b2d8d41926a4
let
	τ = 20
	
	plot(legend=false, ylims=(0.4, 1.0))
	boxplot!(data_final, Dict("cell"=>"AAGRU", "numhidden"=>70, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"])
	boxplot!(data_final, Dict("cell"=>"MAGRU", "numhidden"=>32, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"])
	boxplot!(data_fac_final, Dict("numhidden"=>32, "truncation"=>τ), label="FacGRU 32", color=cell_colors["FacMAGRU"])
	boxplot!(data_fac_final, Dict("numhidden"=>70, "truncation"=>τ), label="FacGRU 70", color=cell_colors["FacMAGRU"])
	
	dotplot!(data_final, Dict("cell"=>"AAGRU", "numhidden"=>70, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"])
	dotplot!(data_final, Dict("cell"=>"MAGRU", "numhidden"=>32, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"])
	dotplot!(data_fac_final, Dict("numhidden"=>32, "truncation"=>τ), label="FacGRU 32", color=cell_colors["FacMAGRU"])
	dotplot!(data_fac_final, Dict("numhidden"=>70, "truncation"=>τ), label="FacGRU 70", color=cell_colors["FacMAGRU"])
end

# ╔═╡ a7d49c82-67d0-4e24-81f8-5083c9817545
let
	τ = 20

	plt = plot(legend=false, ylims=(0.4, 1.0), grid=false, tickdir=:out, tickfontsize=12)
	
	violin!(data_final, Dict("cell"=>"AAGRU", "numhidden"=>132, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"], linecolor=cell_colors["AAGRU"], lw=1)
	boxplot!(data_final, Dict("cell"=>"AAGRU", "numhidden"=>132, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"], fillalpha=0.75, lw=2, linecolor=:black)
	# dotplot!(data_final, Dict("cell"=>"AAGRU", "numhidden"=>132, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"])
	
	violin!(data_final, Dict("cell"=>"MAGRU", "numhidden"=>64, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"], linecolor=cell_colors["MAGRU"], lw=1)
	boxplot!(data_final, Dict("cell"=>"MAGRU", "numhidden"=>64, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"], fillalpha=0.75, lw=2, linecolor=:black)
	# dotplot!(data_final, Dict("cell"=>"MAGRU", "numhidden"=>64, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"])
	
	
	violin!(data_fac_final, Dict("numhidden"=>64, "truncation"=>τ), label="FacGRU 64", color=cell_colors["FacMAGRU"], lw=1, linecolor=cell_colors["FacMAGRU"])
	boxplot!(data_fac_final, Dict("numhidden"=>64, "truncation"=>τ), label="FacGRU 64", color=cell_colors["FacMAGRU"], fillalpha=0.75, lw=2, linecolor=:black)
	# dotplot!(data_fac_final, Dict("numhidden"=>64, "truncation"=>12), label="FacGRU 32", color=cell_colors["FacMAGRU"])
	
	violin!(data_fac_final, Dict("numhidden"=>132, "truncation"=>τ), label="FacGRU 132", color=cell_colors["FacMAGRU"], lw=1, linecolor=cell_colors["FacMAGRU"])
	boxplot!(data_fac_final, Dict("numhidden"=>132, "truncation"=>τ), label="FacGRU 132", color=cell_colors["FacMAGRU"], fillalpha=0.75, lw=2, linecolor=:black)
	# dotplot!(data_fac_final, Dict("numhidden"=>132, "truncation"=>12), label="FacGRU 70", color=cell_colors["FacMAGRU"])
	savefig("../plots/viz_dir_tmaze_$(τ).pdf")
	plt
end

# ╔═╡ 733fb23e-d2ff-4078-8f10-fdb53034ebb6
let
	τ = 12

	plt = plot(legend=false, ylims=(0.4, 1.0), grid=false, tickdir=:out, tickfontsize=12)
	
	violin!(data_final, Dict("cell"=>"AAGRU", "numhidden"=>70, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"], linecolor=cell_colors["AAGRU"], lw=1)
	boxplot!(data_final, Dict("cell"=>"AAGRU", "numhidden"=>70, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"], fillalpha=0.75, lw=2, linecolor=:black)
	# dotplot!(data_final, Dict("cell"=>"AAGRU", "numhidden"=>132, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"])
	
	violin!(data_final, Dict("cell"=>"MAGRU", "numhidden"=>32, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"], linecolor=cell_colors["MAGRU"], lw=1)
	boxplot!(data_final, Dict("cell"=>"MAGRU", "numhidden"=>32, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"], fillalpha=0.75, lw=2, linecolor=:black)
	# dotplot!(data_final, Dict("cell"=>"MAGRU", "numhidden"=>64, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"])
	
	
	violin!(data_fac_final, Dict("numhidden"=>32, "truncation"=>τ), label="FacGRU 32", color=cell_colors["FacMAGRU"], lw=1, linecolor=cell_colors["FacMAGRU"])
	boxplot!(data_fac_final, Dict("numhidden"=>32, "truncation"=>τ), label="FacGRU 32", color=cell_colors["FacMAGRU"], fillalpha=0.75, lw=2, linecolor=:black)
	# dotplot!(data_fac_final, Dict("numhidden"=>64, "truncation"=>12), label="FacGRU 32", color=cell_colors["FacMAGRU"])
	
	violin!(data_fac_final, Dict("numhidden"=>70, "truncation"=>τ), label="FacGRU 70", color=cell_colors["FacMAGRU"], lw=1, linecolor=cell_colors["FacMAGRU"])
	boxplot!(data_fac_final, Dict("numhidden"=>70, "truncation"=>τ), label="FacGRU 70", color=cell_colors["FacMAGRU"], fillalpha=0.75, lw=2, linecolor=:black)
	# dotplot!(data_fac_final, Dict("numhidden"=>132, "truncation"=>12), label="FacGRU 70", color=cell_colors["FacMAGRU"])
	savefig("../plots/viz_dir_tmaze_$(τ)_small.pdf")
	plt
end

# ╔═╡ 0e699ab4-722b-4b82-945f-3bd59e360774
let
	τ = 20

	plot(legend=false, ylims=(0.4, 1.0), grid=false, tickdir=:out, tickfontsize=12)
	boxplot!(data_final, Dict("cell"=>"AAGRU", "numhidden"=>70, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"])
	boxplot!(data_final, Dict("cell"=>"MAGRU", "numhidden"=>32, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"])
	boxplot!(data_fac, Dict("numhidden"=>64, "truncation"=>τ), label="FacGRU 32", color=cell_colors["FacMAGRU"])
	boxplot!(data_fac, Dict("numhidden"=>132, "truncation"=>τ), label="FacGRU 70", color=cell_colors["FacMAGRU"])
	
	dotplot!(data_final, Dict("cell"=>"AAGRU", "numhidden"=>132, "truncation"=>τ), label_idx="cell", color=cell_colors["AAGRU"])
	dotplot!(data_final, Dict("cell"=>"MAGRU", "numhidden"=>64, "truncation"=>τ), label_idx="cell", color=cell_colors["MAGRU"])
	dotplot!(data_fac, Dict("numhidden"=>64, "truncation"=>τ), label="FacGRU 32", color=cell_colors["FacMAGRU"])
	dotplot!(data_fac, Dict("numhidden"=>132, "truncation"=>τ), label="FacGRU 70", color=cell_colors["FacMAGRU"])
end

# ╔═╡ Cell order:
# ╠═7525a21d-d7b4-4799-baaa-87fe30a1307d
# ╠═936c3912-bca1-11eb-3195-47018b99d2ce
# ╠═cef11852-f81b-47db-ba65-7bc097493c70
# ╠═98ec85a9-1f76-46d3-8c86-6fed39cffc57
# ╠═71d8dbae-27c7-4403-b46c-7ebfcdd8d362
# ╠═db4df601-6c3e-4c6f-81c9-59972e7fda21
# ╠═ba710c0c-4967-4524-b802-d737998cd537
# ╠═d43d0180-0efc-48f7-970c-3098d05f0b69
# ╠═cf46b03d-ad07-454e-9481-5a6ce725b980
# ╠═197d1f95-a93f-45fc-8afe-b4174fbc4011
# ╠═7d800cd0-5f42-4039-b050-8989bd26c226
# ╠═b84e3eac-582b-45bd-8852-162c2039c63a
# ╠═721c1ef2-e402-4aaa-b686-81e6f8c0ebdd
# ╠═e33f5c8a-b5d9-46b5-b94b-633fcbcc6951
# ╠═d8e7ac8a-c2c7-4b51-9a78-2a63a8b02ace
# ╠═0ce9b0f8-0e10-4f39-9aa5-3670e598e8cd
# ╠═dccacee5-b3e0-4405-a262-21a10a39dc2c
# ╠═1733c0a0-3572-42fa-b618-171cb9659a7b
# ╠═d671e923-a428-4103-95b1-77e2d835c755
# ╠═7fb294f5-8db0-4c11-875a-e74d4340de9b
# ╠═71979e6f-f13f-440f-b787-844970148628
# ╠═c75da967-a8d0-49f0-98cf-087f07823cb7
# ╠═8cbbc5fe-0daa-4168-85fa-2a4beab15240
# ╠═84d7b94a-18da-4176-87ba-4962ea3a06da
# ╠═e94b8cf0-30ab-46ba-bdab-34d8ac36ae1f
# ╠═9441e2ee-8d19-4128-bf39-098e81cf792b
# ╠═43f20292-f128-471e-88ec-5ebcb429edb6
# ╠═e98a6bce-649f-4eb0-9986-0fd47c53c08f
# ╠═54d147f2-0574-4f45-a230-073264c87093
# ╠═0c988e39-6731-4849-9df7-bf3479e7a2c0
# ╟─3d34794d-b601-4eb8-ac67-284b4dbf04ff
# ╠═ee067d4a-6502-43d8-9b17-b2d8d41926a4
# ╠═a7d49c82-67d0-4e24-81f8-5083c9817545
# ╠═733fb23e-d2ff-4078-8f10-fdb53034ebb6
# ╠═0e699ab4-722b-4b82-945f-3bd59e360774
