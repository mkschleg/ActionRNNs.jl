### A Pluto.jl notebook ###
# v0.15.1

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

# ╔═╡ 817410e7-a12d-4318-bf92-be86d32f45b6
let
   import Pkg
   Pkg.activate("../..")
end

# ╔═╡ 0f11b684-a3f9-4d40-9c63-33b28136d4f5
using Revise

# ╔═╡ 95498906-eea1-11eb-26f4-8574b531e0dc
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 78632921-6931-4ca0-9e8a-095a7f2c99b0
using JLD2

# ╔═╡ 8b4861e4-cfbc-4a0e-8246-7d6182c83586
const RPU = ReproducePlotUtils

# ╔═╡ edd86b13-bdb5-4811-8e0a-36230d1162ae
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

# ╔═╡ 1c2a5244-0999-44f2-a952-3f495a8ec136
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2],
	"ActionGatedRNN" => color_scheme[end-1])

# ╔═╡ f197d16b-eb04-408c-a020-9023f2eb02b2
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ a67aba27-40ef-4ba2-816a-6da61a1d523b
ic, dd = RPU.load_data("../../local_data/gated_rnns/dir_tmaze_gated_er_rmsprop_10/")

# ╔═╡ f9c41407-995c-4ca5-b5d4-46bce5f9585a
ic_gaigru, dd_gaigru = RPU.load_data("../../local_data/gated_rnns/dir_tmaze_er_rmsprop_10_gaigru/")

# ╔═╡ 3a1b767e-d09b-4079-b2a5-7e2e189a1600
ic_gaiagru, dd_gaiagru = RPU.load_data("../../local_data/gated_rnns/dir_tmaze_er_rmsprop_10_gaiagru/")

# ╔═╡ 606b1f70-976d-4adc-9d1b-a34835b947e4
ic_gaiarnn, dd_gaiarnn = RPU.load_data("../../local_data/gated_rnns/dir_tmaze_er_rmsprop_10_gaiarnn/")

# ╔═╡ a1bde34d-7216-4dc0-a08e-eae8a03148a6
ic_gaugru, dd_gaugru = RPU.load_data("../../local_data/gated_rnns/dir_tmaze_er_rmsprop_10_gaugru/")

# ╔═╡ be2a5e97-a857-4c7b-bb50-ef3248cdd54f
data_gairnn = RPU.get_line_data_for(
	ic,
	["cell", "numhidden", "factors", "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 1000))

# ╔═╡ 2e0ff5cc-6758-4cf1-930b-c55f88255e15
data_gaigru = RPU.get_line_data_for(
	ic_gaigru,
	["numhidden", "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 1000))

# ╔═╡ 54cae2a9-e112-4b61-8f39-0dd68deca36f
data_gaiarnn = RPU.get_line_data_for(
	ic_gaiarnn,
	["numhidden", "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 1000))

# ╔═╡ 1b39ba62-7603-450c-a41c-5c67c786ef73
data_gaiagru = RPU.get_line_data_for(
	ic_gaiagru,
	["numhidden", "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 1000))

# ╔═╡ 6e13f656-5db2-4913-ae77-232f4d1cc770
data_gaugru = RPU.get_line_data_for(
	ic_gaugru,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 1000))

# ╔═╡ e3c83d46-f6f0-4d8a-969a-11579c3ca3ce
data_steps_gairnn = RPU.get_line_data_for(
	ic,
	["cell", "numhidden", "factors", "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 1000))

# ╔═╡ c36d3791-8828-48c7-adcd-fb0c0d5547b7
data_steps_gaigru = RPU.get_line_data_for(
	ic_gaigru,
	["numhidden" "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 1000))

# ╔═╡ 357622c5-491f-400d-ac29-2f3a312a7f8c
data_steps_gaiarnn = RPU.get_line_data_for(
	ic_gaiarnn,
	["numhidden" "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 1000))

# ╔═╡ eb740dd2-b236-4f98-9a32-db0e12887595
data_steps_gaiagru = RPU.get_line_data_for(
	ic_gaiagru,
	["numhidden" "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 1000))

# ╔═╡ e29dbabe-be8a-45c5-9b01-68c7d49c306d
data_steps_gaugru = RPU.get_line_data_for(
	ic_gaugru,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :total_steps, 1000))

# ╔═╡ d944303f-cdf4-413e-8932-a04507168eab
boxplot_data = RPU.get_line_data_for(
	ic,
	["cell", "numhidden", "factors", "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 9e8614a4-7e20-404b-a025-77ef51947dde
boxplot_data_gaigru = RPU.get_line_data_for(
	ic_gaigru,
	["numhidden" "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ a41a0df4-670a-474b-8453-2130d67137ba
boxplot_data_gaiagru = RPU.get_line_data_for(
	ic_gaiagru,
	["numhidden" "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 41d54ec8-aeac-40b2-a682-f9724baf4c64
boxplot_data_gaiarnn = RPU.get_line_data_for(
	ic_gaiarnn,
	["numhidden" "internal"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 8c206951-8f90-4a68-be21-12100ba713c1
boxplot_data_gaugru = RPU.get_line_data_for(
	ic_gaugru,
	[],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 01462edf-e99c-4b7a-9f5b-3db66954b77b
md"""
Cell: $(@bind cell Select(string.(dd["cell"])))
NumHidden: $(@bind nh_gairnn Select(string.(dd["numhidden"])))
Factors: $(@bind facs_gairnn Select(string.(dd["factors"])))
Internal: $(@bind internal_gairnn Select(string.(dd["internal"])))
"""

# ╔═╡ 90b1ae5e-43f6-498b-9a00-7979b651935e
let	
	args_list = [
		Dict("cell"=>"RNN", "numhidden"=>30, "factors"=>0, "internal"=>0),
		Dict("cell"=>"AARNN", "numhidden"=>30, "factors"=>0, "internal"=>0),
		Dict("cell"=>"MARNN", "numhidden"=>18, "factors"=>0, "internal"=>0),
		Dict("cell"=>"FacMARNN", "numhidden"=>25, "factors"=>15, "internal"=>0),
	]
	
	color_list = ["RNN", "AARNN", "MARNN", "FacMARNN", "GRU", "AAGRU"]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	plt = plot()
	for i ∈ 1:length(args_list)
		plt = plot!(data_gairnn, args_list[i], z=1.97, lw=3, fillalpha=0.2, label="$(args_list[i]["cell"]) (nh: $(args_list[i]["numhidden"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="Success Rate", xlabel="Episode", title="Dir TMaze ER, Size: 10, 300K Steps", grid=false, tickdir=:out, color=cell_colors[args_list[i]["cell"]], ylim=(0.5, 1))
	end
	plt
end

# ╔═╡ a8c37334-01c6-484a-8e77-e7d023db891d
let	
	args_list = [
		Dict("cell"=>"RNN", "numhidden"=>30, "factors"=>0, "internal"=>0),
		Dict("cell"=>"AARNN", "numhidden"=>30, "factors"=>0, "internal"=>0),
		Dict("cell"=>"MARNN", "numhidden"=>18, "factors"=>0, "internal"=>0),
		Dict("cell"=>"FacMARNN", "numhidden"=>25, "factors"=>15, "internal"=>0),
	]
	
	color_list = ["RNN", "AARNN", "MARNN", "FacMARNN", "GRU", "AAGRU"]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	plt = plot()
	for i ∈ 1:length(args_list)
		plt = plot!(data_steps_gairnn, args_list[i], z=1.97, lw=3, fillalpha=0.2, label="$(args_list[i]["cell"]) (nh: $(args_list[i]["numhidden"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="Steps", xlabel="Episode", title="Dir TMaze ER, Size: 10, 300K Steps", grid=false, tickdir=:out, color=cell_colors[args_list[i]["cell"]], ylim=(0, 200))
	end
	plt
end

# ╔═╡ 6b1e0070-edf6-4125-80f6-45aa497ece18
let
	nh = parse(Int, nh_gairnn)
	factors = parse(Int, facs_gairnn)
	internal = parse(Int, internal_gairnn)
	
	args_list = [
		Dict("cell"=>"GRU", "numhidden"=>17, "factors"=>0, "internal"=>0),
		Dict("cell"=>"AAGRU", "numhidden"=>17, "factors"=>0, "internal"=>0),
		Dict("cell"=>"MAGRU", "numhidden"=>10, "factors"=>0, "internal"=>0),
		Dict("cell"=>"FacMAGRU", "numhidden"=>15, "factors"=>17, "internal"=>0),
	]
	
	color_list = ["RNN", "AARNN", "MARNN", "FacMARNN", "GRU", "AAGRU"]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	plt = plot()
	for i ∈ 1:length(args_list)
		plt = plot!(data_gairnn, args_list[i], z=1.97, lw=3, fillalpha=0.2, label="$(args_list[i]["cell"]) (nh: $(args_list[i]["numhidden"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="Success Rate", xlabel="Episode", title="Dir TMaze ER, Size: 10, 300K Steps", grid=false, tickdir=:out, color=cell_colors[args_list[i]["cell"]], ylim=(0.5, 1))
	end
	plt
end

# ╔═╡ 3d9d5add-a484-4fbc-9d0c-54b06936485d
let
	nh = parse(Int, nh_gairnn)
	factors = parse(Int, facs_gairnn)
	internal = parse(Int, internal_gairnn)
	
	args_list = [
		Dict("cell"=>"GRU", "numhidden"=>17, "factors"=>0, "internal"=>0),
		Dict("cell"=>"AAGRU", "numhidden"=>17, "factors"=>0, "internal"=>0),
		Dict("cell"=>"MAGRU", "numhidden"=>10, "factors"=>0, "internal"=>0),
		Dict("cell"=>"FacMAGRU", "numhidden"=>15, "factors"=>17, "internal"=>0),
	]
	
	color_list = ["RNN", "AARNN", "MARNN", "FacMARNN", "GRU", "AAGRU"]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	plt = plot()
	for i ∈ 1:length(args_list)
		plt = plot!(data_steps_gairnn, args_list[i], z=1.97, lw=3, fillalpha=0.2, label="$(args_list[i]["cell"]) (nh: $(args_list[i]["numhidden"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="Steps", xlabel="Episode", title="Dir TMaze ER, Size: 10, 300K Steps", grid=false, tickdir=:out, color=cell_colors[args_list[i]["cell"]], ylim=(0, 150))
	end
	plt
end

# ╔═╡ 14ad8703-dca4-42a4-9bdc-36ad0fb2e3b5
let
	nh = parse(Int, nh_gairnn)
	factors = parse(Int, facs_gairnn)
	internal = parse(Int, internal_gairnn)
	
	args_list = [
		Dict("cell"=>"ActionGatedRNN", "numhidden"=>10, "factors"=>0, "internal"=>28),
		Dict("cell"=>"ActionGatedRNN", "numhidden"=>15, "factors"=>0, "internal"=>21),
		Dict("cell"=>"ActionGatedRNN", "numhidden"=>20, "factors"=>0, "internal"=>17),
	]
	
	color_list = ["RNN", "AARNN", "MARNN", "FacMARNN", "GRU", "AAGRU"]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	linestyles = [:solid, :dash, :dot]
	plt = plot()
	for i ∈ 1:length(args_list)
		plt = plot!(data_gairnn, args_list[i], z=1.97, lw=3, fillalpha=0.2, label="$(args_list[i]["cell"]) (nh: $(args_list[i]["numhidden"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="Success Rate", xlabel="Episode", title="Dir TMaze ER, Size: 10, 300K Steps", grid=false, tickdir=:out, color=cell_colors[args_list[i]["cell"]], ylim=(0.5, 1), linestyle=linestyles[i])
	end
	plt
end

# ╔═╡ 14e4845e-c8f1-46c1-805f-1cf2b8c2ab52
let
	nh = parse(Int, nh_gairnn)
	factors = parse(Int, facs_gairnn)
	internal = parse(Int, internal_gairnn)
	
	args_list = [
		Dict("cell"=>"ActionGatedRNN", "numhidden"=>10, "factors"=>0, "internal"=>28),
		Dict("cell"=>"ActionGatedRNN", "numhidden"=>15, "factors"=>0, "internal"=>21),
		Dict("cell"=>"ActionGatedRNN", "numhidden"=>20, "factors"=>0, "internal"=>17),
	]
	
	color_list = ["RNN", "AARNN", "MARNN", "FacMARNN", "GRU", "AAGRU"]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	linestyles = [:solid, :dash, :dot]
	plt = plot()
	for i ∈ 1:length(args_list)
		plt = plot!(data_steps_gairnn, args_list[i], z=1.97, lw=3, fillalpha=0.2, label="$(args_list[i]["cell"]) (nh: $(args_list[i]["numhidden"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="Success Rate", xlabel="Episode", title="Dir TMaze ER, Size: 10, 300K Steps", grid=false, tickdir=:out, color=cell_colors[args_list[i]["cell"]], ylim=(0, 200), linestyle=linestyles[i])
	end
	plt
end

# ╔═╡ 89216650-6460-414f-abf1-8dab7010f419
md"""
NumHidden: $(@bind nh MultiSelect(string.(dd_gaigru["numhidden"])))
"""

# ╔═╡ 089b0539-4157-4bed-ad99-010c380953e1
let
	
	
	args_list = [
		Dict("numhidden"=>6, "internal"=>33),
		Dict("numhidden"=>8, "internal"=>25),
		Dict("numhidden"=>10, "internal"=>17),
		Dict("numhidden"=>13, "internal"=>9),
		Dict("numhidden"=>20, "internal"=>40),
	]
	color_list = ["RNN", "AARNN", "MARNN", "FacMARNN", "GRU", "AAGRU"]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	plt = plot()
	for i ∈ 1:length(args_list)
		plt = plot!(data_gaigru, args_list[i], z=1.97, lw=3, label="(nh: $(args_list[i]["numhidden"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="Success Rate", xlabel="Episode", title="GAIGRU", grid=false, tickdir=:out, color=cell_colors[color_list[i]], ylim=(0.5, 1), fillalpha=0.2)
	end
	plt
end

# ╔═╡ 462ec3bb-4e31-4b3b-bdf1-b00216809e99
let
	
	
	args_list = [
		Dict("numhidden"=>6, "internal"=>33),
		Dict("numhidden"=>8, "internal"=>25),
		Dict("numhidden"=>10, "internal"=>17),
		Dict("numhidden"=>13, "internal"=>9),
		Dict("numhidden"=>20, "internal"=>40),
	]
	color_list = ["RNN", "AARNN", "MARNN", "FacMARNN", "GRU", "AAGRU"]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	plt = plot()
	for i ∈ 1:length(args_list)
		plt = plot!(data_steps_gaigru, args_list[i], z=1.97, lw=3, label="(nh: $(args_list[i]["numhidden"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="Steps", xlabel="Episode", title="GAIGRU", grid=false, tickdir=:out, color=cell_colors[color_list[i]], ylim=(0, 200), fillalpha=0.2)
	end
	plt
end

# ╔═╡ 427f2cf1-25b8-4b89-8525-bfcf7ddfe906
let
	
	
	args_list = [
		Dict("numhidden"=>6, "internal"=>65),
		Dict("numhidden"=>13, "internal"=>35),
		Dict("numhidden"=>18, "internal"=>28),
		Dict("numhidden"=>25, "internal"=>20),
		Dict("numhidden"=>30, "internal"=>100),
	]
	color_list = ["RNN", "AARNN", "MARNN", "FacMARNN", "GRU", "AAGRU"]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	plt = plot()
	for i ∈ 1:length(args_list)
		plt = plot!(data_gaiarnn, args_list[i], z=1.97, lw=3, label="(nh: $(args_list[i]["numhidden"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="Success Rate", xlabel="Episode", title="GAIARNN", grid=false, tickdir=:out, color=cell_colors[color_list[i]], ylim=(0.5, 1), fillalpha=0.2)
	end
	plt
end

# ╔═╡ 3878d6cb-f079-4abf-bc70-3b3b71bdb509
let
	
	
	args_list = [
		Dict("numhidden"=>6, "internal"=>65),
		Dict("numhidden"=>13, "internal"=>35),
		Dict("numhidden"=>18, "internal"=>28),
		Dict("numhidden"=>25, "internal"=>20),
		Dict("numhidden"=>30, "internal"=>100),
	]
	color_list = ["RNN", "AARNN", "MARNN", "FacMARNN", "GRU", "AAGRU"]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	plt = plot()
	for i ∈ 1:length(args_list)
		plt = plot!(data_steps_gaiarnn, args_list[i], z=1.97, lw=3, label="(nh: $(args_list[i]["numhidden"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="Steps", xlabel="Episode", title="GAIARNN", grid=false, tickdir=:out, color=cell_colors[color_list[i]], ylim=(0, 200), fillalpha=0.2)
	end
	plt
end

# ╔═╡ 571711af-7d10-42b9-a53c-41b9ec34e161
let
	
	
	args_list = [
		Dict("numhidden"=>6, "internal"=>52),
		Dict("numhidden"=>8, "internal"=>38),
		Dict("numhidden"=>10, "internal"=>27),
		Dict("numhidden"=>13, "internal"=>14),
		Dict("numhidden"=>15, "internal"=>7),
		Dict("numhidden"=>20, "internal"=>50)
	]
	color_list = ["RNN", "AARNN", "MARNN", "FacMARNN", "GRU", "AAGRU"]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	plt = plot()
	for i ∈ 1:length(args_list)
		plt = plot!(data_gaiagru, args_list[i], z=1.97, lw=3, label="(nh: $(args_list[i]["numhidden"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="Success Rate", xlabel="Episode", title="GAIAGRU", grid=false, tickdir=:out, color=cell_colors[color_list[i]], ylim=(0.5, 1), fillalpha=0.2)
	end
	plt
end

# ╔═╡ 215dfdfb-392a-4b4d-add7-4e2b363d5705
let
	
	
	args_list = [
		Dict("numhidden"=>6, "internal"=>52),
		Dict("numhidden"=>8, "internal"=>38),
		Dict("numhidden"=>10, "internal"=>27),
		Dict("numhidden"=>13, "internal"=>14),
		Dict("numhidden"=>15, "internal"=>7),
		Dict("numhidden"=>20, "internal"=>50)
	]
	color_list = ["RNN", "AARNN", "MARNN", "FacMARNN", "GRU", "AAGRU"]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	plt = plot()
	for i ∈ 1:length(args_list)
		plt = plot!(data_steps_gaiagru, args_list[i], z=1.97, lw=3, label="(nh: $(args_list[i]["numhidden"]))", palette=RPU.custom_colorant,legend=:topright, ylabel="Steps", xlabel="Episode", title="GAIAGRU", grid=false, tickdir=:out, color=cell_colors[color_list[i]], ylim=(0, 200), fillalpha=0.2)
	end
	plt
end

# ╔═╡ c5cb2048-33b1-4e42-ab14-cc8a79d494ac
let
	
	
	args_list = [
		Dict()
	]
	color_list = ["RNN", "AARNN", "MARNN", "FacMARNN", "GRU", "AAGRU"]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	plt = plot()
	for i ∈ 1:length(args_list)
		plt = plot!(data_gaugru, args_list[i], z=1.97, lw=3, label="(nh: 20)", palette=RPU.custom_colorant,legend=:topright, ylabel="Success Rate", xlabel="Episode", title="GAUGRU", grid=false, tickdir=:out, color=cell_colors[color_list[i]], ylim=(0.5, 1), fillalpha=0.2)
	end
	plt
end

# ╔═╡ c2da275f-3c50-4eeb-9eff-5363329e68ec
let
	
	
	args_list = [
		Dict()
	]
	color_list = ["RNN", "AARNN", "MARNN", "FacMARNN", "GRU", "AAGRU"]
	marker_shapes = [:circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross]
	plt = plot()
	for i ∈ 1:length(args_list)
		plt = plot!(data_steps_gaugru, args_list[i], z=1.97, lw=3, label="(nh: 20)", palette=RPU.custom_colorant,legend=:topright, ylabel="Steps", xlabel="Episode", title="GAUGRU", grid=false, tickdir=:out, color=cell_colors[color_list[i]], ylim=(0, 200), fillalpha=0.2)
	end
	plt
end

# ╔═╡ 864a5575-58dd-486a-a274-327a82d7c261
let	
	plt = plot()
	
	args_list = [
		Dict("cell"=>"ActionGatedRNN", "numhidden"=>10, "factors"=> 0, "internal"=>28),
		Dict("cell"=>"ActionGatedRNN", "numhidden"=>15, "factors"=> 0, "internal"=>21),
		Dict("cell"=>"ActionGatedRNN", "numhidden"=>20, "factors"=> 0, "internal"=>17),
		]
	names = ["GatedRNN 10", "GatedRNN 15", "GatedRNN 20"]
	for i in 1:length(args_list)
		plt = violin!(boxplot_data, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
	plt = boxplot!(boxplot_data, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=8, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="ER Directional TMaze, τ: 16", ylim=(0.3, 1))
		
		#plt = dotplot!(boxplot_data, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	plt

	#savefig("../../data/paper_plots/factored_tensor/dir_tmaze_box_plots_300k_steps_tau_16_fac_tensor.pdf")
end

# ╔═╡ 9b18dc96-50d0-4668-ad99-320502902ec4
let	
	plt = plot()
	
	args_list = [
		Dict("cell"=>"GRU", "numhidden"=>17, "factors"=> 0, "internal"=>0),
		Dict("cell"=>"AAGRU", "numhidden"=>17, "factors"=> 0, "internal"=>0),
		Dict("cell"=>"MAGRU", "numhidden"=>10, "factors"=> 0, "internal"=>0),
		Dict("cell"=>"FacMAGRU", "numhidden"=>15, "factors"=> 17, "internal"=>0),
		Dict("cell"=>"RNN", "numhidden"=>30, "factors"=> 0, "internal"=>0),
		Dict("cell"=>"AARNN", "numhidden"=>30, "factors"=> 0, "internal"=>0),
		Dict("cell"=>"MARNN", "numhidden"=>18, "factors"=> 0, "internal"=>0),
		Dict("cell"=>"FacMARNN", "numhidden"=>25, "factors"=> 15, "internal"=>0),
		Dict("cell"=>"ActionGatedRNN", "numhidden"=>10, "factors"=> 0, "internal"=>28),
		]
	names = ["GRU", "AAGRU", "MAGRU", "FacGRU", "RNN", "AARNN", "MARNN", "FacRNN",  "GatedRNN 10"]
	for i in 1:length(args_list)
		if i == 5
			plt = vline!([6], linestyle=:dot, color=:white, lw=2)
		end
		plt = violin!(boxplot_data, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
	plt = boxplot!(boxplot_data, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=7, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="ER Directional TMaze, τ: 12", ylim=(0.3, 1))
		
		#plt = dotplot!(boxplot_data, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	args_list = [
		Dict()
		]
	names = ["20"]
	for i in 1:length(args_list)
		plt = violin!(boxplot_data_gaugru, args_list[i], label=names[i], legend=false, color=cell_colors["ActionGatedRNN"], lw=2, linecolor=cell_colors["ActionGatedRNN"])
		
	plt = boxplot!(boxplot_data_gaugru, args_list[i], label=names[i], color=color=cell_colors["ActionGatedRNN"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=8, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="ER Directional TMaze, τ: 12", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data_gaugru, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	plt

	#savefig("../../data/paper_plots/factored_tensor/dir_tmaze_box_plots_300k_steps_tau_16_fac_tensor.pdf")
end

# ╔═╡ 74be388f-528d-4f56-bcde-a86dc2bdb756
let	
	plt = plot()
	
	args_list = [
		Dict("numhidden"=>6, "internal"=>33),
		Dict("numhidden"=>8, "internal"=>25),
		Dict("numhidden"=>10, "internal"=>17),
		Dict("numhidden"=>13, "internal"=>9),
		Dict("numhidden"=>10, "internal"=>30),
		Dict("numhidden"=>20, "internal"=>40),
		]
	names = ["33", "25", "17", "9", "30", "40"]
	for i in 1:length(args_list)
		plt = violin!(boxplot_data_gaigru, args_list[i], label=names[i], legend=false, color=cell_colors["ActionGatedRNN"], lw=2, linecolor=cell_colors["ActionGatedRNN"])
		
	plt = boxplot!(boxplot_data_gaigru, args_list[i], label=names[i], color=color=cell_colors["ActionGatedRNN"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=8, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="GAIGRU", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data_gaigru, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	plt

	#savefig("../../data/paper_plots/factored_tensor/dir_tmaze_box_plots_300k_steps_tau_16_fac_tensor.pdf")
end

# ╔═╡ 4d73f8cb-3848-4848-9894-a9c46ed44594
let	
	plt = plot()
	
	args_list = [
		Dict("numhidden"=>6, "internal"=>52),
		Dict("numhidden"=>8, "internal"=>38),
		Dict("numhidden"=>10, "internal"=>27),
		Dict("numhidden"=>13, "internal"=>14),
		Dict("numhidden"=>15, "internal"=>7),
		Dict("numhidden"=>20, "internal"=>50),
		]
	names = ["52", "38", "27", "14", "7", "50"]
	for i in 1:length(args_list)
		plt = violin!(boxplot_data_gaiagru, args_list[i], label=names[i], legend=false, color=cell_colors["ActionGatedRNN"], lw=2, linecolor=cell_colors["ActionGatedRNN"])
		
	plt = boxplot!(boxplot_data_gaiagru, args_list[i], label=names[i], color=color=cell_colors["ActionGatedRNN"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=8, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="GAIAGRU", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data_gaiagru, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	plt

	#savefig("../../data/paper_plots/factored_tensor/dir_tmaze_box_plots_300k_steps_tau_16_fac_tensor.pdf")
end

# ╔═╡ be41bdf6-6c72-4b0a-97b0-0926c3020d22
let	
	plt = plot()
	
	args_list = [
		Dict("numhidden"=>6, "internal"=>65),
		Dict("numhidden"=>13, "internal"=>35),
		Dict("numhidden"=>18, "internal"=>28),
		Dict("numhidden"=>25, "internal"=>20),
		Dict("numhidden"=>30, "internal"=>100),
		]
	names = ["65", "35", "28", "20", "100"]
	for i in 1:length(args_list)
		plt = violin!(boxplot_data_gaiarnn, args_list[i], label=names[i], legend=false, color=cell_colors["ActionGatedRNN"], lw=2, linecolor=cell_colors["ActionGatedRNN"])
		
	plt = boxplot!(boxplot_data_gaiarnn, args_list[i], label=names[i], color=color=cell_colors["ActionGatedRNN"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=8, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="GAIARNN", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data_gaiarnn, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	plt

	#savefig("../../data/paper_plots/factored_tensor/dir_tmaze_box_plots_300k_steps_tau_16_fac_tensor.pdf")
end

# ╔═╡ 168a727e-bce1-43d5-8c32-215fd370acb4
let	
	plt = plot()
	
	args_list = [
		Dict()
		]
	names = ["20"]
	for i in 1:length(args_list)
		plt = violin!(boxplot_data_gaugru, args_list[i], label=names[i], legend=false, color=cell_colors["ActionGatedRNN"], lw=2, linecolor=cell_colors["ActionGatedRNN"])
		
	plt = boxplot!(boxplot_data_gaugru, args_list[i], label=names[i], color=color=cell_colors["ActionGatedRNN"], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=8, grid=false, tickdir=:out, xguidefontsize=8, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="GAUGRU", ylim=(0.3, 1))
		
		plt = dotplot!(boxplot_data_gaugru, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0))
		
	end
	
	plt

	#savefig("../../data/paper_plots/factored_tensor/dir_tmaze_box_plots_300k_steps_tau_16_fac_tensor.pdf")
end

# ╔═╡ Cell order:
# ╠═817410e7-a12d-4318-bf92-be86d32f45b6
# ╠═0f11b684-a3f9-4d40-9c63-33b28136d4f5
# ╠═95498906-eea1-11eb-26f4-8574b531e0dc
# ╠═78632921-6931-4ca0-9e8a-095a7f2c99b0
# ╠═8b4861e4-cfbc-4a0e-8246-7d6182c83586
# ╠═edd86b13-bdb5-4811-8e0a-36230d1162ae
# ╠═1c2a5244-0999-44f2-a952-3f495a8ec136
# ╠═f197d16b-eb04-408c-a020-9023f2eb02b2
# ╠═a67aba27-40ef-4ba2-816a-6da61a1d523b
# ╠═f9c41407-995c-4ca5-b5d4-46bce5f9585a
# ╠═3a1b767e-d09b-4079-b2a5-7e2e189a1600
# ╠═606b1f70-976d-4adc-9d1b-a34835b947e4
# ╠═a1bde34d-7216-4dc0-a08e-eae8a03148a6
# ╠═be2a5e97-a857-4c7b-bb50-ef3248cdd54f
# ╠═2e0ff5cc-6758-4cf1-930b-c55f88255e15
# ╠═54cae2a9-e112-4b61-8f39-0dd68deca36f
# ╠═1b39ba62-7603-450c-a41c-5c67c786ef73
# ╠═6e13f656-5db2-4913-ae77-232f4d1cc770
# ╠═e3c83d46-f6f0-4d8a-969a-11579c3ca3ce
# ╠═c36d3791-8828-48c7-adcd-fb0c0d5547b7
# ╠═357622c5-491f-400d-ac29-2f3a312a7f8c
# ╠═eb740dd2-b236-4f98-9a32-db0e12887595
# ╠═e29dbabe-be8a-45c5-9b01-68c7d49c306d
# ╠═d944303f-cdf4-413e-8932-a04507168eab
# ╠═9e8614a4-7e20-404b-a025-77ef51947dde
# ╠═a41a0df4-670a-474b-8453-2130d67137ba
# ╠═41d54ec8-aeac-40b2-a682-f9724baf4c64
# ╠═8c206951-8f90-4a68-be21-12100ba713c1
# ╠═01462edf-e99c-4b7a-9f5b-3db66954b77b
# ╟─90b1ae5e-43f6-498b-9a00-7979b651935e
# ╟─a8c37334-01c6-484a-8e77-e7d023db891d
# ╟─6b1e0070-edf6-4125-80f6-45aa497ece18
# ╟─3d9d5add-a484-4fbc-9d0c-54b06936485d
# ╟─14ad8703-dca4-42a4-9bdc-36ad0fb2e3b5
# ╟─14e4845e-c8f1-46c1-805f-1cf2b8c2ab52
# ╠═89216650-6460-414f-abf1-8dab7010f419
# ╟─089b0539-4157-4bed-ad99-010c380953e1
# ╟─462ec3bb-4e31-4b3b-bdf1-b00216809e99
# ╟─427f2cf1-25b8-4b89-8525-bfcf7ddfe906
# ╟─3878d6cb-f079-4abf-bc70-3b3b71bdb509
# ╟─571711af-7d10-42b9-a53c-41b9ec34e161
# ╟─215dfdfb-392a-4b4d-add7-4e2b363d5705
# ╟─c5cb2048-33b1-4e42-ab14-cc8a79d494ac
# ╟─c2da275f-3c50-4eeb-9eff-5363329e68ec
# ╟─864a5575-58dd-486a-a274-327a82d7c261
# ╟─9b18dc96-50d0-4668-ad99-320502902ec4
# ╟─74be388f-528d-4f56-bcde-a86dc2bdb756
# ╟─4d73f8cb-3848-4848-9894-a9c46ed44594
# ╟─be41bdf6-6c72-4b0a-97b0-0926c3020d22
# ╟─168a727e-bce1-43d5-8c32-215fd370acb4
