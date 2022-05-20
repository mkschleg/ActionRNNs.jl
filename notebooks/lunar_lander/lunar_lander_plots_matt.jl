### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 86f1ffa0-bbd9-11eb-25df-0b3f7d5ef725
using Revise

# ╔═╡ d08f12ef-1e7d-4437-920b-7ee85168eb56
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 2788583f-e22c-471c-a6c7-62acb0f604e3
const RPU = ReproducePlotUtils

# ╔═╡ e183be9a-13e8-42af-813d-14bce5b65550
# push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ dbea3d82-da81-49c9-8dc2-393acaf907ea
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

# ╔═╡ 5c75ef81-ec0c-471d-8a58-79966721919f
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[1],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[end-1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ 7014323e-cbd9-4f45-a50f-cde05ac0b125
ic, dd = let
	ic_gru, dd = RPU.load_data("../local_data/lunar_lander_er_rmsprop_os6_sc2_gru_4M/")
	ic_magru,dd = 
		RPU.load_data("../local_data/lunar_lander_er_rmsprop_os6_sc2_magru_4M/")
	ic_aagru,dd = 
		RPU.load_data("../local_data/lunar_lander_er_rmsprop_os6_sc2_aagru_4M/")
	ic_facmagru_150,dd = 
		RPU.load_data("../local_data/lunar_lander_er_rmsprop_os6_sc2_facmagru_4M/")
	ic_facmagru_100,dd = 
		RPU.load_data("../local_data/lunar_lander_er_rmsprop_os6_sc2_facmagru100_4M/")
	ic_facmagru_64,dd = 
		RPU.load_data("../local_data/lunar_lander_er_rmsprop_os6_sc2_facmagru64_4M/")
	ic = ItemCollection([ic_gru.items; ic_magru.items; ic_aagru.items; ic_facmagru_150.items; ic_facmagru_100.items; ic_facmagru_64.items])
	ic, diff(ic)
end

# ╔═╡ 7ff25cc1-e7d3-4c6c-8398-14a52ddb43de
FileIO.load(joinpath(ic[1].folder_str, "results.jld2"))["results"]

# ╔═╡ 1148a5d9-4f62-430f-be97-ad237636f0a9
data = RPU.get_line_data_for(
	ic,
	["save_dir"],
	["eta"];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews, 0.1),
	get_data=(x)->RPU.get_MUE(x, :total_rews, 0.1))

# ╔═╡ cfa21050-8b6e-4e0f-a9b4-0d3c0cc58374
data[1]

# ╔═╡ bd1af140-5e9b-4fae-b71c-ef7d6660a32d
let
	save_dir = "lunar_lander_er_rmsprop_os6_sc2_gru_4M"
	plt = plot()
	boxplot(data, 		               
		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"), 
		legend=nothing, label="GRU", color=color_scheme[1], label_idx=nothing)
	save_dir = "lunar_lander_er_rmsprop_os6_sc2_aagru_4M"
	boxplot!(data, 
		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
		label_idx=nothing, label="AAGRU", color=color_scheme[2])
	save_dir = "lunar_lander_er_rmsprop_os6_sc2_magru_4M"
	boxplot!(data, 
		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
		label_idx=nothing, label="MAGRU", color=color_scheme[3])

	save_dir = "lunar_lander_er_rmsprop_os6_sc2_facmagru_4M"
	boxplot!(data, 
		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
		label_idx=nothing, label="FacMAGRU150", color=color_scheme[4])
	save_dir = "lunar_lander_er_rmsprop_os6_sc2_facmagru100_4M"
	boxplot!(data, 
		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
		label_idx=nothing, label="FacMAGRU100", color=color_scheme[5])
	save_dir = "lunar_lander_er_rmsprop_os6_sc2_facmagru64_4M"
	boxplot!(data, 
		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
		label_idx=nothing, label="FacMAGRU64", color=color_scheme[6])
	
	save_dir = "lunar_lander_er_rmsprop_os6_sc2_gru_4M"
	dotplot!(data, 		               
		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"), 
		legend=nothing, label="GRU", color=color_scheme[1], label_idx=nothing)
	save_dir = "lunar_lander_er_rmsprop_os6_sc2_aagru_4M"
	dotplot!(data, 
		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
		label_idx=nothing, label="AAGRU", color=color_scheme[2])
	save_dir = "lunar_lander_er_rmsprop_os6_sc2_magru_4M"
	dotplot!(data, 
		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
		label_idx=nothing, label="MAGRU", color=color_scheme[3])

	save_dir = "lunar_lander_er_rmsprop_os6_sc2_facmagru_4M"
	dotplot!(data, 
		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
		label_idx=nothing, label="FacMAGRU150", color=color_scheme[4])
	save_dir = "lunar_lander_er_rmsprop_os6_sc2_facmagru100_4M"
	dotplot!(data, 
		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
		label_idx=nothing, label="FacMAGRU100", color=color_scheme[5])
	save_dir = "lunar_lander_er_rmsprop_os6_sc2_facmagru64_4M"
	dotplot!(data, 
		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
		label_idx=nothing, label="FacMAGRU64", color=color_scheme[6])
	
# 	save_dir = "lunar_lander_er_rmsprop_os6_sc2_gru_4M"
# 	dotplot!(data, 		               
# 		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"), 
# 		label_idx="cell", color=color_scheme[1])
# 	save_dir = "lunar_lander_er_rmsprop_os6_sc2_aagru_4M"
# 	dotplot!(data, 
# 		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
# 		label_idx="cell", color=color_scheme[2])
# 	save_dir = "lunar_lander_er_rmsprop_os6_sc2_magru_4M"
# 	dotplot!(data, 
# 		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
# 		label_idx="cell", color=color_scheme[3])

# 	save_dir = "lunar_lander_er_rmsprop_os6_sc2_facmagru_4M"
# 	dotplot!(data, 
# 		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
# 		label_idx="cell", color=color_scheme[4])
# 	save_dir = "lunar_lander_er_rmsprop_os6_sc2_facmagru100_4M"
# 	dotplot!(data, 
# 		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
# 		label_idx="cell", x=["FacMAGRU100"], color=color_scheme[5])
# 	save_dir = "lunar_lander_er_rmsprop_os6_sc2_facmagru64_4M"
# 	dotplot!(data, 
# 		Dict("save_dir"=>"/home/vladtk/scratch/ActionRNNs/$(save_dir)/data"),
# 		label_idx="cell", x=["FacMAGRU64"], color=color_scheme[6])
	
end

# ╔═╡ 1831a46b-58cb-4bf5-b799-ac8b9d41a27c
begin
	# @recipe function f(::Type{Val{:controldist}}, x, y)
	# 	violin!(x, y, color=cell_colors[x], show_mean=true)
	# 	dotplot!(x, y, color=:black)
	# 	boxplot!(x, y, color=cell_colors[x], fillalpha=0.75)
	# end
	# push!(RPU.stats_plot_types, :controldist)

			
end


# ╔═╡ 375ea721-510a-4e2e-ba76-7ef42dc6422a
ic_final, dd_final = let
	ic_aagru, dd = RPU.load_data("../local_data/lunar_lander_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_aagru_4M/")
	ic_tmp,dd = 
		RPU.load_data("../local_data/lunar_lander_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_magru_4M/")
	ic_magru = search(ic_tmp, Dict("eta"=>dd["eta"][2]))
	ic_gru,dd = 
		RPU.load_data("../local_data/lunar_lander_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_gru_4M/")
	ic_facmagru_150,dd = 
		RPU.load_data("../local_data/lunar_lander_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru152_4M")
	ic_facmagru_100,dd = 
		RPU.load_data("../local_data/lunar_lander_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru100_4M")
	ic_facmagru_64,dd = 
		RPU.load_data("../local_data/lunar_lander_final/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru64_4M")
	ic = ItemCollection([ic_gru.items; ic_magru.items; ic_aagru.items; ic_facmagru_150.items; ic_facmagru_100.items; ic_facmagru_64.items])
	# ic = ItemCollection([ic_aagru.items; ic_magru.items; ic_gru.items; ic_facmagru_150.items])
	ic, diff(ic)
end

# ╔═╡ 2e32d31b-e16d-4a67-9f06-3b3aa97823f1
data_final = RPU.get_line_data_for(
	ic_final,
	["save_dir", "cell"],
	[];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews, 0.1),
	get_data=(x)->RPU.get_MUE(x, :total_rews))

# ╔═╡ 50a61077-5c0a-4162-bac8-e9a1b00ddb40
let
	plt = plot(legend=false, grid=false, tickfontsize=9, tickdir=:out)
	
# "/home/vladtk/scratch/ActionRNNs/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru100_4M/data"
# 3
# "/home/vladtk/scratch/ActionRNNs/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru152_4M/data"
# 4
# "/home/vladtk/scratch/ActionRNNs/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru64_4M/data"
	
	violin!(data_final, Dict("cell"=>"GRU"), label_idx="cell", legend=false, color=cell_colors["GRU"], lw=1, linecolor=cell_colors["GRU"])
	StatsPlots.boxplot!(data_final, Dict("cell"=>"GRU"), label_idx="cell", color=cell_colors["GRU"], fillalpha=0.75, outliers=false, lw=2, linecolor=:black)
	# dotplot!(data_final, Dict("cell"=>"GRU"), label_idx="cell", color=:black)
	

	# plot!(data_final, Dict("cell"=>"MAGRU"), seriestype=:controldist)
	
	violin!(data_final, Dict("cell"=>"AAGRU"), label_idx="cell", color=cell_colors["AAGRU"], lw=1, linecolor=cell_colors["AAGRU"])
	boxplot!(data_final, Dict("cell"=>"AAGRU"), label_idx="cell", color=cell_colors["AAGRU"], fillalpha=0.75, outliers=false, lw=2, linecolor=:black)
	# dotplot!(data_final, Dict("cell"=>"AAGRU"), label_idx="cell", color=:black)
	
	
	violin!(data_final, Dict("cell"=>"MAGRU"), label_idx="cell", color=cell_colors["MAGRU"], lw=1, linecolor=cell_colors["MAGRU"])
	boxplot!(data_final, Dict("cell"=>"MAGRU"), label_idx="cell", color=cell_colors["MAGRU"], fillalpha=0.75, outliers=false, lw=2, linecolor=:black)
	# dotplot!(data_final, Dict("cell"=>"MAGRU"), label_idx="cell", color=:black)
	
	nh = 64
	cell = "FacMAGRU"; save_dir = "/home/vladtk/scratch/ActionRNNs/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru$(nh)_4M/data"
	violin!(data_final, Dict("cell"=>cell, "save_dir"=>save_dir), label="FacGRU: $(nh)", color=cell_colors[cell], lw=1, linecolor=cell_colors["FacMAGRU"])
	boxplot!(data_final, Dict("cell"=>cell, "save_dir"=>save_dir), label="FacGRU: $(nh)", color=cell_colors[cell], fillalpha=0.75, outliers=false, lw=2, linecolor=:black)
	# dotplot!(data_final, Dict("cell"=>cell, "save_dir"=>save_dir), label="FacGRU: $(nh)", color=:black)
	
	nh = 100
	cell = "FacMAGRU"; save_dir = "/home/vladtk/scratch/ActionRNNs/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru$(nh)_4M/data"
	violin!(data_final, Dict("cell"=>cell, "save_dir"=>save_dir), label="FacGRU: $(nh)", color=cell_colors[cell], lw=1, linecolor=cell_colors["FacMAGRU"])
	boxplot!(data_final, Dict("cell"=>cell, "save_dir"=>save_dir), label="FacGRU: $(nh)", color=cell_colors[cell], fillalpha=0.75, outliers=false, lw=2, linecolor=:black)
	# dotplot!(data_final, Dict("cell"=>cell, "save_dir"=>save_dir), label="FacGRU: $(nh)", color=:black)
	
	nh = 152
	cell = "FacMAGRU"; save_dir = "/home/vladtk/scratch/ActionRNNs/final_lunar_lander_er_relu_rmsprop_os6_sc2_facmagru$(nh)_4M/data"
	violin!(data_final, Dict("cell"=>cell, "save_dir"=>save_dir), label="FacGRU: $(nh)", color=cell_colors[cell], lw=1, linecolor=cell_colors["FacMAGRU"])
	boxplot!(data_final, Dict("cell"=>cell, "save_dir"=>save_dir), label="FacGRU: $(nh)", color=cell_colors[cell], fillalpha=0.75, outliers=false, lw=2, linecolor=:black)
	# dotplot!(data_final, Dict("cell"=>cell, "save_dir"=>save_dir), label="FacGRU: $(nh)", color=:black)
	
	savefig("../lunar_lander_final_mue.pdf")
	plt

end

# ╔═╡ 3617346d-3859-43ff-8586-aa5490b2b9c5
data_ln = RPU.get_line_data_for(
	ic_final,
	["cell", "save_dir"],
	[];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews, 0.1),
	get_data=(x)->rollmean(x["results"][:total_rews], 100))

# ╔═╡ a952ad12-af9c-431c-99fb-a782afe9aa36
let
	plot(data_ln, Dict("cell"=>"MAGRU"), label="MAGRU")
	plot!(data_ln, Dict("cell"=>"AAGRU"), label="AAGRU")
	plot!(data_ln, Dict("cell"=>"GRU"), label="GRU")
end

# ╔═╡ 68b7a5ed-08f0-4b87-a8b6-2a9349b45d0f
function get_extended_line_mean(ddict, key1, key2; n=0, cutoff=4000000)
    ret = zeros(eltype(ddict["results"][key1]), sum(ddict["results"][key2]))
    cur_idx = 1
    for i in 1:length(ddict["results"][key1])
        ret[cur_idx:(cur_idx + ddict["results"][key2][i] - 1)] .= ddict["results"][key1][i]/ddict["results"][key2][i]
        cur_idx += ddict["results"][key2][i]
    end

    if n == 0
        ret
    else
        # rollmean(ret, n)[1:n:end]
		mean(reshape(ret[1:cutoff], n, :), dims=1)[1, :]
    end
end

# ╔═╡ eef5678f-e093-4d53-9996-cdda2ddbadd7
data_ext_ln = RPU.get_line_data_for(
	ic_final,
	["cell", "save_dir"],
	[];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews, 0.1),
	get_data=(x)->get_extended_line_mean(x, :total_rews, :total_steps, n=10000))

# ╔═╡ 65c50cff-997b-4b56-a295-613c26f76db7
let
	plot(data_ext_ln, Dict("cell"=>"MAGRU"), label="MAGRU")
	plot!(data_ext_ln, Dict("cell"=>"AAGRU"), label="AAGRU")
	plot!(data_ext_ln, Dict("cell"=>"GRU"), label="GRU", legend=nothing)
end

# ╔═╡ 6d564838-32c0-4038-abb5-c42bf3259c5e
data_ln_stps = RPU.get_line_data_for(
	ic_final,
	["cell", "numhidden", "save_dir"],
	[];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :total_rews, 0.1),
	get_data=(x)->rollmean(x["results"][:total_steps], 1000))

# ╔═╡ 11b014d5-2f62-468a-84ee-8664a4e2b726
let

	plt = plot(xlabel="episode", ylabel="steps in episode", grid=false, tickdir=:out, tickfontsize=12)
	plot!(data_ln_stps, Dict("cell"=>"GRU"), label="GRU", color=cell_colors["GRU"])
	plot!(data_ln_stps, Dict("cell"=>"AAGRU"), label="AAGRU", color=cell_colors["AAGRU"])
	plot!(data_ln_stps, Dict("cell"=>"MAGRU"), label="MAGRU", color=cell_colors["MAGRU"])
	savefig("../numsteps_ll.pdf")
	plt
end

# ╔═╡ c1288962-7045-4285-8481-03ba3c1c5cfe
let

	plt = plot(xlabel="episode", ylabel="steps in episode", grid=false, tickdir=:out, tickfontsize=12)
	plot!(data_ln_stps, Dict("cell"=>"FacMAGRU", "numhidden"=>64), label="FacGRU", palette=color_scheme, ls=:auto)
	plot!(data_ln_stps, Dict("cell"=>"FacMAGRU", "numhidden"=>100), label="FacGRU", palette=color_scheme, ls=:auto)
	plot!(data_ln_stps, Dict("cell"=>"FacMAGRU", "numhidden"=>152), label="FacGRU", palette=color_scheme, ls=:auto)
	savefig("../numsteps_ll_fac.pdf")
	plt
end

# ╔═╡ Cell order:
# ╠═86f1ffa0-bbd9-11eb-25df-0b3f7d5ef725
# ╠═d08f12ef-1e7d-4437-920b-7ee85168eb56
# ╠═2788583f-e22c-471c-a6c7-62acb0f604e3
# ╠═e183be9a-13e8-42af-813d-14bce5b65550
# ╠═dbea3d82-da81-49c9-8dc2-393acaf907ea
# ╠═5c75ef81-ec0c-471d-8a58-79966721919f
# ╠═7014323e-cbd9-4f45-a50f-cde05ac0b125
# ╠═7ff25cc1-e7d3-4c6c-8398-14a52ddb43de
# ╠═1148a5d9-4f62-430f-be97-ad237636f0a9
# ╠═cfa21050-8b6e-4e0f-a9b4-0d3c0cc58374
# ╠═bd1af140-5e9b-4fae-b71c-ef7d6660a32d
# ╠═1831a46b-58cb-4bf5-b799-ac8b9d41a27c
# ╠═375ea721-510a-4e2e-ba76-7ef42dc6422a
# ╠═2e32d31b-e16d-4a67-9f06-3b3aa97823f1
# ╠═50a61077-5c0a-4162-bac8-e9a1b00ddb40
# ╠═3617346d-3859-43ff-8586-aa5490b2b9c5
# ╠═a952ad12-af9c-431c-99fb-a782afe9aa36
# ╠═68b7a5ed-08f0-4b87-a8b6-2a9349b45d0f
# ╠═eef5678f-e093-4d53-9996-cdda2ddbadd7
# ╠═65c50cff-997b-4b56-a295-613c26f76db7
# ╠═6d564838-32c0-4038-abb5-c42bf3259c5e
# ╠═11b014d5-2f62-468a-84ee-8664a4e2b726
# ╠═c1288962-7045-4285-8481-03ba3c1c5cfe
