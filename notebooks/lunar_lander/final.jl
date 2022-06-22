### A Pluto.jl notebook ###
# v0.19.3

using Markdown
using InteractiveUtils

# ╔═╡ e765b282-90ca-4c76-81d0-b78a75c150c5
let
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ 7d08af30-bc3c-11eb-1ad7-8fc5f7a1ec61
using Revise

# ╔═╡ 7b444b3b-0f11-425c-aac6-877586515b41
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ dc89338c-ebd4-4f8c-941f-23717d842da7
using StatsBase, Bootstrap

# ╔═╡ f8dcf929-a3a3-44e0-bc25-9782ed4fbfb9
const RPU = ReproducePlotUtils

# ╔═╡ f1d06ce5-cd2b-4955-9a54-7f43e2bf7c0d
pwd()

# ╔═╡ 5a4c004b-4db1-47ea-be17-06e4ba081c05
at(dir) = joinpath("/home/matt/Documents/ActionRNNs.jl/local_data", dir)

# ╔═╡ ce31c440-0fe3-47a6-a035-fe1dac8f56d8
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

# ╔═╡ f0c21ef8-0dbe-4ae3-8c51-ac1674d931fb
cell_colors = Dict(
	"RNN" => color_scheme[3],
	"AARNN" => color_scheme[end],
	"MARNN" => color_scheme[5],
	"FacMARNN" => color_scheme[1],
	"GRU" => color_scheme[4],
	"AAGRU" => color_scheme[2],
	"MAGRU" => color_scheme[6],
	"FacMAGRU" => color_scheme[end-2])

# ╔═╡ c44265f4-3656-4fdc-98a5-57344b634e33
ic, dd = RPU.load_data(at("dir_tmaze_er/final_dir_tmaze_er_rnn_rmsprop_10_2/"))

# ╔═╡ c940c7be-3697-4a5f-bf75-4caeb284b90a
let
	sub_ic = search(ic) do itm
		itm.parsed_args["cell"][end-2:end] == "GRU"
	end
	diff(sub_ic)
end

# ╔═╡ 51458b14-665e-487f-91f7-f55606a09f15
ic_fac, dd_fac = 
	RPU.load_data(at("dir_tmaze_er/final_fac_dir_tmaze_er_rnn_rmsprop_10_2_300k/"))

# ╔═╡ 5e49469b-2bff-411a-a8dc-226fde5eaf41
data = RPU.get_line_data_for(
	ic,
	["cell"],
	[];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes, 0.1))

# ╔═╡ 0db6c0b2-3ce3-4f36-b84c-76031143f970
let
	boxplot(data, Dict("cell"=>"GRU"), label_idx="cell", color=cell_colors["GRU"], legend=nothing, outliers=false)
	boxplot!(data, Dict("cell"=>"AAGRU"), label_idx="cell", color=cell_colors["AAGRU"], outliers=false)
	boxplot!(data, Dict("cell"=>"MAGRU"), label_idx="cell", color=cell_colors["MAGRU"], outliers=false)

	boxplot!(data, Dict("cell"=>"RNN"), label_idx="cell", color=cell_colors["RNN"], outliers=false)
	boxplot!(data, Dict("cell"=>"AARNN"), label_idx="cell", color=cell_colors["AARNN"], outliers=false)
	boxplot!(data, Dict("cell"=>"MARNN"), label_idx="cell", color=cell_colors["MARNN"], outliers=false)
	
	dotplot!(data, Dict("cell"=>"GRU"), label_idx="cell", color=cell_colors["GRU"])
	dotplot!(data, Dict("cell"=>"AAGRU"), label_idx="cell", color=cell_colors["AAGRU"])
	dotplot!(data, Dict("cell"=>"MAGRU"), label_idx="cell", color=cell_colors["MAGRU"])

	dotplot!(data, Dict("cell"=>"RNN"), label_idx="cell", color=cell_colors["RNN"])
	dotplot!(data, Dict("cell"=>"AARNN"), label_idx="cell", color=cell_colors["AARNN"])
	dotplot!(data, Dict("cell"=>"MARNN"), label_idx="cell", color=cell_colors["MARNN"])
end

# ╔═╡ 6fc0bb4b-f202-4b29-8ee1-bb3d768069c9
data_fac_sens = RPU.get_line_data_for(
	ic_fac,
	["cell", "factors", "replay_size"],
	[];
	comp=:max,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes, 0.1))

# ╔═╡ 03841f34-7492-4f73-ac8a-596378c70af7
let
	plot(data_fac_sens, Dict("cell"=>"FacMAGRU", "replay_size"=>10000), 
		sort_idx="factors", lw=2, color=cell_colors["FacMAGRU"])
	plot!(data_fac_sens, Dict("cell"=>"FacMARNN", "replay_size"=>10000), 
		sort_idx="factors", lw=2, color=cell_colors["FacMARNN"])
end

# ╔═╡ 75c2815a-191c-4b2e-96d1-b4b857f204ba
let
	plot(data_fac_sens, Dict("cell"=>"FacMAGRU", "replay_size"=>20000), 
		sort_idx="factors", lw=2, color=cell_colors["FacMAGRU"])
	plot!(data_fac_sens, Dict("cell"=>"FacMARNN", "replay_size"=>20000), 
		sort_idx="factors", lw=2, color=cell_colors["FacMARNN"])
end

# ╔═╡ 7cff19e5-9348-4337-baba-0a3ef1508e64
let
	plot(data_fac_sens, Dict("cell"=>"FacMAGRU", "replay_size"=>20000), 
		sort_idx="factors", lw=2, color=cell_colors["FacMAGRU"], legend=nothing, marker=:circle)
	params = Dict("cell"=>"MAGRU")
	idx = findfirst(data.data) do ld
		all([ld.line_params[k] == params[k] for k in keys(params)])
	end
	d = data.data[idx].data
	plot!(dd_fac["factors"], fill(mean(d), length(dd_fac["factors"])), yerror=fill(sqrt(var(d)/length(d)), length(dd_fac["factors"])), color=cell_colors["MAGRU"], marker=:circle, lw=2)
	
	params = Dict("cell"=>"AAGRU")
	idx = findfirst(data.data) do ld
		all([ld.line_params[k] == params[k] for k in keys(params)])
	end
	d = data.data[idx].data
	plot!(dd_fac["factors"], fill(mean(d), length(dd_fac["factors"])), yerror=fill(sqrt(var(d)/length(d)), length(dd_fac["factors"])), color=cell_colors["AAGRU"], marker=:circle, lw=2)
end

# ╔═╡ d63bce66-2f48-458e-85cc-1f993d4ac85e
let
	plt_rnn = plot(data_fac_sens, Dict("cell"=>"FacMARNN", "replay_size"=>10000), 
		sort_idx="factors", lw=2, color=cell_colors["FacMARNN"], legend=nothing, marker=:circle, ylabel="Perc Success", title="RNN", grid=false, tickdir=:out)
	params = Dict("cell"=>"MARNN")
	idx = findfirst(data.data) do ld
		all([ld.line_params[k] == params[k] for k in keys(params)])
	end
	d = data.data[idx].data
	plot!(dd_fac["factors"], fill(mean(d), length(dd_fac["factors"])), yerror=fill(sqrt(var(d)/length(d)), length(dd_fac["factors"])), color=cell_colors["MARNN"], marker=:circle, lw=2)
	
	params = Dict("cell"=>"AARNN")
	idx = findfirst(data.data) do ld
		all([ld.line_params[k] == params[k] for k in keys(params)])
	end
	d = data.data[idx].data
	plot!(dd_fac["factors"], fill(mean(d), length(dd_fac["factors"])), yerror=fill(sqrt(var(d)/length(d)), length(dd_fac["factors"])), color=cell_colors["AARNN"], marker=:circle, lw=2)

	plt_gru = plot(data_fac_sens, Dict("cell"=>"FacMAGRU", "replay_size"=>10000), 
		sort_idx="factors", lw=2, color=cell_colors["FacMAGRU"], legend=nothing, marker=:circle, xlabel="Factors", ylabel="Perc Success", title="GRU", grid=false, tickdir=:out)
	params = Dict("cell"=>"MAGRU")
	idx = findfirst(data.data) do ld
		all([ld.line_params[k] == params[k] for k in keys(params)])
	end
	d = data.data[idx].data
	plot!(dd_fac["factors"], fill(mean(d), length(dd_fac["factors"])), yerror=fill(sqrt(var(d)/length(d)), length(dd_fac["factors"])), color=cell_colors["MAGRU"], marker=:circle, lw=2)
	
	params = Dict("cell"=>"AAGRU")
	idx = findfirst(data.data) do ld
		all([ld.line_params[k] == params[k] for k in keys(params)])
	end
	d = data.data[idx].data
	plot!(dd_fac["factors"], fill(mean(d), length(dd_fac["factors"])), yerror=fill(sqrt(var(d)/length(d)), length(dd_fac["factors"])), color=cell_colors["AAGRU"], marker=:circle, lw=2)

	plt = plot(plt_rnn, plt_gru, layout = (2, 1))
	# savefig("../plots/dirtmaze_factor_sens_10000.pdf")
	plt
end

# ╔═╡ beeb9b36-8c04-4de6-8297-82ccb82b368f
let
	plot(data_fac_sens, Dict("cell"=>"FacMARNN", "replay_size"=>20000), 
		sort_idx="factors", lw=2, color=cell_colors["FacMARNN"], legend=nothing, marker=:circle)
	params = Dict("cell"=>"MARNN")
	idx = findfirst(data.data) do ld
		all([ld.line_params[k] == params[k] for k in keys(params)])
	end
	d = data.data[idx].data
	plot!(dd_fac["factors"], fill(mean(d), length(dd_fac["factors"])), yerror=fill(sqrt(var(d)/length(d)), length(dd_fac["factors"])), color=cell_colors["MARNN"], marker=:circle, lw=2)
	
	params = Dict("cell"=>"AARNN")
	idx = findfirst(data.data) do ld
		all([ld.line_params[k] == params[k] for k in keys(params)])
	end
	d = data.data[idx].data
	plot!(dd_fac["factors"], fill(mean(d), length(dd_fac["factors"])), yerror=fill(sqrt(var(d)/length(d)), length(dd_fac["factors"])), color=cell_colors["AARNN"], marker=:circle, lw=2)
end

# ╔═╡ 7556fc88-9eb9-4813-bee7-1cf4d032ec0d


# ╔═╡ b176461f-be40-4693-830a-512965db3642
let
	plt = plot(tickfontsize=11, grid=false, tickdir=:out)
	for cell ∈ ["GRU", "AAGRU", "MAGRU"]
		violin!(data, Dict("cell"=>cell), label_idx="cell", color=cell_colors[cell], legend=nothing, outliers=false, lw=1, linecolor=cell_colors[cell])
		boxplot!(data, Dict("cell"=>cell), label_idx="cell", color=cell_colors[cell], legend=nothing, outliers=true, lw=2, fillalpha=0.75, linecolor=:black)
	end
	
	cell="FacMAGRU"
	
	violin!(data_fac_sens, Dict("cell"=>cell,"replay_size"=>10000, "factors"=>15), label="FacGRU", color=cell_colors[cell], legend=nothing, outliers=false, lw=1, linecolor=cell_colors[cell])
	boxplot!(data_fac_sens, Dict("cell"=>cell, "replay_size"=>10000, "factors"=>15), label="FacGRU", color=cell_colors[cell], legend=nothing, outliers=true, lw=2, fillalpha=0.75, linecolor=:black)
	
	plt = vline!([6], linestyle=:dot, color=:white, lw=2)
	
	for cell ∈ ["RNN", "AARNN", "MARNN"]
		violin!(data, Dict("cell"=>cell), label_idx="cell", color=cell_colors[cell], legend=nothing, outliers=false, lw=1, linecolor=cell_colors[cell])
		boxplot!(data, Dict("cell"=>cell), label_idx="cell", color=cell_colors[cell], legend=nothing, outliers=true, lw=2, fillalpha=0.75, linecolor=:black)
	end
	
	cell="FacMARNN"
	
	violin!(data_fac_sens, Dict("cell"=>cell,"replay_size"=>10000, "factors"=>17), label="FacRNN", color=cell_colors[cell], legend=nothing, outliers=false,  lw=1, linecolor=cell_colors[cell])
	boxplot!(data_fac_sens, Dict("cell"=>cell, "replay_size"=>10000, "factors"=>17), label="FacRNN", color=cell_colors[cell], legend=nothing, outliers=true, lw=2, fillalpha=0.75, linecolor=:black)
	
	# savefig("../plots/dirtmaze_er_bp.pdf")
	
	plt
end

# ╔═╡ b50f8b6a-60ee-4b16-8e7b-ec9b5eff5fd3
let
	idx_ma = findfirst(data.data) do ld
		ld.line_params["cell"]=="MAGRU"
	end
	d_ma = data.data[idx_ma].data
	
	bs_size = 10^4
	
	d_bs_ma = [mean(rand(d_ma, length(d_ma))) for i in 1:bs_size]
	
	mean(d_bs_ma), sqrt(var(d_bs_ma)/bs_size)

end


# ╔═╡ 1d58d39d-04b1-4cf5-8a59-887a4b99120b
let
	B = 10^5
	
	idx_ma = findfirst(data.data) do ld
		ld.line_params["cell"]=="MAGRU"
	end
	d_ma = data.data[idx_ma].data
	d_bs_ma = [sample(d_ma, length(d_ma), replace=true) for i in 1:B]
	
	idx_aa = findfirst(data.data) do ld
		ld.line_params["cell"]=="AAGRU"
	end
	d_aa = data.data[idx_aa].data
	d_bs_aa = [sample(d_aa, length(d_aa), replace=true) for i in 1:B]
	
	abs_median(X, Y) = abs(median(X) - median(Y))
	abs_mean(X, Y) = abs(mean(X) - mean(Y))
	
	t1 = abs_median(d_ma, d_aa)
	t2 = abs_mean(d_ma, d_aa)
	
	t1_bs = abs_median.(d_bs_ma, d_bs_aa)
	t2_bs = abs_mean.(d_bs_ma, d_bs_aa)
	
	mean(t1_bs .>= t1), mean(t2_bs .>= t2)
end

# ╔═╡ a859dcbe-7d4c-4ec2-ad46-caec336d4e0e
let
	B = 10^5
	
	idx_ma = findfirst(data.data) do ld
		ld.line_params["cell"]=="MAGRU"
	end
	d_ma = data.data[idx_ma].data
	d_bs_ma = [sample(d_ma, length(d_ma), replace=true) for i in 1:B]
	
	idx_aa = findfirst(data.data) do ld
		ld.line_params["cell"]=="AAGRU"
	end
	d_aa = data.data[idx_aa].data
	d_bs_aa = [sample(d_aa, length(d_aa), replace=true) for i in 1:B]
	
	σ_pool = sqrt((var(d_aa) + var(d_ma))/2)
	abs(mean(d_aa) - mean(d_ma))/(σ_pool/sqrt(length(d_aa)))
end

# ╔═╡ a2157b3c-edd4-4f8f-9747-800492311655
bs1 = let
	B = 10^6
	idx_ma = findfirst(data.data) do ld
		ld.line_params["cell"]=="AAGRU"
	end
	d_ma = data.data[idx_ma].data
	bs1 = bootstrap(mean, d_ma, Bootstrap.BasicSampling(B))
end

# ╔═╡ 743ac888-6e87-46a9-b993-77cb73071d7d
bci1 = confint(bs1, BasicConfInt(0.99))

# ╔═╡ 095f3d78-d11c-406f-966e-704d44419170
let
	B = 10^6
	
	idx_ma = findfirst(data.data) do ld
		ld.line_params["cell"]=="MAGRU"
	end
	d_ma = data.data[idx_ma].data
	μ_ma = mean(d_ma)
	σ²_ma = var(d_ma)
	# d_bs_ma = [sample(d_ma, length(d_ma), replace=true) for i in 1:B]
	
	idx_aa = findfirst(data.data) do ld
		ld.line_params["cell"]=="AAGRU"
	end
	d_aa = data.data[idx_aa].data
	μ_aa = mean(d_aa)
	σ²_aa = var(d_aa)
	# d_bs_aa = [sample(d_aa, length(d_aa), replace=true) for i in 1:B]
	
	t_test(X, Y) = (mean(X) - mean(Y))/sqrt(var(X)/length(X) + var(Y)/length(Y))
	t = t_test(d_ma, d_aa)
	
	z̄ = mean([d_aa; d_ma])
	
	d_ma_star = d_ma .- (μ_ma - z̄)
	d_aa_star = d_aa .- (μ_aa - z̄)
	
	
	t_star = [begin; x = sample(d_ma_star, length(d_ma_star), replace=true); y=sample(d_aa_star, length(d_aa_star), replace=true); t_test(x, y); end for i in 1:B]
	
	mean(t_star .>= t)
	
# 	abs_median(X, Y) = abs(median(X) - median(Y))
# 	abs_mean(X, Y) = abs(mean(X) - mean(Y))
	
# 	t1 = abs_median(d_ma, d_aa)
# 	t2 = abs_mean(d_ma, d_aa)
	
# 	t1_bs = abs_median.(d_bs_ma, d_bs_aa)
# 	t2_bs = abs_mean.(d_bs_ma, d_bs_aa)
	
# 	mean(t1_bs .>= t1), mean(t2_bs .>= t2)
end

# ╔═╡ 35fac577-c50b-481f-be3c-10bc1f52ce06
let
	
	idx_ma = findfirst(data.data) do ld
		ld.line_params["cell"]=="MAGRU"
	end
	d_ma = data.data[idx_ma].data
	x̄_ma = mean(d_ma)
	s²_ma = var(d_ma)
	# d_bs_ma = [sample(d_ma, length(d_ma), replace=true) for i in 1:B]
	
	idx_aa = findfirst(data.data) do ld
		ld.line_params["cell"]=="AAGRU"
	end
	d_aa = data.data[idx_aa].data
	x̄_aa = mean(d_aa)
	s²_aa = var(d_aa)
	
	C = 0.95
	t_star = 7.14
	μpμ = (x̄_ma - x̄_aa) + t_star*sqrt(s²_ma/length(d_ma) + s²_aa/length(d_aa))
	μmμ = (x̄_ma - x̄_aa) - t_star*sqrt(s²_ma/length(d_ma) + s²_aa/length(d_aa))
	
	t = ((x̄_ma - x̄_aa) - μmμ)/sqrt(s²_ma/length(d_ma) + s²_aa/length(d_aa))
	μmμ, μpμ
	# sqrt(s²_ma/length(d_ma)), sqrt(s²_aa/length(d_aa))
end

# ╔═╡ 7def03b4-a460-4ea7-9b12-d4d456c78f17
let
	
	idx_ma = findfirst(data.data) do ld
		ld.line_params["cell"]=="MAGRU"
	end
	d_ma = data.data[idx_ma].data
	x̄_ma = mean(d_ma)
	s²_ma = var(d_ma)
	# d_bs_ma = [sample(d_ma, length(d_ma), replace=true) for i in 1:B]
	
	idx_aa = findfirst(data.data) do ld
		ld.line_params["cell"]=="AAGRU"
	end
	d_aa = data.data[idx_aa].data
	x̄_aa = mean(d_aa)
	s²_aa = var(d_aa)
	
	
	t = ((x̄_ma - x̄_aa))/sqrt(s²_ma/length(d_ma) + s²_aa/length(d_aa))
	# sqrt(s²_ma/length(d_ma)), sqrt(s²_aa/length(d_aa))
end

# ╔═╡ Cell order:
# ╠═e765b282-90ca-4c76-81d0-b78a75c150c5
# ╠═7d08af30-bc3c-11eb-1ad7-8fc5f7a1ec61
# ╠═7b444b3b-0f11-425c-aac6-877586515b41
# ╠═f8dcf929-a3a3-44e0-bc25-9782ed4fbfb9
# ╠═f1d06ce5-cd2b-4955-9a54-7f43e2bf7c0d
# ╠═5a4c004b-4db1-47ea-be17-06e4ba081c05
# ╠═ce31c440-0fe3-47a6-a035-fe1dac8f56d8
# ╠═f0c21ef8-0dbe-4ae3-8c51-ac1674d931fb
# ╠═c44265f4-3656-4fdc-98a5-57344b634e33
# ╠═c940c7be-3697-4a5f-bf75-4caeb284b90a
# ╠═51458b14-665e-487f-91f7-f55606a09f15
# ╠═5e49469b-2bff-411a-a8dc-226fde5eaf41
# ╠═0db6c0b2-3ce3-4f36-b84c-76031143f970
# ╠═6fc0bb4b-f202-4b29-8ee1-bb3d768069c9
# ╠═03841f34-7492-4f73-ac8a-596378c70af7
# ╠═75c2815a-191c-4b2e-96d1-b4b857f204ba
# ╠═7cff19e5-9348-4337-baba-0a3ef1508e64
# ╠═d63bce66-2f48-458e-85cc-1f993d4ac85e
# ╠═beeb9b36-8c04-4de6-8297-82ccb82b368f
# ╠═7556fc88-9eb9-4813-bee7-1cf4d032ec0d
# ╠═b176461f-be40-4693-830a-512965db3642
# ╠═b50f8b6a-60ee-4b16-8e7b-ec9b5eff5fd3
# ╠═dc89338c-ebd4-4f8c-941f-23717d842da7
# ╠═1d58d39d-04b1-4cf5-8a59-887a4b99120b
# ╠═a859dcbe-7d4c-4ec2-ad46-caec336d4e0e
# ╠═a2157b3c-edd4-4f8f-9747-800492311655
# ╠═743ac888-6e87-46a9-b993-77cb73071d7d
# ╠═095f3d78-d11c-406f-966e-704d44419170
# ╠═35fac577-c50b-481f-be3c-10bc1f52ce06
# ╠═7def03b4-a460-4ea7-9b12-d4d456c78f17
