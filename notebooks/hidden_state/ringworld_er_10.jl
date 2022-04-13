### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ dc1d7a9e-9f28-11ec-0dd3-6deead526d0e
let
	import Pkg
	Pkg.activate(joinpath(@__DIR__, "../"))
end

# ╔═╡ 65d6bd03-3850-4d95-acf2-daa63c8aa520
using Revise

# ╔═╡ ed2f0985-f5fe-4299-b935-b5c3b61fe2ef
begin
	using Plots
	plotly()
end

# ╔═╡ d2f25632-02b7-4656-8c0f-0abac73de1f1
using ActionRNNs, Random

# ╔═╡ 1d22a937-c5fa-4201-b44b-728fee5490a7
using RollingFunctions

# ╔═╡ 11448be4-c970-4206-81b5-e3b0d22b44a6
using Graphs, LightGraphs, SimpleWeightedGraphs, GraphPlot

# ╔═╡ 5963143b-49b6-4a54-b304-c55f982517e1
using TSne

# ╔═╡ bb0f868f-f596-4b35-b540-127936670101
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

# ╔═╡ e3403d18-09c7-43d1-a9c3-6763a0cd69d5
import ActionRNNs: MinimalRLCore

# ╔═╡ 1d0b9d2d-7b15-4dfe-a2a4-1741549d93c2
module RingWorldERExperimentWrapper 
	include("../../experiment/RingWorldERExperiment.jl") 
end

# ╔═╡ da5cba57-e7a4-4e25-a4e6-3d3ac95290ff


# ╔═╡ f1e77b5c-fa1f-4fc5-a8f7-05cc81f44ff5
# import RingWorldERExperiment
RingWorldERExperiment = RingWorldERExperimentWrapper.RingWorldERExperiment

# ╔═╡ d39ddd45-faf8-409d-bda7-aa5b11d33dba
import ActionRNNs: @data

# ╔═╡ b8fcad7b-0dda-461b-847d-befb34769023
let
	config = RingWorldERExperiment.default_config()
end

# ╔═╡ 2b510aa6-7c7c-489a-b5b5-8321276d70a4
function plot_learning_curves_rw(results)
	plt1 = plot(sum(abs.(results[:out_err]), dims=1)[1,:], title="AvgRMSE")
	plt2 = plot(
		rollmean(
			sum(abs.(results[:out_err]), dims=1)[1, :], 
			100)[1:100:end], 
		title="AvgRMSE_Win100")
	# plt3 = let
	# 	truth = results[:out_pred] - results[:out_err]
	# 	c = (t, p) -> if t == 1.0 
	# 		p >= 0.6 ? 0.0 : 1.0
	# 	else
	# 		p >= 0.6 ? 1.0 : 0.0
	# 	end
	# 	plot(rollmean(
	# 			sum(c.(truth, results[:out_pred]), dims=1)[1, :], 
	# 			100)[1:100:end], 
	# 		title="successes")
	# end
	# plot(plt1, plt2, plt3, legend=:none)
	plt1, plt2#, plt3
end

# ╔═╡ 2a0fcdf9-0f32-4e27-af25-e668820035ba
function ActionRNNs.ChoosyDataLoggers.process_data(::Val{:get_hs_1_layer}, data)
	data[collect(keys(data))[1]]
end

# ╔═╡ a75cf9f7-f7d0-42a9-bd16-657b87d3e8fe
function get_hs_over_time(
		exp::Function, 
		freeze_training::Bool; 
		rng=Random.GLOBAL_RNG)
	results, agent, env = exp()

	data, data_logger = RingWorldERExperiment.construct_logger(
		extra_groups_and_names = [
			(:Agent, :action),
			(:Agent, :hidden_state, :get_hs_1_layer),
			(:Env, :state)]
	)
	
	RingWorldERExperiment.with_logger(data_logger) do
		a = agent.action
		@data Agent action=a
		for i in 1:1000
			ret = MinimalRLCore.step!(env, a, rng)
			ret_a = MinimalRLCore.step!(agent, ret..., rng)
			if ret_a isa NamedTuple
				a = ret_a.action
			else
				a = ret_a
			end
		end
	end
	results, data
end

# ╔═╡ cb40f1f8-207f-4902-8795-740a9df79219
base_kwargs = (steps = 300000,
	opt = "RMSProp",
	rho = 0.9,
	size = 10,
	batch_size = 4,
	replay_size = 1000,
	warm_up = 1000,
	target_update_freq = 1000,
	update_freq = 4,
	outgamma = 0.9,
	outhorde = "gammas_term",)

# ╔═╡ 8d385e74-bdd9-4183-98f6-ddf37e154c05
res_m, hs, s, obs, a_tm1 = let
	test_ret = get_hs_over_time(false) do
		ret = RingWorldERExperiment.working_experiment(
			false;
			cell="MARNN",
			numhidden=12,
			truncation=10,
			eta=0.0037252902984619128,
			seed=21,
			base_kwargs...,
			log_extras=[["EXPExtra", "agent"], ["EXPExtra", "env"]])
		(ret.data, 
		 ret.data[:EXPExtra][:agent][1], 
		 ret.data[:EXPExtra][:env][1])
	end
	data = test_ret[2]
	hs = data[:Agent][:hidden_state]
	s = data[:Env][:state]
	obs = nothing
	a_tm1 = data[:Agent][:action]
	test_ret[1], hs, s, obs, a_tm1[1:end-1]
end

# ╔═╡ ce312280-1ca5-492e-b774-e372940b7d3a
# let
# 	rnge = 1:1000
# 	data = reduce(hcat, vcat.(hs[rnge], s[rnge]))
# 	plts = []
# 	for i in 1:size(data, 1)
# 		push!(plts, plot(data[i, :]))
# 	end
# 	plot(plts..., layout=(size(data, 1), 1))
# end

# ╔═╡ 8fe2f33c-0367-4ffe-9104-91f50bf160f2
plt_ma = let
	plts = []
	for i in 1:6, j in 1:2
		push!(plts, heatmap(reduce(hcat, hs[(s .== i) .&& (a_tm1 .== j)]), title="s:$(i), a:$(j)", clim=(-1.0, 1.0)))
	end
	plt_hm = plot(plts...)
	plt_lc = plot(plot_learning_curves_rw(res_m[:EXP])..., legend=:none)
	plt_lc, plt_hm
end

# ╔═╡ b2648a98-3861-4040-9651-e384576c8e65
res_a, hs_a, s_a, obs_a, a_tm1_a = let
	test_ret = get_hs_over_time(false) do
		ret = RingWorldERExperiment.working_experiment(
			false;
			cell="AARNN",
			numhidden=15,
			truncation=10,
			eta=0.0037252902984619128,
			seed=21,
			base_kwargs...,
			log_extras=[["EXPExtra", "agent"], ["EXPExtra", "env"]])
		(ret.save_results, 
		 ret.data[:EXPExtra][:agent][1], 
		 ret.data[:EXPExtra][:env][1])
	end
	data = test_ret[2]
	hs = data[:Agent][:hidden_state]
	s = data[:Env][:state]
	obs = nothing
	a_tm1 = data[:Agent][:action]
	test_ret[1], hs, s, obs, a_tm1[1:end-1]
end

# ╔═╡ fd83db31-4e44-4625-b6f6-82a2bca1ef6d
plt_aa = let
	plts = []
	for i in 1:6, j in 1:2
		push!(plts, heatmap(reduce(hcat, hs_a[(s_a .== i) .&& (a_tm1_a .== j)]), title="s:$(i), a:$(j)", clim=(-1.0, 1.0)))
	end
	plt_hm = plot(plts...)
	plt_lc = plot(plot_learning_curves_rw(res_a)..., legend=:none)
	plt_lc, plt_hm
end

# ╔═╡ 80b7441c-d029-41cb-985a-82e6fdc021d7
res_da, hs_da, s_da, obs_da, a_tm1_da = let
	test_ret = get_hs_over_time(false) do
		ret = RingWorldERExperiment.working_experiment(
			false,
			cell="AARNN", 
			deep=true,
			internal_a=6,
			numhidden=6,
			log_extras=[["EXPExtra", "agent"], ["EXPExtra", "env"]])
		(ret.save_results, 
		 ret.data[:EXPExtra][:agent][1], 
		 ret.data[:EXPExtra][:env][1])
	end
	data = test_ret[2]
	hs = data[:Agent][:hidden_state]
	s = data[:Env][:state]
	obs = nothing
	a_tm1 = data[:Agent][:action]
	test_ret[1], hs, s, obs, a_tm1[1:end-1]
end

# ╔═╡ a903b453-2726-47a0-9509-02d259dac016
plt_da = let
	plts = []
	for i in 1:6, j in 1:2
		push!(plts, heatmap(reduce(hcat, hs_da[(s_da .== i) .&& (a_tm1_da .== j)]), title="s:$(i), a:$(j)", clim=(-1.0, 1.0)))
	end
	plt_hm = plot(plts...)
	plt_lc = plot(plot_learning_curves_rw(res_da)..., legend=:none)
	plt_lc, plt_hm
end

# ╔═╡ 454002a1-e7d1-4963-81b6-27f8f4128abc
rcell = res_m[:EXPExtra][:agent][1].model[1]

# ╔═╡ 157c0e8f-bafd-4103-9baa-617a285b5507
rcell.cell.Wx[1, :, :]*[0] + rcell.cell.b[:, 1]

# ╔═╡ 9d390470-2c75-4139-b0e2-dc0e46dc5790
rcell.cell.Wh[1, :, :]

# ╔═╡ f7c226c6-a02f-46eb-b547-445ab646fc9d
import Plots: Colors

# ╔═╡ c5393359-8160-4433-afcd-33779af44ad8
let
	cell = res_m[:EXPExtra][:agent][1].model[1]
	state = copy(rcell.state)
end

# ╔═╡ 546a4bcc-dad1-46f4-9192-e46b49dbf74b
function create_swgraph(weights_h::Matrix)
	# assume rows are edges and columns are vertices?
	g = SimpleWeightedDiGraph(12)
	for i in 1:6, j in 1:6 
		Graphs.add_edge!(g, j, i+6, weights_h[i, j])
	end
	g
end

# ╔═╡ ccdfb017-7f44-4ac0-85dd-ca59cfcd9d02
create_swgraph(rcell.cell.Wh[1, :, :]).weights

# ╔═╡ 5c1a042c-33a9-49de-8aeb-e8d213f1ea07
let
	# layout=(args...)->spring_layout(args...; C=20)
	# nodefillc = distinguishable_colors(12, colorant"blue")
	nodefillc = distinguishable_colors(6, colorant"blue")
	gplot(create_swgraph(rcell.cell.Wh[1, :, :]), nodelabel=[1:6; 1:6], nodefillc=[nodefillc; nodefillc])
end

# ╔═╡ b881e9b2-f7a5-40b5-b9df-aa45cd85e0f2
gr()

# ╔═╡ b2f1774f-7fc1-487a-82f3-c7f28ca01cdd
let
	
	Random.seed!(3)
	base_colors = distinguishable_colors(12, colorant"blue")
	colors = fill(base_colors[1], length(hs))
	for i in 1:10, j in 1:2
		colors[(s .== i) .&& (a_tm1 .== j)] .= base_colors[i]
 	end
	data = tsne(collect(reduce(hcat, hs)'), 2, 0, 1000, progress=false)
	plt = scatter(data[:, 1], data[:, 2], color=colors, legend=nothing, title="Multiplicative")
	# savefig(plt, "tsne-ringworld-mult.pdf")
	plt
end

# ╔═╡ 91e3c04d-986b-42c9-8ed8-274a35813ec1
let
	Random.seed!(3)
	base_colors = distinguishable_colors(12, colorant"blue")
	colors = fill(base_colors[1], length(hs_a))
	for i in 1:12, j in 1:2
		colors[(s_a .== i) .&& (a_tm1_a .== j)] .= base_colors[i]
	end
	data = tsne(collect(reduce(hcat, hs_a)'), 2, 0, 1000, progress=false)
	plt = scatter(data[:, 1], data[:, 2], color=colors, legend=nothing, title="Additive")
	# savefig(plt, "tsne-ringworld-add.pdf")
	plt
end

# ╔═╡ d346da3f-30b4-4108-b056-1f1ec85caf49
let
	Random.seed!(3)
	base_colors = distinguishable_colors(12, colorant"blue")
	colors = fill(base_colors[1], length(hs_a))
	for i in 1:6, j in 1:2
		colors[(s_da .== i) .&& (a_tm1_da .== j)] .= base_colors[i]
	end
	data = tsne(collect(reduce(hcat, hs_da)'), 2, 0, 1000, progress=false)
	plt = scatter(data[:, 1], data[:, 2], color=colors, legend=nothing, title="DeepAdditive")
	# savefig(plt, "tsne-ringworld-deepadd.pdf")
	plt
end

# ╔═╡ Cell order:
# ╠═dc1d7a9e-9f28-11ec-0dd3-6deead526d0e
# ╠═bb0f868f-f596-4b35-b540-127936670101
# ╠═65d6bd03-3850-4d95-acf2-daa63c8aa520
# ╠═ed2f0985-f5fe-4299-b935-b5c3b61fe2ef
# ╠═e3403d18-09c7-43d1-a9c3-6763a0cd69d5
# ╠═1d0b9d2d-7b15-4dfe-a2a4-1741549d93c2
# ╠═da5cba57-e7a4-4e25-a4e6-3d3ac95290ff
# ╠═f1e77b5c-fa1f-4fc5-a8f7-05cc81f44ff5
# ╠═d39ddd45-faf8-409d-bda7-aa5b11d33dba
# ╠═d2f25632-02b7-4656-8c0f-0abac73de1f1
# ╠═1d22a937-c5fa-4201-b44b-728fee5490a7
# ╠═b8fcad7b-0dda-461b-847d-befb34769023
# ╠═2b510aa6-7c7c-489a-b5b5-8321276d70a4
# ╠═2a0fcdf9-0f32-4e27-af25-e668820035ba
# ╠═a75cf9f7-f7d0-42a9-bd16-657b87d3e8fe
# ╠═cb40f1f8-207f-4902-8795-740a9df79219
# ╠═8d385e74-bdd9-4183-98f6-ddf37e154c05
# ╠═ce312280-1ca5-492e-b774-e372940b7d3a
# ╠═8fe2f33c-0367-4ffe-9104-91f50bf160f2
# ╠═b2648a98-3861-4040-9651-e384576c8e65
# ╠═fd83db31-4e44-4625-b6f6-82a2bca1ef6d
# ╠═80b7441c-d029-41cb-985a-82e6fdc021d7
# ╠═a903b453-2726-47a0-9509-02d259dac016
# ╠═454002a1-e7d1-4963-81b6-27f8f4128abc
# ╠═157c0e8f-bafd-4103-9baa-617a285b5507
# ╠═9d390470-2c75-4139-b0e2-dc0e46dc5790
# ╠═11448be4-c970-4206-81b5-e3b0d22b44a6
# ╠═f7c226c6-a02f-46eb-b547-445ab646fc9d
# ╠═c5393359-8160-4433-afcd-33779af44ad8
# ╠═546a4bcc-dad1-46f4-9192-e46b49dbf74b
# ╠═ccdfb017-7f44-4ac0-85dd-ca59cfcd9d02
# ╠═5c1a042c-33a9-49de-8aeb-e8d213f1ea07
# ╠═5963143b-49b6-4a54-b304-c55f982517e1
# ╠═b881e9b2-f7a5-40b5-b9df-aa45cd85e0f2
# ╠═b2f1774f-7fc1-487a-82f3-c7f28ca01cdd
# ╠═91e3c04d-986b-42c9-8ed8-274a35813ec1
# ╠═d346da3f-30b4-4108-b056-1f1ec85caf49
