### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ dc1d7a9e-9f28-11ec-0dd3-6deead526d0e
let
	import Pkg
	Pkg.activate(joinpath(@__DIR__, "../../"))
end

# ╔═╡ 65d6bd03-3850-4d95-acf2-daa63c8aa520
using Revise

# ╔═╡ ed2f0985-f5fe-4299-b935-b5c3b61fe2ef
begin
	using Plots
	plotly()
end

# ╔═╡ d2f25632-02b7-4656-8c0f-0abac73de1f1
using MinimalRLCore, ActionRNNs, Random

# ╔═╡ 1d22a937-c5fa-4201-b44b-728fee5490a7
using RollingFunctions

# ╔═╡ 7306a11a-aade-4fa3-89e2-32f6c80fd343
using Logging, Pluto

# ╔═╡ f1e77b5c-fa1f-4fc5-a8f7-05cc81f44ff5
import RingWorldERExperiment

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
	plt3 = let
		truth = results[:out_pred] - results[:out_err]
		c = (t, p) -> if t == 1.0 
			p >= 0.6 ? 0.0 : 1.0
		else
			p >= 0.6 ? 1.0 : 0.0
		end
		plot(rollmean(
				sum(c.(truth, results[:out_pred]), dims=1)[1, :], 
				100)[1:100:end], 
			title="successes")
	end
	# plot(plt1, plt2, plt3, legend=:none)
	plt1, plt2, plt3
end

# ╔═╡ 2a0fcdf9-0f32-4e27-af25-e668820035ba
function ActionRNNs.ExpUtils.process_data(::Val{:get_hs_1_layer}, data)
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

# ╔═╡ face2685-2609-44e1-9427-b82273f2d19b
test_ret = get_hs_over_time(false) do
	ret = RingWorldERExperiment.working_experiment(
		cell="AARNN", 
		numhidden=10,
		log_extras=[["EXPExtra", "agent"], ["EXPExtra", "env"]])
	(ret.save_results, 
	 ret.data[:EXPExtra][:agent][1], 
	 ret.data[:EXPExtra][:env][1])
end

# ╔═╡ 8d385e74-bdd9-4183-98f6-ddf37e154c05
res_m, hs, s, obs, a_tm1 = let
	test_ret = get_hs_over_time(false) do
		ret = RingWorldERExperiment.working_experiment(
			cell="MARNN", 
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

# ╔═╡ ce312280-1ca5-492e-b774-e372940b7d3a
let
	rnge = 1:1000
	data = reduce(hcat, vcat.(hs[rnge], s[rnge]))
	plts = []
	for i in 1:size(data, 1)
		push!(plts, plot(data[i, :]))
	end
	plot(plts..., layout=(size(data, 1), 1))
end

# ╔═╡ 8fe2f33c-0367-4ffe-9104-91f50bf160f2
plt_ma = let
	plts = []
	for i in 1:6, j in 1:2
		push!(plts, heatmap(reduce(hcat, hs[(s .== i) .&& (a_tm1 .== j)]), title="s:$(i), a:$(j)", clim=(-1.0, 1.0)))
	end
	plt_hm = plot(plts...)
	plt_lc = plot(plot_learning_curves_rw(res_m)..., legend=:none)
	plt_lc, plt_hm
end

# ╔═╡ b2648a98-3861-4040-9651-e384576c8e65
res_a, hs_a, s_a, obs_a, a_tm1_a = let
	test_ret = get_hs_over_time(false) do
		ret = RingWorldERExperiment.working_experiment(
			cell="AARNN", 
			numhidden=10,
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

# ╔═╡ 5d411864-74b3-445a-a07a-c24763d6e352
Logging.current_logger()

# ╔═╡ 454002a1-e7d1-4963-81b6-27f8f4128abc


# ╔═╡ Cell order:
# ╠═dc1d7a9e-9f28-11ec-0dd3-6deead526d0e
# ╠═65d6bd03-3850-4d95-acf2-daa63c8aa520
# ╠═ed2f0985-f5fe-4299-b935-b5c3b61fe2ef
# ╠═f1e77b5c-fa1f-4fc5-a8f7-05cc81f44ff5
# ╠═d39ddd45-faf8-409d-bda7-aa5b11d33dba
# ╠═d2f25632-02b7-4656-8c0f-0abac73de1f1
# ╠═1d22a937-c5fa-4201-b44b-728fee5490a7
# ╠═b8fcad7b-0dda-461b-847d-befb34769023
# ╠═2b510aa6-7c7c-489a-b5b5-8321276d70a4
# ╠═2a0fcdf9-0f32-4e27-af25-e668820035ba
# ╠═a75cf9f7-f7d0-42a9-bd16-657b87d3e8fe
# ╠═face2685-2609-44e1-9427-b82273f2d19b
# ╠═8d385e74-bdd9-4183-98f6-ddf37e154c05
# ╠═ce312280-1ca5-492e-b774-e372940b7d3a
# ╠═8fe2f33c-0367-4ffe-9104-91f50bf160f2
# ╠═b2648a98-3861-4040-9651-e384576c8e65
# ╠═fd83db31-4e44-4625-b6f6-82a2bca1ef6d
# ╠═80b7441c-d029-41cb-985a-82e6fdc021d7
# ╠═a903b453-2726-47a0-9509-02d259dac016
# ╠═7306a11a-aade-4fa3-89e2-32f6c80fd343
# ╠═5d411864-74b3-445a-a07a-c24763d6e352
# ╠═454002a1-e7d1-4963-81b6-27f8f4128abc
