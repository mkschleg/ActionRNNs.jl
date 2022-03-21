### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ dc24843b-568a-4e2b-a998-994e025486b9
let
	import Pkg
	Pkg.activate(joinpath(@__DIR__, "../../"))
end

# ╔═╡ 3782e601-b4c2-4fcf-bae0-9df4d7d5dad6
using Revise

# ╔═╡ edfc15af-382d-480a-9e4d-eb0a88e4481e
begin
	using Plots
	plotly()
end

# ╔═╡ 9116f864-cfba-4f34-8dab-f53f290a523e
using MinimalRLCore, ActionRNNs, Random

# ╔═╡ 90161e4a-79e3-4c24-a138-bbfe2f5e818b
using RollingFunctions

# ╔═╡ bfb9c7a8-15e5-490c-b639-401d1d2b26eb
using Logging: with_logger

# ╔═╡ 38f400f5-c389-4c73-ba16-640d34b0883a
using ProgressLogging

# ╔═╡ cbd23d8d-a9ee-4f6f-b7f0-f86f67ffd054
import DirectionalTMazeERExperiment

# ╔═╡ dcdd9654-0105-464e-88c3-0e6dc7c44b65
import ActionRNNs: @data

# ╔═╡ d0dce888-6254-4477-af25-057be27351e8
function ActionRNNs.ExpUtils.process_data(::Val{:get_hs_1_layer}, data)
	data[collect(keys(data))[1]]
end

# ╔═╡ a29f06c2-6914-48ca-bde6-6a75beadc992
function get_hs_over_time(
		exp::Function, 
		freeze_training::Bool; 
		rng=Random.GLOBAL_RNG)
	results, agent, env = exp()

	data, data_logger = DirectionalTMazeERExperiment.construct_logger(
		extra_groups_and_names = [
			(:Agent, :action),
			(:Agent, :action_tm1),
			(:Agent, :hidden_state, :get_hs_1_layer),
			(:Agent, :hidden_state_init, :get_hs_1_layer),
			(:Env)]
	)
	if freeze_training
		ActionRNNs.turn_off_training(agent)
	end
	
	with_logger(data_logger) do
		max_episode_steps = 5_000
		@progress "episode: " for eps in 1:1_000
			success = false
			rews, steps = run_episode!( 
				env, 
				agent, 
				max_episode_steps, 
				rng) do (s, a, s′, r)
				success = success || (r == 4.0)
			end
			@data "EXP" total_rews=rews
			@data "EXP" total_steps=steps
			@data "EXP" successes=success
		end
	end
	results, data
end

# ╔═╡ 54d595fd-de75-4f4e-ac37-9d5784615307
test_ret = get_hs_over_time(false, rng=Random.MersenneTwister(2)) do
	ret = DirectionalTMazeERExperiment.working_experiment(
		cell="MARNN",
		numhidden=10,
		log_extras=[["EXPExtra", "agent"], ["EXPExtra", "env"]])
	(ret.save_results, 
	 ret.data[:EXPExtra][:agent][1],
	 ret.data[:EXPExtra][:env][1])
end

# ╔═╡ 03785ab7-c3c2-45de-843b-ae1d612cef27
length(test_ret[2][:Env][:state])

# ╔═╡ ef4fb61f-8aae-413f-8606-237b3851743d
length(test_ret[2][:Agent][:action_tm1])

# ╔═╡ 6e14cb0b-9ded-4da1-8250-5a8c784bb3d8
let
	states = test_ret[2][:Env][:state]
	unique_states = unique(states)
	plts = []
	for s in unique_states
		ids = findall((_s)->s==_s, states)
		# split futher by action
		
		
		
		hm = heatmap(reduce(hcat, test_ret[2][:Agent][:hidden_state][ids]), clims=(-1.0, 1.0))
		push!(plts, hm)
	end
	plts
end

# ╔═╡ Cell order:
# ╠═dc24843b-568a-4e2b-a998-994e025486b9
# ╠═3782e601-b4c2-4fcf-bae0-9df4d7d5dad6
# ╠═edfc15af-382d-480a-9e4d-eb0a88e4481e
# ╠═cbd23d8d-a9ee-4f6f-b7f0-f86f67ffd054
# ╠═dcdd9654-0105-464e-88c3-0e6dc7c44b65
# ╠═9116f864-cfba-4f34-8dab-f53f290a523e
# ╠═90161e4a-79e3-4c24-a138-bbfe2f5e818b
# ╠═bfb9c7a8-15e5-490c-b639-401d1d2b26eb
# ╠═38f400f5-c389-4c73-ba16-640d34b0883a
# ╠═d0dce888-6254-4477-af25-057be27351e8
# ╠═a29f06c2-6914-48ca-bde6-6a75beadc992
# ╠═54d595fd-de75-4f4e-ac37-9d5784615307
# ╠═03785ab7-c3c2-45de-843b-ae1d612cef27
# ╠═ef4fb61f-8aae-413f-8606-237b3851743d
# ╠═6e14cb0b-9ded-4da1-8250-5a8c784bb3d8
