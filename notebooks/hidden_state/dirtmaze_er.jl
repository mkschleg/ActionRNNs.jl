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

# ╔═╡ fb97ad51-4e1f-4a43-8b7f-829ad37974e8
using TSne, Statistics, MLDatasets

# ╔═╡ 409c3076-40aa-4d02-8924-38087d8f08da
color_scheme = [
    colorant"#44AA99",
    colorant"#332288",
    colorant"#DDCC77",
    colorant"#999933",
    colorant"#CC6677",
    colorant"#AA4499",
    colorant"#117733",
    colorant"#882255",
    colorant"#1E90FF",
]

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
		size=6,
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

# ╔═╡ f0008d1f-fce0-4575-a831-be71b6a53140
Y, h_idx = let
	h_idx = rand(1:20642, 1000)
	hs = reduce(vcat, [(h' .+ 1) ./ 2 for h in test_ret[2][:Agent][:hidden_state][h_idx]])
	
	# X = rescale(hs, dims=1);
	Y = tsne(hs, 3, 0, 1000, 50.0)
	# scatter3d(Y[:, 1], Y[:,2], Y[:, 3])
	Y, h_idx
end

# ╔═╡ e3acbf85-4ae5-4059-a564-3baba6f69e9e
test_ret[2][:Env][:state]

# ╔═╡ 6be515fa-bb8d-400d-a423-cb72b5d04839
let
	get_color = (idx) -> begin
		state = test_ret[2][:Env][:state][idx]
		num_eps = sum(test_ret[2][:Env][:reset][1:idx])
		gd = test_ret[2][:Env][:goal_dir][num_eps]
		# color_scheme[gd]
		if state.x == 1
			if gd == 1
				color_scheme[1]
			else
				color_scheme[9]		
			end
		elseif state.x != 6
			color_scheme[3]
		elseif state.x == 6
			if state.dir==1
				color_scheme[7]
			elseif state.dir==3
				color_scheme[5]
			else
				color_scheme[2]
			end
		end
		
	end
	scatter3d(Y[:, 1], Y[:,2], Y[:, 3], markersize=1, color=get_color.(h_idx))
	# scatter(Y[:, 1], Y[:,2], markersize=2, color=get_color.(h_idx))
	
end

# ╔═╡ Cell order:
# ╠═dc24843b-568a-4e2b-a998-994e025486b9
# ╠═3782e601-b4c2-4fcf-bae0-9df4d7d5dad6
# ╠═edfc15af-382d-480a-9e4d-eb0a88e4481e
# ╠═409c3076-40aa-4d02-8924-38087d8f08da
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
# ╠═fb97ad51-4e1f-4a43-8b7f-829ad37974e8
# ╠═f0008d1f-fce0-4575-a831-be71b6a53140
# ╠═e3acbf85-4ae5-4059-a564-3baba6f69e9e
# ╠═6be515fa-bb8d-400d-a423-cb72b5d04839
