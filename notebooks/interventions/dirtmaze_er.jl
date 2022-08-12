### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ dc24843b-568a-4e2b-a998-994e025486b9
let
	import Pkg
	Pkg.activate(joinpath(@__DIR__, "../"))
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
module DirectionalTMazeERExperimentWrapper
	include("../../experiment/DirectionalTMazeERExperiment.jl")
end
# import DirectionalTMazeERExperiment

# ╔═╡ 1aa6ea1f-a6f8-4b93-b995-dd6b6646f2cf
DirectionalTMazeERExperiment = 
	DirectionalTMazeERExperimentWrapper.DirectionalTMazeERExperiment

# ╔═╡ dcdd9654-0105-464e-88c3-0e6dc7c44b65
import ActionRNNs: @data, ChoosyDataLoggers

# ╔═╡ d0dce888-6254-4477-af25-057be27351e8
function ChoosyDataLoggers.process_data(::Val{:get_hs_1_layer}, data)
	data[collect(keys(data))[1]]
end

# ╔═╡ a29f06c2-6914-48ca-bde6-6a75beadc992
function run_experiment(
		exp::Function, 
		freeze_training::Bool; 
		rng=Random.GLOBAL_RNG)
	
	results, agent, env = exp()

	results, agent, env
end

# ╔═╡ 8132f595-ce46-4a78-816e-3a6b279b3a06
function intervention_experiment(agent, env, intervention, start_dir, rng)

	s_str = Bool[]
	max_episode_steps = 5_000
	@progress for eps in 1:1_000
		success = false
		intervention_episode!(
			env,
			agent,
			intervention,
			start_dir isa Int ? start_dir : start_dir(rng),
			max_episode_steps,
			rng) do (s, a, s′, r)
			
			success = success || (r == 4.0)
		end
		# @data "EXP" total_rews=rews
		# @data "EXP" total_steps=steps
		j = 1
		push!(s_str, success)
	end
	@data "EXP" successes=sum(s_str)
end

# ╔═╡ 9154942d-8523-45ac-8c5b-fb8dac8c5ece
begin
	struct NoIntervention end
	reset!(::NoIntervention) = nothing
	make_intervention(agent, env, ::NoIntervention, agent_ret) = 
		if agent_ret isa Int
			agent_ret
		else
			agent_ret.action
		end
end

# ╔═╡ f99084ba-21ec-4677-9e18-0963bebf6cf0
function run_intervention_experiment(
		exp::Function;
		freeze_training::Bool=true,
		rng=Random.GLOBAL_RNG,
		inter_start_dir_list=[(NoIntervention(),rand(rng, 1:4))]
)
	
	results, agent, env = exp()
	if freeze_training
		ActionRNNs.turn_off_training!(agent)
	end
	
	ret = []
	for (inter, start_dir) in inter_start_dir_list
		cp_agent, cp_env = deepcopy(agent), deepcopy(env)
		data, data_logger = DirectionalTMazeERExperiment.construct_logger()
		with_logger(data_logger) do
			intervention_experiment(agent, env, inter, start_dir, rng)
		end
		push!(ret, (inter, start_dir) => data)
	end

	ret
end

# ╔═╡ e78776da-715c-458f-8db2-4212a82418d5
mutable struct ActionInterventionUnderCondition
	action::Int
	condition::Function
	intervened::Bool
end

# ╔═╡ d6c234ea-823d-498b-8c14-a6e60ae374f6
function reset!(intervention::ActionInterventionUnderCondition)
	intervention.intervened = false
end

# ╔═╡ bad54c0a-2c48-4464-9d6a-a54427719f10
function make_intervention(
	agent::ActionRNNs.DRQNAgent,
	env::ActionRNNs.DirectionalTMaze,
	intervention::ActionInterventionUnderCondition, 
	agent_ret)

	if !intervention.intervened && intervention.condition(env)
		agent.action = intervention.action
		intervention.intervened = true
	end

	agent.action
end

# ╔═╡ 0cc00a08-a4a5-4e94-b281-de25630c167a
function intervention_episode!(
	f, 
	env, 
	agent, 
	intervention, 
	start_dir,
	max_episode_steps, 
	rng)
	terminal = false

	s = MinimalRLCore.start!(env, rand(rng, [1,3]), start_dir)
	ar = MinimalRLCore.start!(agent, s, rng)
	reset!(intervention)

	na = make_intervention(agent, env, intervention, ar)
	step = 0
	while !terminal && step < max_episode_steps

		s′, r, terminal = MinimalRLCore.step!(env, na, rng)
		ar = MinimalRLCore.step!(agent, s′, r, terminal, rng)

		# na = _get_action(ar)
		na = make_intervention(agent, env, intervention, ar)
		
		f((s, na, s′, r, terminal))

		step+=1 
		# @info step
	end
end

# ╔═╡ 14657dfb-a87b-416a-b15a-fc0bdb798293
DirectionalTMazeERExperiment.default_config()

# ╔═╡ 3f276e08-fd7b-4122-b1ff-968370ff2fca
base_args = Dict(
	:steps => 300000,
	:opt => "RMSProp",
	:rho => 0.99,

	:size => 10,
	:gamma => 0.99,
	:batch_size => 8,
	:replay_size => 10000,
	:warm_up => 1000,
	:hs_learnable => true,
	:update_wait => 4,
	:target_update_wait => 1000,
)

# ╔═╡ 54d595fd-de75-4f4e-ac37-9d5784615307
test_ret = run_experiment(true, rng=Random.MersenneTwister(2)) do
	ret = DirectionalTMazeERExperiment.working_experiment(
		false;
		cell="MAGRU",
		numhidden=10,
		truncation=12,
		eta = 0.0003125,
		base_args...,
		seed=3,
		log_extras=[["EXPExtra", "agent"], ["EXPExtra", "env"]])
	(ret.save_results, 
	 ret.data[:EXPExtra][:agent][1],
	 ret.data[:EXPExtra][:env][1])
end

# ╔═╡ 2258a53e-a0d5-4408-9763-259c9a0fc5c8
plot(rollmean(test_ret[1][:successes], 100))

# ╔═╡ a721c380-ba5d-4430-adfb-6dd9c291e2d5
let
	rng = Random.MersenneTwister(4)
	intervention = ActionInterventionUnderCondition(
		2, (env)->env.state.x == 1, false
	)
	env, agent = test_ret[3], test_ret[2]
	ActionRNNs.turn_off_training!(agent)

	successes = Bool[]
	@progress for i in 1:1_000
		success = false
		intervention_episode!(env, agent, intervention, 2, 5_000, rng) do (s, a, s′, r, t) 
			success = success || r == 4.0
		end
		push!(successes, success)
	end
	sum(successes)
end

# ╔═╡ ac2ee0dc-dd6a-42d4-a61c-dc625461061f
let
	intervention = NoIntervention()
	env, agent = test_ret[3], test_ret[2]
	ActionRNNs.turn_off_training!(agent)
	rng = Random.MersenneTwister(2)
	successes = Bool[]
	@progress for i in 1:1_000
		success = false
		intervention_episode!(env, agent, intervention, 2, 100_000, rng) do (s, a, s′, r, t) 
			success = success || r == 4.0
		end
		push!(successes, success)
	end
	sum(successes)
end

# ╔═╡ f81c9ceb-c6cc-4448-8dcb-f4b79aeb47b6
function inter_exp(seed, cell_args, base_args; freeze_training=true)
	rng=Random.MersenneTwister(seed)
	inter_start_dir_list = [
		(NoIntervention(), 2),
		(ActionInterventionUnderCondition(
			2, (env)->env.state.x == 1, false), 2),
		(NoIntervention(), (_rng)->rand(rng, 1:4)),
		(ActionInterventionUnderCondition(
			1, (env)->env.state.x == 5, false), (_rng)->rand(rng, 1:4))
	]
	ret = run_intervention_experiment(
			freeze_training=freeze_training, 
			rng = rng, inter_start_dir_list=inter_start_dir_list,
			) do
		ret = DirectionalTMazeERExperiment.working_experiment(
			false;
			cell_args...,
			# cell="MAGRU",
			# numhidden=10,
			# truncation=12,
			# eta = 0.0003125,
			base_args...,
			seed=seed,
			log_extras=[["EXPExtra", "agent"], ["EXPExtra", "env"]])
		(ret.save_results, 
		 ret.data[:EXPExtra][:agent][1],
		 ret.data[:EXPExtra][:env][1])
	end

	[sum(ret[i].second[:EXP][:successes]) for i in 1:length(inter_start_dir_list)]
	
end

# ╔═╡ ab4c5a80-0400-49e4-82e2-fc81051983b8
inter_exp(
	3, 
	Dict(:cell=>"MAGRU", :numhidden=>10, :truncation=>12, :eta => 0.0003125), base_args)

# ╔═╡ Cell order:
# ╠═dc24843b-568a-4e2b-a998-994e025486b9
# ╠═3782e601-b4c2-4fcf-bae0-9df4d7d5dad6
# ╠═edfc15af-382d-480a-9e4d-eb0a88e4481e
# ╠═409c3076-40aa-4d02-8924-38087d8f08da
# ╠═cbd23d8d-a9ee-4f6f-b7f0-f86f67ffd054
# ╠═1aa6ea1f-a6f8-4b93-b995-dd6b6646f2cf
# ╠═dcdd9654-0105-464e-88c3-0e6dc7c44b65
# ╠═9116f864-cfba-4f34-8dab-f53f290a523e
# ╠═90161e4a-79e3-4c24-a138-bbfe2f5e818b
# ╠═bfb9c7a8-15e5-490c-b639-401d1d2b26eb
# ╠═38f400f5-c389-4c73-ba16-640d34b0883a
# ╠═d0dce888-6254-4477-af25-057be27351e8
# ╠═a29f06c2-6914-48ca-bde6-6a75beadc992
# ╠═f99084ba-21ec-4677-9e18-0963bebf6cf0
# ╠═8132f595-ce46-4a78-816e-3a6b279b3a06
# ╠═9154942d-8523-45ac-8c5b-fb8dac8c5ece
# ╠═e78776da-715c-458f-8db2-4212a82418d5
# ╠═d6c234ea-823d-498b-8c14-a6e60ae374f6
# ╠═bad54c0a-2c48-4464-9d6a-a54427719f10
# ╠═0cc00a08-a4a5-4e94-b281-de25630c167a
# ╠═14657dfb-a87b-416a-b15a-fc0bdb798293
# ╠═3f276e08-fd7b-4122-b1ff-968370ff2fca
# ╠═54d595fd-de75-4f4e-ac37-9d5784615307
# ╠═2258a53e-a0d5-4408-9763-259c9a0fc5c8
# ╠═a721c380-ba5d-4430-adfb-6dd9c291e2d5
# ╠═ac2ee0dc-dd6a-42d4-a61c-dc625461061f
# ╠═f81c9ceb-c6cc-4448-8dcb-f4b79aeb47b6
# ╠═ab4c5a80-0400-49e4-82e2-fc81051983b8
