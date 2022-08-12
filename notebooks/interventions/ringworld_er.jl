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

# ╔═╡ 04f9a921-7648-4c81-855c-11dfc25fa26c
using Statistics

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
module RingWorldERExperiment_FixRNGWrapper
	include("../../experiment/RingWorldERExperiment_FixRNG.jl")
end
# import DirectionalTMazeERExperiment

# ╔═╡ 1aa6ea1f-a6f8-4b93-b995-dd6b6646f2cf
RingWorldERExperiment_FixRNG = 
	RingWorldERExperiment_FixRNGWrapper.RingWorldERExperiment_FixRNG

# ╔═╡ dcdd9654-0105-464e-88c3-0e6dc7c44b65
import ActionRNNs: @data, ChoosyDataLoggers

# ╔═╡ 4d2261fc-f4e9-4d88-99c0-e25219480f0e
const RWU = ActionRNNs.ExpUtils.RingWorldUtils

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

# ╔═╡ 9154942d-8523-45ac-8c5b-fb8dac8c5ece
begin
	struct NoIntervention end
	reset!(::NoIntervention) = nothing
	make_intervention(agent, env, ::NoIntervention, agent_ret) = 
		if agent_ret isa Int
			agent_ret
		else
			agent.action
		end
end

# ╔═╡ e6e41135-a0e7-476f-a51d-b98787f7d699
mutable struct ActionInterventionPersistent
	action::Int
end

# ╔═╡ 3f624303-938d-4eca-badc-c4af068167f9
reset!(intervention::ActionInterventionPersistent) = nothing

# ╔═╡ ffb26b02-a0d8-4731-8962-9fa2587abf88
function make_intervention(
	agent::ActionRNNs.AbstractERAgent,
	env::ActionRNNs.RingWorld,
	intervention::ActionInterventionPersistent, 
	agent_ret)
	
	agent.action = intervention.action
	intervention.action
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
	env::ActionRNNs.RingWorld,
	intervention::ActionInterventionUnderCondition, 
	agent_ret)

	if !intervention.intervened && intervention.condition(env)
		agent.action = intervention.action
		intervention.intervened = true
	end

	agent.action
end

# ╔═╡ f5533e88-c50e-4082-b746-debf0bafe90a
function intervention_continuing!(
	f, 
	env, 
	agent,
	intervention, 
	numsteps, 
	rng)
	
	na = make_intervention(agent, env, intervention, nothing)
	s = MinimalRLCore.get_state(env)
	step = 0
	while step < numsteps

		s′, r, terminal = MinimalRLCore.step!(env, na, rng)
		ar = MinimalRLCore.step!(agent, s′, r, terminal, rng)

		na = make_intervention(agent, env, intervention, ar)
		
		f((s, na, ar, s′, r, terminal))

		step+=1 
		# @info step
	end
end

# ╔═╡ 8132f595-ce46-4a78-816e-3a6b279b3a06
function intervention_experiment(agent, env::ActionRNNs.RingWorld, intervention, rng)

	steps = 50_000

	intervention_continuing!(
		env,
		agent,
		intervention,
		steps,
		rng) do (s, na, ar, s′, r, terminal)


		out_preds = ar.preds[:, 1]
		
		@data EXP out_pred=out_preds
		out_errs = out_preds .- Float32.(RWU.oracle(env, "gammas_term"))
		@data EXP out_err=out_errs
	end
	
end

# ╔═╡ f99084ba-21ec-4677-9e18-0963bebf6cf0
function run_intervention_experiment(
		exp::Function;
		freeze_training::Bool=true,
		rng=Random.GLOBAL_RNG,
		intervention=NoIntervention())
	
	results, agent, env = exp()
	if freeze_training
		ActionRNNs.turn_off_training!(agent)
	end
	
	data, data_logger = RingWorldERExperiment_FixRNG.construct_logger()
	with_logger(data_logger) do
		intervention_experiment(agent, env, intervention, rng)
	end

	data
end

# ╔═╡ 14657dfb-a87b-416a-b15a-fc0bdb798293
RingWorldERExperiment_FixRNG.default_config()

# ╔═╡ 3f276e08-fd7b-4122-b1ff-968370ff2fca
base_args = Dict(
	# :steps => 300000,
	# :opt => "RMSProp",
	# :rho => 0.99,

	# :size => 10,
	# :gamma => 0.99,
	# :batch_size => 8,
	# :replay_size => 10000,
	# :warm_up => 1000,
	# :hs_learnable => true,
	# :update_wait => 4,
	# :target_update_wait => 1000,
	:steps => 300000,
	:opt => "RMSProp",
	:rho => 0.9,
	:size => 10,
	:batch_size => 4,
	:replay_size => 1000,
	:warm_up => 1000,
	:target_update_freq => 1000,

	:outgamma => 0.9,
	:synopsis => true,

	:outhorde => "gammas_term",
	:hs_learnable => true,
	:update_freq => 4

)

# ╔═╡ f81c9ceb-c6cc-4448-8dcb-f4b79aeb47b6
function inter_exp(seed, cell_args, base_args, intervention; freeze_training=true)
	
	rng=Random.MersenneTwister(seed)
	# intervention = NoIntervention()
	
	ret = run_intervention_experiment(
		freeze_training=freeze_training, 
		rng = rng, 
		intervention=intervention) do
		
			ret = RingWorldERExperiment_FixRNG.working_experiment(
				false;
				cell_args...,
				base_args...,
				seed=seed,
				log_extras=[["EXPExtra", "agent"], ["EXPExtra", "env"]])
			(ret.save_results, 
			 ret.data[:EXPExtra][:agent][1],
			 ret.data[:EXPExtra][:env][1])
	end

	# [sum(ret[i].second[:EXP][:successes]) for i in 1:length(inter_start_dir_list)]
	ret
end

# ╔═╡ ab4c5a80-0400-49e4-82e2-fc81051983b8
inter_res_1 = [inter_exp(
	r, 
	Dict(:cell=>"MARNN", 
		 :numhidden=>12, 
		 :truncation=>6, 
		 :eta => 0.0037252902984619128),
	base_args,
	ActionInterventionPersistent(1))[:EXP][:out_err] for r in 1:20]

# ╔═╡ be7e38a3-3476-495a-ab42-75d8b8b17673
plot(mean([rollmean(sqrt.(mean(reduce(hcat, inter_res_1[i]).^2, dims=1))[1, :] , 1000) for i in 1:20]))

# ╔═╡ Cell order:
# ╠═dc24843b-568a-4e2b-a998-994e025486b9
# ╠═3782e601-b4c2-4fcf-bae0-9df4d7d5dad6
# ╠═04f9a921-7648-4c81-855c-11dfc25fa26c
# ╠═edfc15af-382d-480a-9e4d-eb0a88e4481e
# ╠═409c3076-40aa-4d02-8924-38087d8f08da
# ╠═cbd23d8d-a9ee-4f6f-b7f0-f86f67ffd054
# ╠═1aa6ea1f-a6f8-4b93-b995-dd6b6646f2cf
# ╠═dcdd9654-0105-464e-88c3-0e6dc7c44b65
# ╠═9116f864-cfba-4f34-8dab-f53f290a523e
# ╠═90161e4a-79e3-4c24-a138-bbfe2f5e818b
# ╠═bfb9c7a8-15e5-490c-b639-401d1d2b26eb
# ╠═38f400f5-c389-4c73-ba16-640d34b0883a
# ╠═4d2261fc-f4e9-4d88-99c0-e25219480f0e
# ╠═d0dce888-6254-4477-af25-057be27351e8
# ╠═a29f06c2-6914-48ca-bde6-6a75beadc992
# ╠═f99084ba-21ec-4677-9e18-0963bebf6cf0
# ╠═8132f595-ce46-4a78-816e-3a6b279b3a06
# ╠═9154942d-8523-45ac-8c5b-fb8dac8c5ece
# ╠═e6e41135-a0e7-476f-a51d-b98787f7d699
# ╠═3f624303-938d-4eca-badc-c4af068167f9
# ╠═ffb26b02-a0d8-4731-8962-9fa2587abf88
# ╠═e78776da-715c-458f-8db2-4212a82418d5
# ╠═d6c234ea-823d-498b-8c14-a6e60ae374f6
# ╠═bad54c0a-2c48-4464-9d6a-a54427719f10
# ╠═f5533e88-c50e-4082-b746-debf0bafe90a
# ╠═14657dfb-a87b-416a-b15a-fc0bdb798293
# ╠═3f276e08-fd7b-4122-b1ff-968370ff2fca
# ╠═f81c9ceb-c6cc-4448-8dcb-f4b79aeb47b6
# ╠═ab4c5a80-0400-49e4-82e2-fc81051983b8
# ╠═be7e38a3-3476-495a-ab42-75d8b8b17673
