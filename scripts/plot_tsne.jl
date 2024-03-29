


import ActionRNNs: MinimalRLCore
import ActionRNNs: @data
using ActionRNNs, Random
using RollingFunctions

import RingWorldERExperiment

using Plots
import Plots: @colorant_str



function ActionRNNs.ChoosyDataLoggers.process_data(::Val{:get_hs_1_layer}, data)
    data[collect(keys(data))[1]]
end

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

function some_name(args)
    args["log_extras"] = [["EXPExtra", "agent"], ["EXPExtra", "env"]]
    test_ret = get_hs_over_time(false) do
	ret = RingWorldERExperiment.main_experiment(args)
	(ret.data, 
	 ret.data[:EXPExtra][:agent][1], 
	 ret.data[:EXPExtra][:env][1])
    end
    data = test_ret[2]
    hs = data[:Agent][:hidden_state]
    s = data[:Env][:state]
    obs = nothing
    a_tm1 = data[:Agent][:action]

    	
    Random.seed!(3)
    base_colors = distinguishable_colors(12, colorant"blue")
    colors = fill(base_colors[1], length(hs))
    for i in 1:10, j in 1:2
	colors[(s .== i) .&& (a_tm1 .== j)] .= base_colors[i]
    end
    data = tsne(collect(reduce(hcat, hs)'), 2, 0, 1000, progress=false)
    plt = scatter(data[:, 1], data[:, 2], color=colors, legend=nothing, title="Multiplicative")
    plt
end
