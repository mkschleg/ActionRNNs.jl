module RingWorldERExperiment

using MinimalRLCore

import ActionRNNs: ActionRNNs, ExpUtils, RingWorld, glorot_uniform

import .ExpUtils: RingWorldUtils, FluxUtils, experiment_wrapper


import Flux
import JLD2
import LinearAlgebra.Diagonal

# using DataStructures: CircularBuffer

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random


# using Plots

const RWU = RingWorldUtils
const FLU = FluxUtils

function results_synopsis(res, ::Val{true})
    rmse = sqrt.(mean(res["err"].^2; dims=2))
    Dict([
        "desc"=>"All operations are on the RMSE",
        "all"=>mean(rmse),
        "end"=>mean(rmse[end-50000:end]),
        "lc"=>mean(reshape(rmse, 1000, :); dims=1)[1,:],
        "var"=>var(reshape(rmse, 1000, :); dims=1)[1,:]
    ])
end
results_synopsis(res, ::Val{false}) = res

#=

Time: 0:00:56
Dict{String, Matrix{Float32}} with 2 entries:
  "err"  => [0.0 0.0; 0.0 0.0; … ; 0.0128746 -0.000129551; 0.00117147 -0.00140008]
  "pred" => [0.0 0.0; 0.0 0.0; … ; 0.0128746 -0.000129551; 1.00117 -0.00140008]

=#
function default_config()
    Dict{String,Any}(

        "save_dir" => "tmp/ringworld",
        "seed" => 1,
        "synopsis" => false,
        
        "steps" => 200000,
        "size" => 6,

        # Network
        "cell" => "MARNN",
        "numhidden" => 6,

        # Problem
        "outhorde" => "onestep",
        "outgamma" => 0.9,

        # Opt
        "opt" => "RMSProp",
        "eta" => 0.001,
        "rho" => 0.9,

        # BPTT
        "truncation" => 3,

        # Replay
        "replay_size"=>1000,
        "warm_up" => 1000,
        "batch_size"=>4,
        "update_freq"=>4,
        "target_update_freq"=>1000,
        "hs_learnable"=>true)

end

function build_deep_action_rnn_layers(in, actions, out, parsed, rng)

    # Deep actions for RNNs from Zhu et al 2018
    internal_a = parsed["internal_a"]
    
    init_func, initb = ActionRNNs.get_init_funcs(rng)
    
    action_stream = Flux.Chain(
        (a)->Flux.onehotbatch(a, 1:actions),
        Flux.Dense(actions, internal_a, Flux.relu, initW=init_func),
    )

    obs_stream = identity
        # Flux.Dense(in, internal_o, Flux.relu, initW=init_func)
    # )
    
    (ActionRNNs.DualStreams(action_stream, obs_stream),
     ActionRNNs.build_rnn_layer(in, internal_a, out, parsed, rng))
end

function build_ann(in, actions, out, parsed, rng)
    
    nh = parsed["numhidden"]
    init_func, initb = ActionRNNs.get_init_funcs(rng)

    deep_action = get(parsed, "deep", false)
    rnn = if deep_action
        build_deep_action_rnn_layers(in, actions, nh, parsed, rng)
    else
        (ActionRNNs.build_rnn_layer(in, actions, nh, parsed, rng),)
    end

    Flux.Chain(rnn...,
               Flux.Dense(nh, actions; initW=init_func))
    
end


function construct_agent(parsed, rng)

    fc = RWU.StandardFeatureCreator{false}()
    fs = size(fc)

    out_horde = RWU.get_horde(parsed, "out")
    τ = parsed["truncation"]

    ap = ActionRNNs.RandomActingPolicy([0.5, 0.5])

    opt = FLU.get_optimizer(parsed)

    chain = build_ann(fs, 2, length(out_horde), parsed, rng)

    ActionRNNs.DRTDNAgent(out_horde,
                          chain,
                          opt,
                          τ,
                          fc,
                          fs,
                          1,
                          parsed["replay_size"],
                          parsed["warm_up"],
                          parsed["batch_size"],
                          parsed["update_freq"],
                          parsed["target_update_freq"],
                          ap,
                          parsed["hs_learnable"])

end

function main_experiment(parsed=default_config(); working=false, progress=false, overwrite=false)

    if "numhidden_factors" ∈ keys(parsed)
        parsed["numhidden"] = parsed["numhidden_factors"][1]
        parsed["factors"] = parsed["numhidden_factors"][2]
    end
    
    experiment_wrapper(parsed, working, overwrite=overwrite) do (parsed)
        
        num_steps = parsed["steps"]
        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = RingWorld(parsed["size"])
        agent = construct_agent(parsed, rng)

        out_pred_strg, out_err_strg =
            experiment_loop(env, agent, parsed["outhorde"], num_steps, rng; prgs=progress)
        
        results = Dict(["pred"=>out_pred_strg, "err"=>out_err_strg])
        save_results = results_synopsis(results, Val(parsed["synopsis"]))
        (save_results=save_results)
    end
end

# Creating an environment for to run in jupyter.
function experiment_loop(env, agent, outhorde_str, num_steps, rng; prgs=false)

    out_pred_strg = zeros(Float32, num_steps, length(agent.horde))
    out_err_strg = zeros(Float32, num_steps, length(agent.horde))
    out_loss_strg = zeros(Float32, num_steps)

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    
    cur_step = 1
    MinimalRLCore.run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
        
        out_preds = a.preds
        
        out_pred_strg[cur_step, :] .= out_preds[:,1]
        out_err_strg[cur_step, :] = out_pred_strg[cur_step, :] .- RWU.oracle(env, outhorde_str);
        if !(a.update_state isa Nothing)
            out_loss_strg[cur_step] = a.update_state.loss
        end
        
        if prgs
            ProgressMeter.next!(prg_bar)
        end

        cur_step += 1
    end
    
    out_pred_strg, out_err_strg, out_loss_strg
    
end


end
