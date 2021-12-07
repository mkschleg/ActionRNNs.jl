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

function get_model(parsed, out_horde, fs, rng)

    nh = parsed["numhidden"]
    na = 2
    init_func = (dims...)->glorot_uniform(rng, dims...)
    num_gvfs = length(out_horde)

    chain = begin
        if parsed["cell"] ∈ ActionRNNs.fac_rnn_types()

            rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
            factors = parsed["factors"]
            init_style = get(parsed, "init_style", "standard")

            init_func = (dims...; kwargs...)->
                ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
            initb = (dims...; kwargs...) -> Flux.zeros(dims...)
            
            Flux.Chain(rnn(fs, na, nh, factors;
                           init_style=init_style,
                           init=init_func,
                           initb=initb),
                       Flux.Dense(nh, num_gvfs; initW=init_func))

        elseif parsed["cell"] ∈ ActionRNNs.fac_tuc_rnn_types()

            rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
            action_factors = parsed["action_factors"]
            out_factors = parsed["out_factors"]
            in_factors = parsed["in_factors"]
            init_func = (dims...; kwargs...)->
                ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
            initb = (dims...; kwargs...) -> Flux.zeros(dims...)

            Flux.Chain(rnn(fs, na, nh, action_factors, out_factors, in_factors;
                           init=init_func,
                           initb=initb),
                       Flux.Dense(nh, num_gvfs; initW=init_func))
            
        elseif parsed["cell"] ∈ ActionRNNs.rnn_types() && !get(parsed, "deep", false)

            rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
            
            init_func = (dims...; kwargs...)->
                ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
            initb = (dims...; kwargs...) -> Flux.zeros(dims...)
            
            m = Flux.Chain(
                rnn(fs, 2, nh;
                    init=init_func,
                    initb=initb),
                Flux.Dense(nh, num_gvfs; initW=init_func))
        elseif parsed["cell"] ∈ ActionRNNs.gated_rnn_types()

        rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))

        ninternal = parsed["internal"]

        init_func = (dims...; kwargs...)->
            ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
        initb = (dims...; kwargs...) -> Flux.zeros(dims...)

        m = Flux.Chain(
            rnn(fs, na, ninternal, nh;
                init=init_func,
                initb=initb),
            Flux.Dense(nh, num_gvfs; initW=init_func))

        elseif parsed["cell"]  ∈ ActionRNNs.rnn_types() && get(parsed, "deep", false)

            # Deep actions for RNNs from Zhu et al 2018
            
            rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
            
            internal_a = parsed["internal_a"]
            internal_o = parsed["internal_o"]
            
            init_func = (dims...; kwargs...)->
                ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
            initb = (dims...; kwargs...) -> Flux.zeros(dims...)

            action_stream = Flux.Chain(
                (a)->Flux.onehotbatch(a, 1:na),
                Flux.Dense(na, internal_a, Flux.relu, initW=init_func),
            )

            obs_stream = Flux.Chain(
                Flux.Dense(fs, internal_o, Flux.relu, initW=init_func)
            )
            
            m = Flux.Chain(
                ActionRNNs.DualStreams(action_stream, obs_stream),
                rnn(internal_o, internal_a, nh;
                    init=init_func,
                    initb=initb),
                Flux.Dense(nh, num_gvfs; initW=init_func))
 
        else
            
            rnntype = getproperty(Flux, Symbol(parsed["cell"]))
            Flux.Chain(rnntype(fs, nh; init=init_func),
                       Flux.Dense(nh,
                                  num_gvfs;
                                  initW=init_func))
        end
    end

    chain
end


function construct_agent(parsed, rng)

    fc = RWU.StandardFeatureCreator{false}()
    fs = size(fc)

    out_horde = RWU.get_horde(parsed, "out")
    τ = parsed["truncation"]

    ap = ActionRNNs.RandomActingPolicy([0.5, 0.5])

    opt = FLU.get_optimizer(parsed)

    chain = get_model(parsed, out_horde, fs, rng)

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
