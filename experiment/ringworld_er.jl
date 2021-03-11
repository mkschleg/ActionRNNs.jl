module RingWorldExperiment

import Flux
# import Flux.Tracker
import JLD2
import LinearAlgebra.Diagonal

# include("../src/ActionRNNs.jl")
import ActionRNNs

using DataStructures: CircularBuffer
using ActionRNNs: RingWorld, step!, start!, glorot_uniform

# using ActionRNNs
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random
using MinimalRLCore

# using Plots

const RWU = ActionRNNs.RingWorldUtils
const FLU = ActionRNNs.FluxUtils

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

function default_arg_parse()
    Dict{String,Any}(
        "save_dir" => "ringworld",

        "seed" => 1,
        "steps" => 200000,
        "size" => 6,

        # "features" => "OneHot",
        # "cell" => "ARNN",
        "rnn_config" => "ARNN_OneHot",
        "numhidden" => 6,
        "hs_learnable" => true,
        
        "outhorde" => "gammas_term",
        "outgamma" => 0.9,
        
        "opt" => "RMSProp",
        "alpha" => 0.001,
        "truncation" => 3,

        "replay_size"=>1000,
        "warm_up" => 100,
        "batch_size"=>4,
        "update_wait"=>4,

        "synopsis" => false)

end

function get_rnn_config(parsed, out_horde, rng)

    rnn_config_str = parsed["rnn_config"]
    
    cell_str, fc_str = split(rnn_config_str, "_")

    fc = if fc_str == "OneHot"
        RWU.OneHotFeatureCreator()
    elseif fc_str == "SansAction"
        RWU.SansActionFeatureCreator()
    else
        throw(fc_str * " not a feature creator.")
    end
    
    fs = MinimalRLCore.feature_size(fc)
    
    init_func = (dims...)->glorot_uniform(rng, dims...)

    chain = begin
        if cell_str == "FacARNN"
            Flux.Chain(ActionRNNs.FacARNN(fs, 2, parsed["numhidden"], parsed["factors"], hs_learnable=parsed["hs_learnable"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func))
        elseif cell_str == "ARNN"
            Flux.Chain(ActionRNNs.ARNN(fs, 2, parsed["numhidden"], hs_learnable=parsed["hs_learnable"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func))
        elseif cell_str == "RNN"
            Flux.Chain(ActionRNNs.RNN(fs, parsed["numhidden"], hs_learnable=parsed["hs_learnable"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func))
        elseif cell_str == "ALSTM"
            Flux.Chain(ActionRNNs.ALSTM(fs, 2, parsed["numhidden"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func))
        elseif cell_str == "LSTM"
            Flux.Chain(Flux.LSTM(fs, parsed["numhidden"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func))
        else
            throw("Unknown Cell type " * cell_str)
        end
    end

    fc, fs, chain
end

function construct_agent(parsed, rng)


    # out_horde = RWU.gammas_term(collect(0.0:0.1:0.9))
    out_horde = RWU.get_horde(parsed, "out")

    fc, fs, chain = get_rnn_config(parsed, out_horde, rng)
    opt_func = getproperty(Flux, Symbol(parsed["opt"]))
    opt = opt_func(parsed["alpha"])

    
    ap = ActionRNNs.RandomActingPolicy([0.5, 0.5])
    τ = parsed["truncation"]

    ActionRNNs.PredERAgent(out_horde,
                           chain,
                           opt,
                           τ,
                           fc,
                           fs,
                           1,
                           parsed["replay_size"],
                           parsed["warm_up"],
                           parsed["batch_size"],
                           ap)

end

function main_experiment(parsed::Dict{String, Any}; working=false, progress=false)

    ActionRNNs.experiment_wrapper(parsed, working) do (parsed)
        # savefile = ActionRNNs.save_setup(parsed)
        # if isnothing(savefile)
        #     return
        # end
        
        num_steps = parsed["steps"]
        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = RingWorld(parsed)
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
    run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
        
        out_preds = a.preds
        out_pred_strg[cur_step, :] .= out_preds
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
