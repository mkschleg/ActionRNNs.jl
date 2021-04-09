module RingWorldExperiment

import Flux
# import Flux.Tracker
import JLD2
import LinearAlgebra.Diagonal

# include("../src/ActionRNNs.jl")
import ActionRNNs
import MinimalRLCore

using DataStructures: CircularBuffer
using ActionRNNs: RingWorld, step!, start!, glorot_uniform

# using ActionRNNs
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random


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

function default_args()
    Dict{String,Any}(

        "agent"=>"new",
        "save_dir" => "ringworld",

        "seed" => 1,
        "steps" => 200000,
        "size" => 6,

        # "features" => "OneHot",
        # "cell" => "ARNN",
        "cell" => "MARNN",
        "numhidden" => 6,
        "hs_learnable" => true,
        
        "outhorde" => "onestep",
        "outgamma" => 0.9,
        
        "opt" => "RMSProp",
        "eta" => 0.001,
        "rho" => 0.9,
        "truncation" => 3,

        "action_features"=>false,

        "replay_size"=>1000,
        "warm_up" => 100,
        "batch_size"=>4,
        "update_freq"=>1,
        "target_update_freq"=>1,

        "synopsis" => false)

end

function get_model(parsed, out_horde, fc, rng)

    nh = parsed["numhidden"]
    init_func = (dims...)->glorot_uniform(rng, dims...)
    fs = size(fc)
    num_gvfs = length(out_horde)
    @info fs
    
    chain = begin
        if parsed["cell"] == "FacARNN"
            
            factors = parsed["factors"]
            Flux.Chain(ActionRNNs.FacARNN(fs, 2, nh, factors; init=init_func),
                       Flux.Dense(nh, num_gvfs; initW=init_func))
            
        elseif parsed["cell"] ∈ ActionRNNs.rnn_types()

            rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
            
            init_func = (dims...; kwargs...)->
                ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
            initb = (dims...; kwargs...) -> Flux.zeros(dims...)
            
            m = Flux.Chain(
                rnn(fs, 4, nh;
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

    fc = RWU.StandardFeatureCreator{parsed["action_features"]}()
    fs = size(fc)

    out_horde = RWU.get_horde(parsed, "out")

    chain = get_model(parsed, out_horde, fc, rng)
    opt = FLU.get_optimizer(parsed)

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

function construct_new_agent(parsed, rng)

    fc = RWU.StandardFeatureCreator{parsed["action_features"]}()
    fs = size(fc)

    out_horde = RWU.get_horde(parsed, "out")

    chain = get_model(parsed, out_horde, fc, rng)
    opt = FLU.get_optimizer(parsed)

    ap = ActionRNNs.RandomActingPolicy([0.5, 0.5])
    τ = parsed["truncation"]
    
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

function main_experiment(parsed=default_args(); working=false, progress=false)

    ActionRNNs.experiment_wrapper(parsed, working) do (parsed)
        
        num_steps = parsed["steps"]
        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = RingWorld(parsed)
        agent = if parsed["agent"] == "new"
            construct_new_agent(parsed, rng)
        else
            construct_agent(parsed, rng)
        end

        @show typeof(agent)
        
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
