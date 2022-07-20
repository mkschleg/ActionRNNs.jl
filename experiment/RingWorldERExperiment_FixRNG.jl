module RingWorldERExperiment_FixRNG

using MinimalRLCore

import ActionRNNs: ActionRNNs, ExpUtils, RingWorld, glorot_uniform
import ActionRNNs: @data
import .ExpUtils: RingWorldUtils, FluxUtils, experiment_wrapper, Macros, construct_logger
import .Macros: @info_str, @generate_config_funcs

import Flux
import JLD2
import LinearAlgebra.Diagonal

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random
using LoggingExtras

import ChoosyDataLoggers

const RWU = RingWorldUtils
const FLU = FluxUtils

function results_synopsis(res, ::Val{true})
    rmse = sqrt.(mean(res[:out_err].^2; dims=1))[1, :]
    Dict([
        # "description"=>"All operations are on the RMSE",
        "avg_all"=>mean(rmse),
        "avg_end"=>mean(rmse[end-50000:end]),
        "lc"=>mean(reshape(rmse, 1000, :); dims=1)[1,:],
        "var"=>var(reshape(rmse, 1000, :); dims=1)[1,:]
    ])
end
results_synopsis(res, ::Val{false}) = res

ChoosyDataLoggers.@init
function __init__()
    ChoosyDataLoggers.@register
end

@generate_config_funcs begin

    info"""
        Experiment details.
    --------------------
    - `seed::Int`: seed of RNG
    - `steps::Int`: Number of steps taken in the experiment
    - `synopsis::Bool`: Report full results or a synopsis.

    """
    "seed" => 1,
    "synopsis" => false,
    "steps" => 200000,

    info"""
    ### Logging Extras
    
    By default the experiment will log and save (depending on the synopsis flag) the logging group `:EXP`. 
    You can add extra logging groups and [group, name] pairs using the below arguments. Everything 
    added to `save_extras` will be passed to the save operation, and will be logged automatically. The 
    groups and names added to `log_extras` will be ommited from save_results but still passed back to the user
    through the data dict.

    - `<log_extras::Vector{Union{String, Vector{String}}>`: which group and <name> to log to the data dict. This **will not** be passed to save.
    - `<save_extras::Vector{Union{String, Vector{String}}>`: which groups and <names> to log to the data dict. This **will** be passed to save.
    """

    info"""
    Environment details
    -------------------
    This experiment uses the RingWorld environment. The usable args are:
    - `size::Int`: Number of states in the ring.
    """
    "size" => 6,
    
    info"""
    agent details
    -------------
    ### RNN
    The RNN used for this experiment and its total hidden size, 
    as well as a flag to use (or not use) zhu's deep 
    action network. See 
    - `cell::String`: The typeof cell. Many types are possible.
    - `deepaction::Bool`: Whether to use Zhu et. al.'s deep action 4 RNNs idea.
        -`internal_a::Int`: the size of the action representation layer when `deepaction=true`
    - `numhidden::Int`:  Size of hidden state in RNNs.   
    """
    "cell" => "MARNN",
    "numhidden" => 6,
    "deepaction" => false,

    info"""
    ### Prediction Problem
    Define the prediction problem for the experiment.
    - `outhorde::String`: The horde used to construct the targets.
    - `outgamma::Float64`: The discount (used by specific hordes).
    """
    "outhorde" => "onestep",
    "outgamma" => 0.9,

    info"""
    ### Optimizer details
    Flux optimizers are used. See flux documentation and `ExpUtils.Flux.get_optimizer` for details.
    - `opt::String`: The name of the optimizer used
    - Parameters defined by the particular optimizer.
    """
    "opt" => "RMSProp",
    "eta" => 0.001,
    "rho" => 0.9,

    info"""
    ### Learning update and replay details including:
    - Replay: 
        - `replay_size::Int`: How many transitions are stored in the replay.
        - `warm_up::Int`: How many steps for warm-up (i.e. before learning begins).
    """
    "replay_size"=>1000,
    "warm_up" => 1000,
    
    info"""
    - Update details: 
        - `batch_size::Int`: size of batch
        - `truncation::Int`: Length of sequences used for training.
        - `update_freq::Int`: Time between updates (counted in agent interactions)
        - `target_update_freq::Int`: Time between target network updates (counted in agent interactions)
        - `hs_learnable::Bool`: Strategy for dealing w/ hidden state in buffer.
    """
    "truncation" => 3,
    "batch_size"=>4,
    "update_freq"=>4,
    "target_update_freq"=>1000,
    "hs_learnable"=>true

    info"""
    Default Performance:
    --------------------
    ```
    Time: 0:00:56
    Dict{String, Matrix{Float32}} with 2 entries:
      "err"  => [0.0 0.0; 0.0 0.0; … ; 0.0128746 -0.000129551; 0.00117147 -0.00140008]
      "pred" => [0.0 0.0; 0.0 0.0; … ; 0.0128746 -0.000129551; 1.00117 -0.00140008]
    ```
    """
end

function build_ann(in, actions, out, config, rng)
    
    nh = config["numhidden"]
    init_func, initb = ActionRNNs.get_init_funcs(rng)

    deep_action = if "deepaction" ∈ keys(config)
        config["deepaction"]
    else
        get(config, "deep", false)
    end
    
    rnn = if deep_action
        internal_a = config["internal_a"]
    
        init_func, initb = ActionRNNs.get_init_funcs(rng)

        layers = get(config, "internal_a_layers", 1)
        action_stream = Flux.Chain(
            (a)->Flux.onehotbatch(a, 1:actions),
            Flux.Dense(actions, internal_a, Flux.relu, initW=init_func),
            (layers > 1 ? (Flux.Dense(internal_a, internal_a, Flux.relu, initW=init_func) for l in 2:layers) : ())...
        )

        obs_stream = identity
        
        (ActionRNNs.DualStreams(action_stream, obs_stream),
         ActionRNNs.build_rnn_layer(in, internal_a, nh, config, rng))

    else
        (ActionRNNs.build_rnn_layer(in, actions, nh, config, rng),)
    end

    Flux.Chain(rnn...,
               Flux.Dense(nh, out; initW=init_func))
    
end


function construct_agent(env, config, rng)

    fc = RWU.StandardFeatureCreator{false}()
    fs = size(fc)

    out_horde = RWU.get_horde(config, "out")
    τ = config["truncation"]

    ap = ActionRNNs.RandomActingPolicy([0.5, 0.5])

    opt = FLU.get_optimizer(config)

    chain = build_ann(fs, 2, length(out_horde), config, rng)

    ActionRNNs.DRTDNAgent(out_horde,
                          chain,
                          opt,
                          τ,
                          fc,
                          fs,
                          1,
                          config["replay_size"],
                          config["warm_up"],
                          config["batch_size"],
                          config["update_freq"],
                          config["target_update_freq"],
                          ap,
                          config["hs_learnable"])

end

function construct_env(config, rng=Random.default_rng())
    RingWorld(config["size"])
end

Macros.@generate_ann_size_helper
Macros.@generate_working_function


function main_experiment(config; progress=false, testing=false, overwrite=false)

    if "numhidden_factors" ∈ keys(config)
        config["numhidden"] = config["numhidden_factors"][1]
        config["factors"] = config["numhidden_factors"][2]
    end
    
    experiment_wrapper(config, use_git_info=false, testing=testing, overwrite=overwrite) do (config)

        num_steps = config["steps"]
        seed = config["seed"]
        rng = Random.MersenneTwister(seed)
        exp_rng = Random.Xoshiro(rand(Int, rng))

        extras = union(get(config, "log_extras", []), get(config, "save_extras", []))
        data, logger = ExpUtils.construct_logger(steps=num_steps, extra_groups_and_names=extras)
        
        with_logger(logger) do
            env = construct_env(config)
            agent = construct_agent(env, config, rng)
            experiment_loop(env, agent, config["outhorde"], num_steps, exp_rng; prgs=progress)

            @data EXPExtra env
            @data EXPExtra agent
        end

        # out_pred_strg = data[:EXP][:out_pred]
        # out_err_strg = data[:EXP][:out_err]
        # results = Dict(["pred"=>out_pred_strg, "err"=>out_err_strg])
        results = ExpUtils.prep_save_results(data, get(config, "save_extras", []))
        save_results = results_synopsis(results, Val(config["synopsis"]))
        (save_results=save_results, data=data)
    end
end

# Creating an environment for to run in jupyter.
function experiment_loop(env, agent, outhorde_str, num_steps, rng; prgs=false)

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    
    cur_step = 1
    MinimalRLCore.run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
        
        out_preds = a.preds[:, 1]
        
        @data EXP out_pred=out_preds idx=cur_step
        out_errs = out_preds .- Float32.(RWU.oracle(env, outhorde_str))
        @data EXP out_err=out_errs idx=cur_step
        
        if prgs
            ProgressMeter.next!(prg_bar)
        end

        cur_step += 1
    end
    
    # out_pred_strg, out_err_strg, out_loss_strg


    
end


end
