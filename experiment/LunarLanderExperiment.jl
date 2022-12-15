"""
    LunarLanderExperiment

Experiment module for running experiments in Lunar Lander.
"""
module LunarLanderExperiment

# include("../src/ActionRNNs.jl")

import MinimalRLCore: MinimalRLCore, run_episode!, get_actions
import ActionRNNs: ActionRNNs, LunarLander, ExpUtils

import .ExpUtils: SimpleLogger, UpdateStateAnalysis, l1_grad, experiment_wrapper
import .ExpUtils: LunarLanderUtils as LMU, FluxUtils as FLU, Macros
import .Macros: @info_str, @generate_config_funcs

import Flux
import Flux: gpu
import JLD2

import LinearAlgebra: BLAS

import ChoosyDataLoggers: ChoosyDataLoggers, @data

import Logging: with_logger

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random



ChoosyDataLoggers.@init
function __init__()
    ChoosyDataLoggers.@register
end

#=
Time: 0:01:08
  episode:     100
  total_rews:  -249.31464
  loss:        146.8288
  l1:          0.6579351
  action:      4
  preds:       Float32[-18.375952, -17.984146, -17.7312, -17.009594]
=#
@generate_config_funcs begin
    info"""
    Experiment details.
    --------------------
    - `seed::Int`: seed of RNG
    - `steps::Int`: Number of steps taken in the experiment
    """
    "seed" => 1
    "steps" => 10000

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
    This experiment uses the DirectionalTMaze environment. The usable args are:
    - `size::Int`: Size of the hallway in directional tmaze.
    """

    "omit_states" => [6]
    "state_conditions" => [2]

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
    "cell" => "MAGRU"
    "numhidden" => 64
    "deepaction" => false

    info"""
    ### Optimizer details
    Flux optimizers are used. See flux documentation and `ExpUtils.Flux.get_optimizer` for details.
    - `opt::String`: The name of the optimizer used
    - Parameters defined by the particular optimizer.
    """

    "opt" => "RMSProp"
    "eta" => 0.000355
    "rho" =>0.99

    info"""
    ### Learning update and replay details including:
    - Replay: 
        - `replay_size::Int`: How many transitions are stored in the replay.
        - `warm_up::Int`: How many steps for warm-up (i.e. before learning begins).
    """
    "replay_size"=>100000
    "warm_up" => 1000

    info"""
    - Update details: 
        - `lupdate::String`: Learning update name
        - `gamma::Float`: the discount for learning update.
        - `batch_size::Int`: size of batch
        - `truncation::Int`: Length of sequences used for training.
        - `update_wait::Int`: Time between updates (counted in agent interactions)
        - `target_update_wait::Int`: Time between target network updates (counted in agent interactions)
        - `hs_strategy::String`: Strategy for dealing w/ hidden state in buffer.
    """
    "gamma" => 0.99
    "batch_size"=>32
    "update_wait"=>8
    "target_update_wait"=>1000
    "truncation" => 16
    
    "hs_learnable" => "minimize"
    "encoding_size" => 128

    info"""
    ## Default Performance
    ```
    Time: 0:01:08
      episode:     100
      total_rews:  -249.31464
      loss:        146.8288
      l1:          0.6579351
      action:      4
      preds:       Float32[-18.375952, -17.984146, -17.7312, -17.009594]
    ```
    """
end


function build_ann(in, actions::Int, config, rng)
    
    nh = config["numhidden"]
    init_func, initb = ActionRNNs.get_init_funcs(rng)

    es = config["encoding_size"]

    deep_action = get(config, "deep", false) || get(config, "deepaction", false)


    # construction dictated to maintain consistancy from previous experiments. Looks awkward. Is kinda awkward....
    encoding_network = if deep_action

        internal_a = config["internal_a"]
        
        rnn_layer = ActionRNNs.build_rnn_layer(es, internal_a, nh, config, rng)

        action_stream = Flux.Chain(
            (a)->Flux.onehotbatch(a, 1:actions),
            Flux.Dense(actions, internal_a, Flux.relu, initW=init_func),
        )

        obs_stream = Flux.Chain(
            Flux.Dense(in, es, Flux.relu, initW=init_func)
        )

        (ActionRNNs.DualStreams(action_stream, obs_stream), rnn_layer)
    else
        
        rnn_layer = ActionRNNs.build_rnn_layer(es, actions, nh, config, rng)
        
        (Flux.Dense(in, es, Flux.relu; initW=init_func), rnn_layer)
    end



    Flux.Chain(encoding_network...,
               Flux.Dense(nh, nh, Flux.relu; initW=init_func),
               Flux.Dense(nh, actions; initW=init_func))

    
end

"""
    construct_agent

Construct the agent for lunar lander.
"""
function construct_agent(env, config, rng)

    fc = LMU.IdentityFeatureCreator()
    fs = MinimalRLCore.feature_size(fc, config["omit_states"])

    γ = Float32(config["gamma"])
    τ = config["truncation"]

    ap = ActionRNNs.ϵGreedyDecay((1.0, 0.05), 10000, 1000, MinimalRLCore.get_actions(env))

    opt = FLU.get_optimizer(config)

    chain = build_ann(fs, length(get_actions(env)), config, rng)# |> gpu

    hs_strategy = ActionRNNs.get_hs_strategy(
        "hs_learnable" ∈ keys(config) ? config["hs_learnable"] : config["hs_strategy"])
    
    ActionRNNs.DRQNAgent(chain,
                         opt,
                         τ,
                         ActionRNNs.QLearningSUM(γ),
                         fc,
                         fs,
                         3,
                         config["replay_size"],
                         config["warm_up"],
                         config["batch_size"],
                         config["update_wait"],
                         config["target_update_wait"],
                         ap,
                         hs_strategy)
end

"""
    construct_environment

"""
function construct_env(config)
    ActionRNNs.LunarLander(
        config["seed"],
        false,
        config["omit_states"],
        config["state_conditions"])
end

Macros.@generate_ann_size_helper
Macros.@generate_working_function

"""
    main_experiment

Run an experiment from config. See [`LunarLanderExperiment.working_experiment`](@ref) 
for details on running on the command line and [`LunarLanderExperiment.default_config`](@ref) 
for info about the default configuration.
"""
function main_experiment(config;
                         progress=false,
                         testing=false,
                         overwrite=false)

    GC.gc(true)

    if "SLURM_CPUS_PER_TASK" ∈ keys(ENV)
        avail_cores = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
        BLAS.set_num_threads(avail_cores)
    elseif "ARNN_CPUS_PER_TASK" ∈ keys(ENV)
        avail_cores = parse(Int, ENV["ARNN_CPUS_PER_TASK"])
        BLAS.set_num_threads(avail_cores)
    end

    ll_experiment_wrapper(config; use_git_info=false, testing=testing, overwrite=overwrite) do config, save_setup_ret

        num_steps = config["steps"]

        seed = config["seed"]
        rng = Random.MersenneTwister(seed)

        extras = union(get(config, "log_extras", []), get(config, "save_extras", []))
        # extra_proc = [c isa AbstractArray ? (Symbol(c[1]), Symbol(c[2])) : Symbol(c) for c in extras]
        data, logger = ExpUtils.construct_logger(extra_groups_and_names=extras)


        with_logger(logger) do
            env = construct_env(config)
            agent = construct_agent(env, config, rng)

            experiment_loop(env, agent, num_steps, config, data, save_setup_ret, rng; progress=progress)

            @data EXPExtra env
            @data EXPExtra agent
            
        end

        
        save_results = ExpUtils.prep_save_results(data, get(config, "save_extras", []))
        (;save_results = save_results)
    end
end


function ll_experiment_wrapper(exp_func::Function, config;
                               filter_keys=String[],
                               use_git_info=true,
                               hash_exclude_save_dir=true,
                               testing=false,
                               overwrite=false)

    SAVE_KEY = Reproduce.SAVE_KEY
    save_setup_ret = if SAVE_KEY ∉ keys(config)
        if isinteractive() 
            @warn "No arg at \"$(SAVE_KEY)\". Assume testing in repl." maxlog=1
            config[SAVE_KEY] = nothing
        elseif testing
            @warn "No arg at \"$(SAVE_KEY)\". Testing Flag Set." maxlog=1
            config[SAVE_KEY] = nothing
        else
            @error "No arg found at $(SAVE_KEY). Please use savetypes here."
        end
        nothing
    else
        save_setup_ret = Reproduce.save_setup(config;
                                              filter_keys=filter_keys,
                                              use_git_info=use_git_info,
                                              hash_exclude_save_dir=hash_exclude_save_dir)
        
        if Reproduce.check_experiment_done(config, save_setup_ret) && !overwrite
            Reproduce.post_save_setup(config[SAVE_KEY])
            return
        end
        save_setup_ret
    end

    Reproduce.post_save_setup(config[SAVE_KEY])

    ret = exp_func(config, save_setup_ret)

    if ret isa NamedTuple
        Reproduce.save_results(config[SAVE_KEY], save_setup_ret, ret.save_results)
    else
        Reproduce.save_results(config[SAVE_KEY], save_setup_ret, ret)
    end
    
    Reproduce.post_save_results(config[SAVE_KEY])
    
    if isinteractive() || testing
        ret
    end
end


function experiment_loop(env, agent, num_steps, config, data, save_setup_ret, rng; progress=false)

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")

    eps = 1
    total_steps = 0
    checkpoint = 1
    
    while total_steps <= num_steps

        # checkpoint
        if total_steps > checkpoint * 500000
            data_dir = dirname(save_setup_ret)
            save_results = ExpUtils.prep_save_results(data, get(config, "save_extras", []))
            Reproduce.save_results(config["_SAVE"], joinpath(data_dir, "results_temp.jld2"), save_results)
            
            GC.gc()
            checkpoint += 1
        end

        # update analysis
        usa = UpdateStateAnalysis(
            (l1 = 0.0f0, loss = 0.0f0, avg_loss = 1.0f0),
            Dict(
                :l1 => l1_grad,
                :loss => (s, us) -> s + us.loss,
                :avg_loss => (s, us) -> 0.99 * s + 0.01 * us.loss
            ))
        success = false
        
        max_episode_steps = min(max((num_steps - total_steps), 2), 10000)
        n = 0
        
        tr = if (:EXP ∈ keys(data)) && (:total_rews ∈ keys(data[:EXP])) #&& (length(data[:EXP][:total_rews]) > 100)
            d = data[:EXP][:total_rews]
            mean(d[(end - min(length(d)-1, 100)) : end])
        else
            0.0f0
        end
        
        total_rew, steps = run_episode!(env, agent, max_episode_steps, rng) do (s, a, s′, r)
            if progress
                next!(prg_bar, showvalues=[(:episode, eps),
                                           (:total_rews, tr),
                                           (:loss, usa[:avg_loss]),
                                           (:l1, usa[:l1]/n),
                                           (:action, a.action),
                                           (:preds, a.preds)])

            end
            
            n+=1
        end

        total_steps += steps

        @data EXP total_rews=total_rew
        @data EXP total_steps=steps

        @data EXPExtra losses=usa[:loss]/steps
        @data EXPExtra l1=usa[:l1]/steps

        eps += 1
    end
end



end
