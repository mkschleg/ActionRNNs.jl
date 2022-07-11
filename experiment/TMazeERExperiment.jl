module TMazeERExperiment

import Flux
import JLD2

import MinimalRLCore: MinimalRLCore, run_episode!, get_actions
import ActionRNNs: ActionRNNs, TMaze, ExpUtils, DRQNAgent
import ActionRNNs: @data

import .ExpUtils: experiment_wrapper, TMazeUtils, FluxUtils, Macros
import .ExpUtils: SimpleLogger, UpdateStateAnalysis, l1_grad
import .Macros: @info_str, @generate_config_funcs

import Logging: with_logger

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

const TMU = TMazeUtils
const FLU = FluxUtils


@generate_config_funcs begin
    info"""
    Experiment details.
    --------------------
    - `seed::Int`: seed of RNG
    - `steps::Int`: Number of steps taken in the experiment
    """
    "seed" => 1
    "steps" => 80000

    info"""
    Environment details
    -------------------
    This experiment uses the TMaze environment. The usable args are:
    - `size::Int`: Size of the hallway in tmaze.
    """
    "size" => 6
    
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
    "cell" => "MARNN"
    "deepaction" => false
    "numhidden" => 6

    info"""
    ### Optimizer details
    Flux optimizers are used. See flux documentation and `ExpUtils.Flux.get_optimizer` for details.
    - `opt::String`: The name of the optimizer used
    - Parameters defined by the particular optimizer.
    """
    "opt" => "RMSProp"
    "eta" => 0.0005
    "rho" =>0.99

    info"""
    ### Learning update and replay details including:
    - Replay: 
        - `replay_size::Int`: How many transitions are stored in the replay.
        - `warm_up::Int`: How many steps for warm-up (i.e. before learning begins).
    """
    "replay_size"=>1000
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

    "lupdate" => "QLearning"
    "gamma"=>0.99    
    "batch_size"=>4
    "truncation" => 8
    
    "update_wait"=>4
    "target_update_wait"=>1000

    "hs_strategy" => "minimize"

    info"""
    ## Default performance:
    ```
    Time: 0:01:19
      episode:    6455
      successes:  0.9600399600399601
      loss:       0.990142
      l1:         0.000502145
      action:     2
      preds:      Float32[0.3016336, 3.6225605, -2.5592222, 1.884988]
      grad:       0.0
    ```
    """
end


function build_ann(config, in, actions::Int, rng)
    
    nh = config["numhidden"]
    init_func, initb = ActionRNNs.get_init_funcs(rng)

    deep_action = "deep" ∈ keys(config) ? config["deep"] : get(config, "deepaction", false)

    rnn = if deep_action
        
        #=
        If we are using [zhu et al 2018]() style action embeddings
        
        DirectionalTMazeERExperiment only re-embeds the action encoding. Here we encode the integer as a 
        one-hot encoding then pass to a dense layer w/ relu activation.
        =#
        
        internal_a = config["internal_a"]

        layers = get(config, "internal_a_layers", 1)
        action_stream = Flux.Chain(
            (a)->Flux.onehotbatch(a, 1:actions),
            Flux.Dense(actions, internal_a, Flux.relu, initW=init_func),
            (layers > 1 ? (Flux.Dense(internal_a, internal_a, Flux.relu, initW=init_func) for l in 2:layers) : ())...
        )


        #=
        The obs stream will always be the identity function to make comparisons with non-action embeddings fair.
        =#
        
        obs_stream = identity

        #=
        the pre-network is the [`DualStreams`](@ref) which allows for two paralle streams (for each of the tuple of inputs).
        =#
        
        (ActionRNNs.DualStreams(action_stream, obs_stream),
         ActionRNNs.build_rnn_layer(config, in, internal_a, nh, rng))
    else
        (ActionRNNs.build_rnn_layer(config, in, actions, nh, rng),)
    end # if deep_action

    Flux.Chain(rnn...,
               Flux.Dense(nh, actions; initW=init_func))
    
end

function construct_agent(env, config, rng)

    #=
    Standard feature creator is identity but changes features to Float32. Does not encorporate actions into state.
    =#
    
    fc = TMU.StandardFeatureCreator{false}()
    fs = MinimalRLCore.feature_size(fc)
    num_actions = length(get_actions(env))

    γ = Float32(config["gamma"])
    τ = config["truncation"]
    
    # fc = TMU.StandardFeatureCreator{false}()
    # fs = MinimalRLCore.feature_size(fc)

    # γ = Float32(config["gamma"])
    # τ = config["truncation"]
        #=
    Set policy to always be [`ϵGreedy`](@ref)
    =#
    
    ap = ActionRNNs.ϵGreedy(0.1, MinimalRLCore.get_actions(env))

    #=
    construct the optimizer from Flux. Only consider standard flux optimizers.
    =#
    
    opt = FLU.get_optimizer(config)

    #=
    Use config to build the neural network. See [`build_ann`](@ref) for more details.
    =#
    
    chain = build_ann(config, fs, num_actions, rng)


        #=
    Figuring out the hs_strategy. "hs_learnable" is for legacy experiments where we assumed
    true => minimize
    false => stale.
    Otherwise it should be a string or symbol which passes to [`ActionRNNs.get_hs_strategy`](@ref).
    =#
    
    hs_strategy = ActionRNNs.get_hs_strategy(
        "hs_learnable" ∈ keys(config) ? config["hs_learnable"] : config["hs_strategy"])
     
    #=
    This looks scary, but isn't.
    =#

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
    construct_env

Construct direction tmaze using:
- `size::Int` size of hallway.
"""
function construct_env(config, rng=Random.default_rng())
    ActionRNNs.TMaze(config["size"])
end

Macros.@generate_ann_size_helper
Macros.@generate_working_function

function main_experiment(config;
                         progress=false,
                         testing=false,
                         overwrite=false)

    if "numhidden_factors" ∈ keys(config)
        @warn "\"numhidden_factors\" no longer supported. Use Reproduce utilities."
        config["numhidden"] = config["numhidden_factors"][1]
        config["factors"] = config["numhidden_factors"][2]
    end

    experiment_wrapper(config; use_git_info=false, testing=testing, overwrite=overwrite) do config
        

        num_steps = config["steps"]

        seed = config["seed"]
        rng = Random.MersenneTwister(seed)
        
        extras = union(get(config, "log_extras", []), get(config, "save_extras", []))
        data, logger = ExpUtils.construct_logger(extra_groups_and_names=extrsa

        with_logger(logger) do
            env = construct_env(config)
            agent = construct_agent(env, config, rng)
            
            experiment_loop(env, agent, num_steps, data, rng, progress=progress)
            
            @data EXPExtra env
            @data EXPExtra agent
        end

        
        save_results = ExpUtils.prep_save_results(data, get(config, "save_extras", []))
        (;save_results = save_results, data=data)
        # save_results = logger.data
        # (;save_results = save_results)
    end
end

function experiment_loop(env, agent, num_steps, data, rng; progress=false)

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    generate_showvalues(eps, usa, a, n) = () -> begin
        pr_suc = if (:EXP ∈ keys(data)) && (:successes ∈ keys(data[:EXP]))
            if (length(data[:EXP][:successes]) <= 1000)
                mean(data[:EXP][:successes])
            else
                mean(data[:EXP][:successes][end-1000:end])
            end
        end
        [(:episode, eps),
         (:successes, pr_suc),
         (:loss, usa[:avg_loss]),
         (:l1, usa[:l1]/n),
         (:action, a.action),
         (:preds, a.preds)]
    end
    
    eps = 1
    total_steps = 0
    while total_steps <= num_steps
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
        
        total_rew, steps = run_episode!(env, agent, max_episode_steps, rng) do (s, a, s′, r)
            if progress
                next!(prg_bar, showvalues=generate_showvalues(eps, usa, a, n))
            end
            success = success || (r == 4.0)
            if !(a.update_state isa Nothing)
                usa(a.update_state)
            end
            n+=1
        end
        
        total_steps += steps

        @data EXP total_rews=total_rew
        @data EXP successes=success
        @data EXP total_steps=steps
        
        @data EXPExtra losses=usa[:loss]/steps
        @data EXPExtra l1=usa[:l1]/steps

        eps += 1
    end
end

end
