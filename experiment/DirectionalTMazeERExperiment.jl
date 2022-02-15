"""
    DirectionalTMazeERExperiment

An experiment to compare different RNN cells using the [`ActionRNNs.DirectionalTMaze`](@ref) environment.

Usage is detailed through the docs for 
- [`DirectionalTMazeERExperiment.default_config`](@ref)
- [`DirectionalTMazeERExperiment.main_experiment`](@ref)
- [`DirectionalTMazeERExperiment.working_experiment`](@ref)
- [`DirectionalTMazeERExperiment.construct_env`](@ref)
- [`DirectionalTMazeERExperiment.construct_agent`](@ref)

"""
module DirectionalTMazeERExperiment

# import Flux
import JLD2
import ActionRNNs: ActionRNNs, DirectionalTMaze, ExpUtils, MinimalRLCore, Flux

import MinimalRLCore: run_episode!, get_actions
import .ExpUtils: experiment_wrapper, TMazeUtils, FluxUtils
import .ExpUtils: SimpleLogger, UpdateStateAnalysis, l1_grad
import .ExpUtils: Macros
import .Macros: @info_str, @generate_config_funcs

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

const TMU = TMazeUtils
const FLU = FluxUtils
#=
"[[:space:]]+info\"\\{3\\}\n\\([[:space:]]+.+\n\\)+[[:space:]]+\"\\{3\\}"
=#
@generate_config_funcs begin

    info"""
    Experiment details.
    --------------------
    - `seed::Int`: seed of RNG
    - `steps::Int`: Number of steps taken in the experiment
    """
    "seed" => 2
    "steps" => 150000

    info"""
    Environment details
    -------------------
    This experiment uses the DirectionalTMaze environment. The usable args are:
    - `size::Int`: Size of the hallway in directional tmaze.
    """
    "size" => 10
    
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
    "deepaction" => false
    "numhidden" => 10

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
    "replay_size"=>20000
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
    "batch_size"=>8
    "truncation" => 12
    
    "update_wait"=>4
    "target_update_wait"=>1000

    "hs_strategy" => "minimize"

    info"""
    ## Default performance:
    ```
    Time: 0:02:28
      episode:    5385
      successes:  0.8351648351648352
      loss:       1.0
      l1:         0.0
      action:     2
      preds:      Float32[0.369189, 0.48326853, 0.993273]
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

        action_stream = Flux.Chain(
            (a)->Flux.onehotbatch(a, 1:actions),
            Flux.Dense(actions, internal_a, Flux.relu, initW=init_func),
        )

        #=
        The obs stream will always be the identity function to make comparisons with non-action embeddings fair.
        =#
        
        obs_stream = identity

        #=
        the pre-network is the [`DualStreams`](@ref) which allows for two paralle streams (for each of the tuple of inputs).
        =#
        
        (ActionRNNs.DualStreams(action_stream, obs_stream),
         ActionRNNs.build_rnn_layer(internal_o, internal_a, out, parsed, rng))
    else
        (ActionRNNs.build_rnn_layer(config, in, actions, nh, rng),)
    end # if deep_action

    Flux.Chain(rnn...,
               Flux.Dense(nh, actions; initW=init_func))
    
end

"""
    construct_agent

Construct the agent for `DirectionalTMazeERExperiment`. See 
"""
function construct_agent(env, config, rng)

    #=
    Standard feature creator is identity but changes features to Float32. Does not encorporate actions into state.
    =#
    
    fc = TMU.StandardFeatureCreator{false}()
    fs = MinimalRLCore.feature_size(fc)
    num_actions = length(get_actions(env))

    γ = Float32(config["gamma"])
    τ = config["truncation"]

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
function construct_env(config)
    DirectionalTMaze(config["size"])
end


Macros.@generate_ann_size_helper
Macros.@generate_working_function


"""
    main_experiment

Run an experiment from config. See [`DirectionalTMazeERExperiment.working_experiment`](@ref) 
for details on running on the command line and [`DirectionalTMazeERExperiment.default_config`](@ref) 
for info about the default configuration.
"""
function main_experiment(config;
                         progress=false,
                         testing=false,
                         overwrite=false)

    if "cell_numhidden" ∈ keys(config)
        @warn "\"cell_numhidden\" no longer supported. Use Reproduce utilities."
        config["cell"] = config["cell_numhidden"][1]
        config["numhidden"] = config["cell_numhidden"][2]
        delete!(config, "cell_numhidden")
    end

    experiment_wrapper(config; use_git_info=false, testing=testing, overwrite=overwrite) do config
        
        num_steps = config["steps"]

        seed = config["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = construct_env(config)
        agent = construct_agent(env, config, rng)

        logger = SimpleLogger(
            (:total_rews, :losses, :successes, :total_steps, :l1),
            (Float32, Float32, Bool, Int, Float32),
            Dict(
                :total_rews => (rew, steps, success, usa) -> rew,
                :losses => (rew, steps, success, usa) -> usa[:loss]/steps,
                :successes => (rew, steps, success, usa) -> success,
                :total_steps => (rew, steps, success, usa) -> steps,
                :l1 => (rew, steps, success, usa) -> usa[:l1]/steps
            )
        )

        
        prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
        generate_showvalues(eps, logger, usa, a, n) = () -> begin
            pr_suc = if length(logger.data.successes) <= 1000
                mean(logger.data.successes)
            else
                mean(logger.data.successes[end-1000:end])
            end
            [(:episode, eps),
             (:successes, pr_suc),
             (:loss, usa[:avg_loss]),
             (:l1, usa[:l1]/n),
             (:action, a.action),
             (:preds, a.preds)]
        end
            
        eps = 1
        while sum(logger.data.total_steps) <= num_steps
            usa = UpdateStateAnalysis(
                (l1 = 0.0f0, loss = 0.0f0, avg_loss = 1.0f0),
                Dict(
                    :l1 => l1_grad,
                    :loss => (s, us) -> s + us.loss,
                    :avg_loss => (s, us) -> 0.99 * s + 0.01 * us.loss
                ))
            success = false
            max_episode_steps = min(max((num_steps - sum(logger[:total_steps])), 2), 10000)
            n = 0
            total_rew, steps = run_episode!(env, agent, max_episode_steps, rng) do (s, a, s′, r)
                if progress
                    next!(prg_bar, showvalues=generate_showvalues(eps, logger, usa, a, n))
                end
                success = success || (r == 4.0)
                if !(a.update_state isa Nothing)
                    usa(a.update_state)
                end
                n+=1
            end
            logger(total_rew, steps, success, usa)
            eps += 1
        end
        save_results = logger.data
        (;save_results = save_results)
    end
end

end
