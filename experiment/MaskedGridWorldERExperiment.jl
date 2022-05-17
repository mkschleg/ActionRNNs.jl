"""
    MaskedGridWorldERExperiment

Module for running a standard experiment in masked grid world.
An experiment to compare different RNN cells using the [`ActionRNNs.MaskedGridWorld`](@ref) environment.

Usage is detailed through the docs for 
- [`MaskedGridWorldERExperiment.default_config`](@ref)
- [`MaskedGridWorldERExperiment.main_experiment`](@ref)
- [`MaskedGridWorldERExperiment.working_experiment`](@ref)
- [`MaskedGridWorldERExperiment.construct_env`](@ref)
- [`MaskedGridWorldERExperiment.construct_agent`](@ref)

"""
module MaskedGridWorldERExperiment

import Flux
import JLD2

import MinimalRLCore: MinimalRLCore, run_episode!, get_actions
import ActionRNNs: ActionRNNs, DirectionalTMaze, ExpUtils
import ActionRNNs: @data

import .ExpUtils: experiment_wrapper, TMazeUtils, FluxUtils, Macros
import .ExpUtils: SimpleLogger, UpdateStateAnalysis, l1_grad
import .Macros: @generate_config_funcs
import .Macros: @info_str, @generate_config_funcs


import Logging: with_logger

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

import ChoosyDataLoggers

ChoosyDataLoggers.@init
function __init__()
    ChoosyDataLoggers.@register
end


const FLU = FluxUtils


@generate_config_funcs quote
    info"""
            Experiment details.
            --------------------
            - `seed::Int`: seed of RNG
            - `steps::Int`: Number of steps taken in the experiment
            """
    "seed" => 2 # seed of RNG
    "steps" => 150000 # number of total steps in experiment.
    
    info"""
            Environment details
            -------------------
            This experiment uses the MaskedGridWorldExperiment environment. The usable args are:
            - `width::Int, height::Int`: Width and height of the grid world
            - `num_anchors::Int`: number of states with active observations
            - `num_goals::Int`: number of goals
            """
    "width" => 10
    "height" => 10
    "num_anchors" => 10
    "num_goals" => 1
    "goal_rew" => 4.0

    info"""
            agent details
            -------------
            ### RNN
            The RNN used for this experiment and its total hidden size, 
            as well as a flag to use (or not use) zhu's deep 
            action network. See 
            - `cell::String`: The typeof cell. Many types are possible.
            - `deepaction::Bool`: Whether to use Zhu et. al.'s deep action 4 RNNs idea.
                - `internal_a::Int`: the size of the action representation layer when `deepaction=true`
            - `numhidden::Int`:  Size of hidden state in RNNs.   
            """
    "cell" => "MARNN"
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
            - `lupdate_agg::String`: the aggregation function for the QLearning update.
            - `gamma::Float`: the discount for learning update.
            - `batch_size::Int`: size of batch
            - `truncation::Int`: Length of sequences used for training.
            - `update_wait::Int`: Time between updates (counted in agent interactions)
            - `target_update_wait::Int`: Time between target network updates (counted in agent interactions)
            - `hs_strategy::String`: Strategy for dealing w/ hidden state in buffer.
        """
    "lupdate_agg" => "SUM"
    "gamma"=>0.9    
    "batch_size"=>8
    "truncation" => 1
    
    "update_wait"=>4
    "target_update_wait"=>1000

    "hs_strategy" => "minimize"

    """
    ```julia
    julia> MaskedGridWorldERExperiment.working_experiment()
    ┌ Warning: No arg at "_SAVE". Assume testing in repl.
    └ @ Reproduce /Users/Matt/.julia/packages/Reproduce/SGO8J/src/exp_util.jl:60
    Step: 100%|███████████████████████████████████████████| Time: 0:01:45
      episode:  3032
      loss:     1.2959449
      action:   1
      preds:    Float32[0.36110675, 0.17198458, 0.121491514, 0.23019025]
    Episode: 100%|████████████████████████████████████████| Time: 0:00:00
    (save_results = Dict{Symbol, AbstractArray}(:total_rews => Float32[0.5000012, -17.600046, 0.8000009, -22.200064, -10.100018, -8.400011, -5.5, -0.9999976, -6.800005, -10.900021  …  -1.8999968, -24.700073, 2.6999998, -18.100048, -13.800032, 2.3999996, -12.700027, 3.4, -7.799995, -0.1], :total_steps => [36, 217, 33, 263, 142, 125, 96, 51, 109, 150  …  60, 288, 14, 222, 179, 17, 168, 7, 78, 1], :test_total_steps => [20, 69, 26, 51, 2, 3, 316, 71, 68, 41  …  185, 86, 31, 28, 1, 4, 39, 34, 109, 186], :test_total_rews => Float32[2.0999997, -2.799996, 1.5000002, -0.9999976, 3.9, 3.8, -27.500084, -2.9999957, -2.699996, 1.66893f-6  …  -14.400034, -4.499996, 1.0000007, 1.3000004, 4.0, 3.7, 0.20000148, 0.700001, -6.800005, -14.500034]), data = Dict{Symbol, Dict{Symbol, AbstractArray}}(:EXP => Dict(:total_rews => Float32[0.5000012, -17.600046, 0.8000009, -22.200064, -10.100018, -8.400011, -5.5, -0.9999976, -6.800005, -10.900021  …  -1.8999968, -24.700073, 2.6999998, -18.100048, -13.800032, 2.3999996, -12.700027, 3.4, -7.799995, -0.1], :total_steps => [36, 217, 33, 263, 142, 125, 96, 51, 109, 150  …  60, 288, 14, 222, 179, 17, 168, 7, 78, 1], :test_total_steps => [20, 69, 26, 51, 2, 3, 316, 71, 68, 41  …  185, 86, 31, 28, 1, 4, 39, 34, 109, 186], :test_total_rews => Float32[2.0999997, -2.799996, 1.5000002, -0.9999976, 3.9, 3.8, -27.500084, -2.9999957, -2.699996, 1.66893f-6  …  -14.400034, -4.499996, 1.0000007, 1.3000004, 4.0, 3.7, 0.20000148, 0.700001, -6.800005, -14.500034])))
    ```
    """
end

function build_ann(in, actions::Int, config, rng=Random.GLOBAL_RNG)
    
    nh = config["numhidden"]
    init_func, initb = ActionRNNs.get_init_funcs(rng)

    deepaction = get(config, "deepaction", false)
    
    rnn = if deepaction

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
         ActionRNNs.build_rnn_layer(in, internal_a, nh, config, rng))
    else
        (ActionRNNs.build_rnn_layer(in, actions, nh, config, rng),)
    end # if deep_action

    Flux.Chain(rnn...,
               Flux.Dense(nh, actions; initW=init_func))
    
end


"""
    construct_agent

Construct the agent for `MaskedGridWorldERExperiment`. See 
"""
function construct_agent(env, config, rng)


    #=
    Identity feature creator just returns the observations of the environment
    =#
    fc = ActionRNNs.IdentityFeatureCreator(ActionRNNs.obs_size(env))
    fs = MinimalRLCore.feature_size(fc)

    num_actions = length(get_actions(env))

    #=
    Set policy to always be [`ϵGreedy`](@ref)
    =#
    ap = ActionRNNs.ϵGreedy(0.1, MinimalRLCore.get_actions(env))

    #=
    Learning update is always QLearningSum
    =#
    lupdate_agg = config["lupdate_agg"]
    lu = if lupdate_agg == "SUM"
        ActionRNNs.QLearningSUM(Float32(config["gamma"]))
    elseif lupdate_agg == "HUBER"
        ActionRNNs.QLearningHUBER(Float32(config["gamma"]))
    else
        @error "$(lupdate_agg) not supported with QLearning"
    end
    τ = config["truncation"]
    
    opt = FLU.get_optimizer(config)
    chain = build_ann(fs, num_actions, config, rng)

    ActionRNNs.DRQNAgent(chain,
                         opt,
                         τ,
                         lu,
                         fc,
                         fs,
                         ActionRNNs.obs_size(env),
                         # configure replay buffer
                         config["replay_size"],
                         config["warm_up"],
                         config["batch_size"],
                         config["update_wait"],
                         config["target_update_wait"],
                         #
                         ap, # acting policy
                         ActionRNNs.get_hs_strategy(config["hs_strategy"]))
end

"""
    construct_env

Construct MaskedGridWorld. settings
- "width": width of gridworld
- "height": height of gridworld
- "num_anchors": number of anchors
- "num_goals": number of goals
"""
function construct_env(config, rng)
    ActionRNNs.MaskedGridWorld(config["width"],
                               config["height"],
                               config["num_anchors"],
                               (config["num_goals"], Float32(config["goal_rew"])),
                               rng;
                               obs_strategy=:aliased,
                               pacman_wrapping=true)
end

Macros.@generate_ann_size_helper
Macros.@generate_working_function


"""
    main_experiment

Run an experiment from config. See [`MaskedGridWorldERExperiment.working_experiment`](@ref) 
for details on running on the command line and [`MaskedGridWorldERExperiment.default_config`](@ref) 
for info about the default configuration.
"""
function main_experiment(config = default_config(); progress=false, testing=false, overwrite=false)

    experiment_wrapper(config;
                       use_git_info=false,
                       testing=testing,
                       overwrite=overwrite,
                       hash_exclude_save_dir=true) do config
                        

        # Initialize the default RNG seed. use global rngs.
        seed = config["seed"]
        Random.seed!(seed)

        extras = union(get(config, "log_extras", []), get(config, "save_extras", []))
        extra_proc = [c isa AbstractArray ? (Symbol(c[1]), Symbol(c[2])) : Symbol(c) for c in extras]
        data, logger = ExpUtils.construct_logger(extra_groups_and_names=extra_proc)


        with_logger(logger) do

            #=
            Construct environment and agent
            =#
            env = construct_env(config, Random.default_rng())
            agent = construct_agent(env, config, Random.default_rng())

            train_loop!(env, agent, config, progress)
            test_loop!(env, agent, config, progress)
        end

        #=
        Log:
        - total_rews: the return per episode
        - losses: The average loss for an episode
        - total_step: the number of steps per episode
        =#
        


        # Test agent:
        # num_test_episodes = get(config, "num_test_episodes", 0)



        save_results = ExpUtils.prep_save_results(data, get(config, "save_extras", []))
        (;save_results = save_results, data=data)
        # save_results = (;logger.data..., test_total_rews=trv, test_total_steps=tsv)

        # (;save_results = save_results)
    end
end


function train_loop!(env, agent, config, progress)

    num_steps = config["steps"]
    
    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    get_showvalues(eps, usa, a) = () -> begin
        [(:episode, eps),
         (:loss, usa[:avg_loss]),
         (:action, a.action),
         (:preds, a.preds)]
    end

    #=
    Start of _actual_ experiment. 
    =#
    cur_total_steps = 0
    eps = 1
    while cur_total_steps <= num_steps
        usa = UpdateStateAnalysis(
            (loss = 0.0f0, avg_loss = 1.0f0),
            Dict(
                :loss => (s, us) -> s + us.loss,
                :avg_loss => (s, us) -> 0.99 * s + 0.01 * us.loss
            ))

        max_episode_steps = min(
            max(num_steps - cur_total_steps, 1),
            10000)

        total_rews, total_steps = run_episode!(env, agent, max_episode_steps) do (s, a, s′, r)
            if progress
                next!(prg_bar, showvalues=get_showvalues(eps, usa, a))
            end
            if !(a.update_state isa Nothing)
                usa(a.update_state)
            end

        end

        #=
        Log data
        =#
        @data EXP total_rews
        @data EXP total_steps
        @data EXPExtra loss=usa[:loss]
        @data EXPExtra avg_loss=usa[:avg_loss]

        eps += 1
        cur_total_steps += total_steps
    end
end

function test_loop!(env, agent, config, progress)
    
    ActionRNNs.turn_off_training!(agent)
    
    test_states = [(x, y) for x in 1:config["width"], y in 1:config["height"]]
    num_test_episodes = length(test_states)
    prg_bar = ProgressMeter.Progress(num_test_episodes, "Episode: ")
    
    for eps in 1:num_test_episodes
        ts = 0
        tr = 0.0f0
        s = MinimalRLCore.start!(env, test_states[eps])
        agent_ret = MinimalRLCore.start!(agent, s)
        
        t = false
        while (ts < 10000) && !t

            s′, r, t = if agent_ret isa NamedTuple
                MinimalRLCore.step!(env, agent_ret.action)
            else
                MinimalRLCore.step!(env, agent_ret)
            end

            if t
                agent_ret = MinimalRLCore.step!(agent, s′, r, t)
            else
                agent_ret = MinimalRLCore.end!(agent, s′, r)
            end                
            tr += r
            ts += 1
        end

        if progress
            next!(prg_bar)
        end

        @data EXP test_total_rews=tr
        @data EXP test_total_steps=ts
    end
end


end
