module LinkedChainsERExperiment


import Flux
import JLD2

import MinimalRLCore: MinimalRLCore, run_episode!, get_actions
import ActionRNNs: ActionRNNs, ExpUtils

import .ExpUtils: experiment_wrapper, TMazeUtils, FluxUtils
import .ExpUtils: SimpleLogger, UpdateStateAnalysis, l1_grad
import .ExpUtils: Macros
import .Macros: @help_str, @generate_config_funcs

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

const TMU = TMazeUtils
const FLU = FluxUtils

@generate_config_funcs begin
    "save_dir" => "tmp/torus2d"

    help"""
    Experiment details.
    --------------------
    - `seed::Int`: seed of RNG
    - `steps::Int`: Number of steps taken in the experiment
    """
    "seed" => 2 # seed of RNG
    "steps" => 150000 # number of total steps in experiment.
    
    help"""
    Environment details
    -------------------
    This experiment uses the LinkedChain environment. The usable args are:
    - `chain_sizes::Vector{Int}`: the size of the chains linked together (as an array)
    - `time_to_fork::Int`: the number of states between the linking state and the fork.
    """
    "chain_sizes"=>[5, 10]
    "time_to_fork"=>0

    help"""
    agent details
    -------------
    ### RNN
    The RNN used for this experiment and its total hidden size, 
    as well as a flag to use (or not use) zhu's deep 
    action network.
    - `cell::String`: The typeof cell. Many types are possible.
    - `deepaction::Bool`: Whether to use Zhu et. al.'s deep action 4 RNNs idea.
    - `numhidden::Int`:  Size of hidden state in RNNs.
    """
    "cell" => "MARNN"
    "deepaction" => false
    "numhidden" => 10

    help"""
    ### Optimizer details
    Flux optimizers are used. See flux documentation and ExpUtils.Flux for details.
    - `opt::String`: The name of the optimizer used
    - Parameters defined by the particular optimizer.
    """
    "opt" => "RMSProp"
    "eta" => 0.0005
    "rho" => 0.99

    help"""
    ### Learning update and replay details including:
    - Replay: 
        - `replay_size::Int`: How many transitions are stored in the replay.
        - `warm_up::Int`: How many steps for warm-up (i.e. before learning begins).
    - Update details: 
        - `lupdate::String`: Learning update name
        - `gamma::Float`: the discount for learning update.
        - `batch_size::Int`: size of batch
        - `update_wait::Int`: Time between updates (counted in agent interactions)
        - `target_update_wait::Int`: Time between target network updates (counted in agent interactions)
        - `truncation::Int`: Length of sequences used for training.
        - `hs_strategy::String`: Strategy for dealing w/ hidden state in buffer.
    """
    "replay_size"=>20000
    "warm_up" => 1000
    
    "lupdate" => "QLearning"
    "gamma"=>0.99    
    "batch_size"=>8
    "update_wait"=>4
    "target_update_wait"=>1000
    
    "truncation" => 12
    "hs_strategy" => "minimize"
end


function build_deep_action_rnn_layers(in, actions, out, config, rng=Random.GLOBAL_RNG)


    # Deep actions for RNNs from Zhu et al 2018
    internal_a = config["internal_a"]
    internal_o = config["internal_o"]

    init_func, initb = ActionRNNs.get_init_funcs(rng)
    
    action_stream = Flux.Chain(
        (a)->Flux.onehotbatch(a, 1:actions),
        Flux.Dense(actions, internal_a, Flux.relu, initW=init_func),
    )
    obs_stream = identity

    (ActionRNNs.DualStreams(action_stream, obs_stream),
     ActionRNNs.build_rnn_layer(internal_o, internal_a, out, config, rng))
end

function build_ann(in, actions::Int, config, rng=Random.GLOBAL_RNG)
    
    nh = config["numhidden"]
    init_func, initb = ActionRNNs.get_init_funcs(rng)

    deep_action = get(config, "deepaction", false)
    rnn = if deep_action
        build_deep_action_rnn_layers(in, actions, nh, config, rng)
    else
        (ActionRNNs.build_rnn_layer(in, actions, nh, config, rng),)
    end

    Flux.Chain(rnn...,
               Flux.Dense(nh, actions; initW=init_func))
    
end


function construct_agent(env, config, rng)

    # fc = TMU.StandardFeatureCreator{false}()
    fc = ActionRNNs.IdentityFeatureCreator(ActionRNNs.obs_size(env))
    fs = MinimalRLCore.feature_size(fc)

    num_actions = length(get_actions(env))

    ap = ActionRNNs.ϵGreedy(0.1, MinimalRLCore.get_actions(env))
    # ap = ActionRNNs.construct_policy(config, MinimalRLCore.get_actions(env))
    
    # γ = Float32(config["gamma"])
    # lu = ActionRNNs.construct_control_update(config)
    lu = ActionRNNs.QLearningSUM(Float32(config["gamma"]))
    τ = config["truncation"]
    
    opt = FLU.get_optimizer(config)
    chain = build_ann(fs, num_actions, config, rng)

    ActionRNNs.DRQNAgent(chain,
                         opt,
                         τ,
                         lu, # γ,
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
                         ActionRNNs.HSMinimize())
end

function construct_env(config)
    # 
    ActionRNNs.LinkedChains{:CONT}(
        config["time_to_fork"],
        config["chain_sizes"]...
    )
end

function main_experiment(config = default_config(); progress=false, testing=false, overwrite=false)

    experiment_wrapper(config; use_git_info=false, testing=testing, overwrite=overwrite) do config

        num_steps = config["steps"]

        # Initialize the RNG seed
        seed = config["seed"]
        # rng = Random.MersenneTwister(seed)
        Random.seed!(seed)
        
        env = construct_env(config)
        agent = construct_agent(env, config, Random.GLOBAL_RNG)
        
        logger = SimpleLogger(
            (:step_to_link, ),
            (Int, ),
            Dict(
                :step_to_link => (num_steps) -> num_steps,
            )
        )

        mean_loss = 1.0f0
        
        prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
        eps = 1

        # while sum(logger.data.total_steps) <= num_steps
        usa = UpdateStateAnalysis(
            (l1 = 0.0f0, loss = 0.0f0, avg_loss = 1.0f0),
            Dict(
                :l1 => l1_grad,
                :loss => (s, us) -> s + us.loss,
                :avg_loss => (s, us) -> 0.99 * s + 0.01 * us.loss
            ))

        # max_episode_steps = min(max((num_steps - sum(logger[:total_steps])), 2), num_steps)
        n = 0
        step_counter = 0
        total_rew, steps = run_episode!(env, agent, num_steps) do (s, a, s′, r)
            step_counter += 1
            if progress
                next!(prg_bar, showvalues=[(:ns, step_counter),
                                           (:action, a.action),
                                           (:preds, a.preds)])
            end
            if !(a.update_state isa Nothing)
                usa(a.update_state)
            end

            if env.state == ActionRNNs.LinkedChainsConst.LINKING_STATE
                logger(step_counter)
                step_counter = 0
            end
            n+=1
        end
        # logger(total_rew, steps, usa)


        save_results = logger.data
        (;save_results = save_results)
    end
end


function working_experiment()
    args = Dict{String,Any}(
        "save_dir" => "tmp/linkedlist",

        #= MD
        # Experiment details.
        ------------------
        =#
        "seed" => 2, # seed of RNG
        "steps" => 150000, # number of total steps in experiment.
        
        #= MD
        Environment details
        -------------------
        =#
        "chain_sizes"=>[5, 10],
        

        #= MD
        agent details
        -------------
        =#

        #= MD
        The RNN used for this experiment and its total hidden size, 
        as well as a flag to use (or not use) zhu's deep 
        action network.
        =#
        "cell" => "MARNN",
        "deepaction" => false,
        "numhidden" => 5,

        #= MD
        optimizer details
        =#
        "opt" => "RMSProp",
        "eta" => 0.0005,
        "rho" =>0.99,

        #= MD
        Learning update details including:
        - ER: replay_size, warm_up
        - LU: learning_update, batch_size, update_wait
        - truncation is used by the agent to decide how long the temporal dependency is.
        - hs_learnable: determines whether the hidden state is learned. Change to a mode?
        =#
        "replay_size"=>20000,
        "warm_up" => 1000,
        "batch_size"=>8,
        "update_wait"=>4,
        "target_update_wait"=>1000,
        
        "lupdate" => "QLearning",
        "gamma"=>0.9,
        "truncation" => 20,
        "hs_strategy" => "minimize")

    main_experiment(args, progress=true)
    
end


end
