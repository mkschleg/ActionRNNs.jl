module Torus2dERExperiment


import Flux
import JLD2

import MinimalRLCore: MinimalRLCore, run_episode!, get_actions
import ActionRNNs: ActionRNNs, DirectionalTMaze, ExpUtils

import .ExpUtils: experiment_wrapper, TMazeUtils, FluxUtils
import .ExpUtils: SimpleLogger, UpdateStateAnalysis, l1_grad

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

const TMU = TMazeUtils
const FLU = FluxUtils


"""
    default_config()

The default config for torus2d.
- TODO: test this config when experiment is up and running.
"""
function default_config()
    Dict{String,Any}(
        "save_dir" => "tmp/torus2d",

        #= MD
        # Experiment details.
        ------------------
        =#
        "seed" => 2, # seed of RNG
        "steps" => 150000, # number of total steps in experiment.
        
        #= MD
        Environment details
        -------------------
        Torus2d environment has: 
        width, height, anchors, 
        non_euclidean,
        fix_goal, goal_idx as possible inputs.
        =#
        "width" => 5,
        "height" => 5,
        "num_anchors" => 5,
        "non_euclidean" =>  true,
        "fix_goal" => false,
        "goal_idx" => 1,
        

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
        "numhidden" => 10,

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
        "gamma"=>0.99,
        "truncation" => 12,
        "hs_strategy" => "minimize")

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
    # Flux.Chain(
    #     Flux.Dense(in, internal_o, Flux.relu, initW=init_func)
    # )

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
    @show fs
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
    ActionRNNs.Torus2d(
        config["width"],
        config["height"],
        config["num_anchors"];
        po=true,
        non_euclidean=config["non_euclidean"],
        fix_goal=config["fix_goal"],
        goal_idx=config["goal_idx"]
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
            (:total_rews, :losses, :total_steps, :l1),
            (Float32, Float32, Int, Float32),
            Dict(
                :total_rews => (rew, steps, usa) -> rew,
                :losses => (rew, steps, usa) -> usa[:loss]/steps,
                :total_steps => (rew, steps, usa) -> steps,
                :l1 => (rew, steps, usa) -> usa[:l1]/steps
            )
        )

        mean_loss = 1.0f0
        
        prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
        eps = 1
        while sum(logger.data.total_steps) <= num_steps
            usa = UpdateStateAnalysis(
                (l1 = 0.0f0, loss = 0.0f0, avg_loss = 1.0f0),
                Dict(
                    :l1 => l1_grad,
                    :loss => (s, us) -> s + us.loss,
                    :avg_loss => (s, us) -> 0.99 * s + 0.01 * us.loss
                ))

            max_episode_steps = min(max((num_steps - sum(logger[:total_steps])), 2), 10000)
            n = 0
            total_rew, steps = run_episode!(env, agent, max_episode_steps) do (s, a, s′, r)
                if progress
                    next!(prg_bar, showvalues=[(:episode, eps),
                                               (:loss, usa[:avg_loss]),
                                               (:l1, usa[:l1]/n),
                                               (:action, a.action),
                                               (:preds, a.preds)])
                end
                if !(a.update_state isa Nothing)
                    usa(a.update_state)
                end
                n+=1
            end
            logger(total_rew, steps, usa)
            eps += 1
        end
        save_results = logger.data
        (;save_results = save_results)
    end
end



end
