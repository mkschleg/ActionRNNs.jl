module MaskedGridWorldERExperiment

import Flux
import JLD2

import MinimalRLCore: MinimalRLCore, run_episode!, get_actions
import ActionRNNs: ActionRNNs, DirectionalTMaze, ExpUtils

import .ExpUtils: experiment_wrapper, TMazeUtils, FluxUtils, Macros
import .ExpUtils: SimpleLogger, UpdateStateAnalysis, l1_grad
import .Macros: @generate_config_funcs

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random


const FLU = FluxUtils


"""
    default_config()

The default config for torus2d.
- TODO: test this config when experiment is up and running.
"""

@generate_config_funcs begin
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
    "numhidden" => 10

    info"""
            ### Optimizer details
            Flux optimizers are used. See flux documentation and `ExpUtils.Flux.get_optimizer` for details.
            - `opt::String`: The name of the optimizer used
            - Parameters defined by the particular optimizer.
            """
    "opt" => "RMSProp",
    "eta" => 0.0005,
    "rho" =>0.99,

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
            - `gamma::Float`: the discount for learning update.
            - `batch_size::Int`: size of batch
            - `truncation::Int`: Length of sequences used for training.
            - `update_wait::Int`: Time between updates (counted in agent interactions)
            - `target_update_wait::Int`: Time between target network updates (counted in agent interactions)
            - `hs_strategy::String`: Strategy for dealing w/ hidden state in buffer.
        """
    "lupdate_agg" => "SUM"
    "gamma"=>0.99    
    "batch_size"=>8
    "truncation" => 12
    
    "update_wait"=>4
    "target_update_wait"=>1000

    "hs_strategy" => "minimize"


end

function build_ann(in, actions::Int, config, rng=Random.GLOBAL_RNG)
    
    nh = config["numhidden"]
    init_func, initb = ActionRNNs.get_init_funcs(rng)

    deep_action = get(config, "deepaction", false)
    
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
         ActionRNNs.build_rnn_layer(internal_o, internal_a, out, config, rng))
    else
        (ActionRNNs.build_rnn_layer(in, actions, nh, config, rng),)
    end # if deep_action

    Flux.Chain(rnn...,
               Flux.Dense(nh, actions; initW=init_func))
    
end

# function get_ann_size(;kwargs...)
#     config = default_config()
#     for (k, v) in kwargs
#         config[string(k)] = v
#     end
#     env = construct_env(config, Random.GLOBAL_RNG)
#     agent = construct_agent(env, config, Random.GLOBAL_RNG)
#     sum(length, Flux.params(agent.model))
# end



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
                         ActionRNNs.HSMinimize())
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
                               config["num_goals"],
                               rng;
                               obs_strategy=:aliased,
                               pacman_wrapping=true)
end

Macros.@generate_ann_size_helper
Macros.@generate_working_function

function main_experiment(config = default_config(); progress=false, testing=false, overwrite=false)

    experiment_wrapper(config; use_git_info=false, testing=testing, overwrite=overwrite) do config

        num_steps = config["steps"]

        # Initialize the RNG seed
        seed = config["seed"]
        # rng = Random.MersenneTwister(seed)
        Random.seed!(seed)
        
        env = construct_env(config, Random.GLOBAL_RNG)
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

        prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
        get_showvalues(eps, logger, usa, a, n) = () -> begin
            [(:episode, eps),
             (:loss, usa[:avg_loss]),
             (:l1, usa[:l1]/n),
             (:action, a.action),
             (:preds, a.preds)]
        end

        #=
        Start of _actual_ experiment. 
        =#
        eps = 1
        while sum(logger.data.total_steps) <= num_steps
            usa = UpdateStateAnalysis(
                (l1 = 0.0f0, loss = 0.0f0, avg_loss = 1.0f0),
                Dict(
                    :l1 => l1_grad,
                    :loss => (s, us) -> s + us.loss,
                    :avg_loss => (s, us) -> 0.99 * s + 0.01 * us.loss
                ))

            max_episode_steps = min(
                max(num_steps - sum(logger[:total_steps]),
                    1),
                10000)
            n = 0
            total_rew, steps = run_episode!(env, agent, max_episode_steps) do (s, a, s′, r)
                if progress
                    next!(prg_bar, showvalues=get_showvalues(eps, logger, usa, a, n))
                end
                if !(a.update_state isa Nothing)
                    usa(a.update_state)
                end
                n+=1
            end

            #=
            Log data
            =#
            logger(total_rew, steps, usa)
            eps += 1
        end
        
        save_results = logger.data
        (;save_results = save_results)
    end
end





end
