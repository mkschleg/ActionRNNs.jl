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
    "dynmode"=>"STRAIGHT"

    help"""
    agent details
    -------------
    ### RNN
    The RNN used for this experiment and its total hidden size, 
    as well as a flag to use (or not use) zhu's deep 
    action network. See [`build_rnn_layer`](@ref) for more details.
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
    
    "lupdate_agg" => "SUM"
    "gamma"=>0.99    
    "batch_size"=>8
    "update_wait"=>4
    "target_update_wait"=>1000
    
    "truncation" => 12
    "hs_strategy" => "minimize"
end

function build_ann(config, in, actions::Int, rng)
    
    nh = config["numhidden"]
    init_func, initb = ActionRNNs.get_init_funcs(rng)


    deep_action = config["deepaction"]

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
         ActionRNNs.build_rnn_layer(internal_o, internal_a, nh, config, rng))
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
    fc = ActionRNNs.IdentityFeatureCreator(ActionRNNs.obs_size(env))
    fs = MinimalRLCore.feature_size(fc)
    
    # fc = TMU.StandardFeatureCreator{false}()
    # fs = MinimalRLCore.feature_size(fc)
    num_actions = length(get_actions(env))

    #=
    Set policy to always be [`ϵGreedy`](@ref)
    =#
    
    ap = ActionRNNs.ϵGreedy(0.1, MinimalRLCore.get_actions(env))


    #=
    Set learning rate aggregations
    =#
    lupdate_agg = config["lupdate_agg"]
    lu = if lupdate_agg == "SUM"
        ActionRNNs.QLearningSUM(Float32(config["gamma"]))
    elseif lupdate_agg == "HUBER"
        ActionRNNs.QLearningHUBER(Float32(config["gamma"]))
    else
        @error "$(lupdate_agg) not supported with QLearning"
    end


    #= 
    Truncation value for BPTT.
    =#
    τ = config["truncation"]
    
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
    
    hs_strategy = ActionRNNs.get_hs_strategy(config["hs_strategy"])
     
    #=
    This looks scary, but isn't.
    =#

    ActionRNNs.DRQNAgent(chain,
                         opt,
                         τ,
                         lu,
                         fc,
                         fs,
                         ActionRNNs.obs_size(env),
                         config["replay_size"],
                         config["warm_up"],
                         config["batch_size"],
                         config["update_wait"],
                         config["target_update_wait"],
                         ap,
                         hs_strategy)
end


function construct_env(config)
    @show keys(config)
    ActionRNNs.LinkedChains(
        time_to_fork = config["time_to_fork"],
        sizes = config["chain_sizes"],
        termmode = :CONT,
        dynmode = Symbol(config["dynmode"])
    )
end

Macros.@generate_ann_size_helper
Macros.@generate_working_function

function main_experiment(config = default_config(); progress=false, testing=false, overwrite=false)

    experiment_wrapper(config; use_git_info=false, testing=testing, overwrite=overwrite) do config

        num_steps = config["steps"]

        # Initialize the RNG seed
        seed = config["seed"]
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


end
