"""
    DirectionalTMazeLearnInterExperiment

An experiment to compare different RNN cells using the [`ActionRNNs.DirectionalTMaze`](@ref) environment.

Usage is detailed through the docs for 
- [`DirectionalTMazeInterventionExperiment.default_config`](@ref)
- [`DirectionalTMazeInterventionExperiment.main_experiment`](@ref)
- [`DirectionalTMazeInterventionExperiment.working_experiment`](@ref)
- [`DirectionalTMazeInterventionExperiment.construct_env`](@ref)
- [`DirectionalTMazeInterventionExperiment.construct_agent`](@ref)

"""
module DirectionalTMazeLearnInterExperiment

# import Flux
import JLD2
import ActionRNNs: ActionRNNs, DirectionalTMaze, ExpUtils, MinimalRLCore, Flux
import ActionRNNs: @data

import MinimalRLCore: run_episode!, get_actions
import .ExpUtils: Macros, TMazeUtils, FluxUtils
import .ExpUtils: SimpleLogger, UpdateStateAnalysis, l1_grad, construct_logger
import .ExpUtils: experiment_wrapper
import .Macros: @info_str, @generate_config_funcs

import Logging: with_logger

import StatsBase: percentile

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
    - Intervention details:
        - `inter_list::String`: Points to the constructor fot the list of interventions to run on the agent after training.
            - `"DTMazeV1"`: Testing the start intervention and middle intervention.
        - `inter_num_episodes::Int`: Number of episodes to perform interventions (these are after training episodes).
    """
    "inter_sequence"=>"DTMazeV1"
    "inter_num_episodes"=>1000


end


function build_ann(config, in, actions::Int, rng)
    
    nh = config["numhidden"]
    init_func, initb = ActionRNNs.get_init_funcs(rng)


    deep_action = "deep" ∈ keys(config) ? config["deep"] : get(config, "deepaction", false)

    rnn = if deep_action
        
        #=
        If we are using [zhu et al 2018]() style action embeddings
        
        DirectionalTMazeInterventionExperiment only re-embeds the action encoding. Here we encode the integer as a 
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

"""
    construct_agent

Construct the agent for `DirectionalTMazeInterventionExperiment`. See 
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
function construct_env(config, args...)
    DirectionalTMaze(config["size"])
end

Macros.@generate_ann_size_helper
Macros.@generate_working_function

"""
    main_experiment

Run an experiment from config. See [`DirectionalTMazeInterventionExperiment.working_experiment`](@ref) 
for details on running on the command line and [`DirectionalTMazeInterventionExperiment.default_config`](@ref) 
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

        extras = union(get(config, "log_extras", []), get(config, "save_extras", []))
        extra_proc = [c isa AbstractArray ? (Symbol(c[1]), Symbol(c[2])) : Symbol(c) for c in extras]

        env = construct_env(config)
        agent = construct_agent(env, config, rng)

        # throw away logger for training.
        _dta, _lgr = ChoosyDataLoggers.construct_logger([:NA])
        with_logger(_lgr) do
            train_loop!(env, agent, num_steps, _dta, rng, progress=progress)
        end

        inter_start_dir_list = get_intervention_list(config)

        data, logger = ExpUtils.construct_logger(extra_groups_and_names=extra_proc)

        with_logger(logger) do
            run_intervention_experiment(config, agent, env;
                                        rng=rng, inter_start_dir_list)
        end

        save_results = ExpUtils.prep_save_results(data, get(config, "save_extras", []))
        (;save_results = save_results, data=data)
    end
end


function train_loop!(env, agent, num_steps, data, rng; progress=false)

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    # generate_showvalues(eps, usa, a, n) = () -> begin
    #     pr_suc = if (:EXP ∈ keys(data)) && (:successes ∈ keys(data[:EXP]))
    #         if (length(data[:EXP][:successes]) <= 1000)
    #             mean(data[:EXP][:successes])
    #         else
    #             mean(data[:EXP][:successes][end-1000:end])
    #         end
    #     end
    #     [(:episode, eps),
    #      (:successes, pr_suc),
    #      (:loss, usa[:avg_loss]),
    #      (:l1, usa[:l1]/n),
    #      (:action, a.action),
    #      (:preds, a.preds)]
    # end
    
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

        max_episode_steps = min(max((num_steps - total_steps), 2), 10_000)
        n = 0
        
        total_rew, steps = run_episode!(env, agent, max_episode_steps, rng) do (s, a, s′, r)
            if progress
                next!(prg_bar)
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

struct NoIntervention end
reset!(::NoIntervention) = nothing
make_intervention(agent, env, ::NoIntervention, agent_ret) = 
    if agent_ret isa Int
	agent_ret
    else
	agent_ret.action
    end

mutable struct ActionInterventionUnderCondition
    action::Int
    condition::Function
    intervened::Bool
end

function reset!(intervention::ActionInterventionUnderCondition)
    intervention.intervened = false
end

function make_intervention(agent::ActionRNNs.DRQNAgent,
                           env::ActionRNNs.DirectionalTMaze,
                           intervention::ActionInterventionUnderCondition, 
                           agent_ret)

    if !intervention.intervened && intervention.condition(env)
	agent.action = intervention.action
	intervention.intervened = true
    end

    agent.action
end

mutable struct MultiActionInterventionUnderCondition
    action_seq::Vector{Int}
    condition::Function
    i::Int
    ing::Bool
    ed::Bool
    MultiActionInterventionUnderCondition(action_seq, condition) =
        new(action_seq, condition, 1, false, false)
end

function reset!(intervention::MultiActionInterventionUnderCondition)
    intervention.ed = false
    intervention.ing = false
end

function make_intervention(agent::ActionRNNs.DRQNAgent,
                           env::ActionRNNs.DirectionalTMaze,
                           intervention::MultiActionInterventionUnderCondition, 
                           agent_ret)

    if !intervention.ed && (intervention.condition(env) || intervention.ing)
	# agent.action = intervention.action
        if intervention.i <= length(intervention.action_seq)
            intervention.ing = true
            agent.action = intervention.action_seq[intervention.i]
            intervention.i += 1
        else
            intervention.ing = false
            intervention.ed = true
        end
    end

    agent.action
end

function get_intervention_list(config)
    get_intervention_list(Val(Symbol(config["inter_list"])))
end

function get_intervention_list(::Val{:DTMazeStep})
    inter_start_dir_list = [
	(NoIntervention(), 2),
	(MultiActionInterventionUnderCondition(
	    [2, 3, 3], (env)->env.state.x == 1), 2),
        (MultiActionInterventionUnderCondition(
	    [2, 3], (env)->env.state.x == 1), 2),
        (MultiActionInterventionUnderCondition(
	    [2], (env)->env.state.x == 1), 2),
        (MultiActionInterventionUnderCondition(
	    [2, 2, 3, 3], (env)->env.state.x == 1), 2),
        (MultiActionInterventionUnderCondition(
	    [2, 2, 3], (env)->env.state.x == 1), 2),
        (MultiActionInterventionUnderCondition(
	    [2, 2], (env)->env.state.x == 1), 2),
    ]
end

function get_intervention_list(::Val{:DTMazeStepSmall})
    inter_start_dir_list = [
	(NoIntervention(), 2),
	# (MultiActionInterventionUnderCondition(
	#     [2, 3, 3], (env)->env.state.x == 1), 2),
        # (MultiActionInterventionUnderCondition(
	#     [2, 3], (env)->env.state.x == 1), 2),
        (MultiActionInterventionUnderCondition(
	    [2], (env)->env.state.x == 1), 2),
        # (MultiActionInterventionUnderCondition(
	#     [2, 2, 3, 3], (env)->env.state.x == 1), 2),
        # (MultiActionInterventionUnderCondition(
	#     [2, 2, 3], (env)->env.state.x == 1), 2),
        (MultiActionInterventionUnderCondition(
	    [2, 2], (env)->env.state.x == 1), 2),
    ]
end

function get_intervention_list(::Val{:DTMazeStraight1})
    inter_start_dir_list = [
        (ActionInterventionUnderCondition(
	    2, (env)->env.state.x == 1, false), 2),
    ]
end

function get_intervention_list(::Val{:DTMazeStraight2})
    inter_start_dir_list = [
        (MultiActionInterventionUnderCondition(
	    [2, 2], (env)->env.state.x == 1), 2),
    ]
end


function run_intervention_experiment(config, agent, env;
                                     rng=Random.GLOBAL_RNG,
                                     inter_start_dir_list=[(NoIntervention(),rand(rng, 1:4))])
	
    for (inter, start_dir) in inter_start_dir_list
        if  "inter_num_episodes" ∈ keys(config)
            inter_num_episodes = config["inter_num_episodes"]
	    intervention_experiment(agent, env, inter, start_dir, inter_num_episodes,
                                    rng)
        elseif "inter_num_steps" ∈ keys(config)
            inter_num_steps = config["inter_num_steps"]
            intervention_experiment_by_steps(
                agent, env, inter, start_dir, inter_num_steps, rng)
        end
    end
end


function intervention_experiment_by_steps(agent,
                                          env::DirectionalTMaze,
                                          intervention,
                                          start_dir,
                                          inter_num_steps,
                                          rng)
    s = Bool[]
    ts = Int[]

    while sum(ts) < inter_num_steps
        max_episode_steps = min(10_000, max(inter_num_steps - sum(ts), 2)) # set in training loop
	success = false
	total_rew, steps = intervention_episode!(
	    env,
	    agent,
	    intervention,
	    start_dir isa Int ? start_dir : start_dir(rng),
	    max_episode_steps,
	    rng) do (s, a, s′, r)
		success = success || (r == 4.0)
	    end

        push!(s, success)
        push!(ts, steps)

    end
    
    @data EXP inter_successes=s
    @data EXP inter_total_steps=ts
end

function intervention_experiment(agent,
                                 env::DirectionalTMaze,
                                 intervention,
                                 start_dir,
                                 inter_num_episodes,
                                 rng)
    s = Bool[]
    ts = Int[]
    max_episode_steps = 10_000 # set in training loop
    for eps in 1:inter_num_episodes
	success = false
	total_rew, steps = intervention_episode!(
	    env,
	    agent,
	    intervention,
	    start_dir isa Int ? start_dir : start_dir(rng),
	    max_episode_steps,
	    rng) do (s, a, s′, r)
		success = success || (r == 4.0)
	    end

        push!(s, success)
        push!(ts, steps)

    end
    
    @data EXP inter_successes=s
    @data EXP inter_total_steps=ts
end


function intervention_episode!(f, 
                               env::DirectionalTMaze, 
                               agent, 
                               intervention, 
                               start_dir,
                               max_episode_steps, 
                               rng)
    
    terminal = false

    s = MinimalRLCore.start!(env, rand(rng, [1,3]), start_dir)
    ar = MinimalRLCore.start!(agent, s, rng)
    reset!(intervention)

    na = make_intervention(agent, env, intervention, ar)
    step = 0
    rews = 0.0f0
    while !terminal && step < max_episode_steps

	s′, r, terminal = MinimalRLCore.step!(env, na, rng)
	ar = MinimalRLCore.step!(agent, s′, r, terminal, rng)

	# na = _get_action(ar)
	na = make_intervention(agent, env, intervention, ar)
	
	f((s, na, s′, r, terminal))

        rews += r

	step += 1 
    end

    rews, step
end



end
