module MaskedGridWorldRandAgent

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


end



function construct_agent(env, config, rng)

    agent = ActionRNNs.RandomAgent(MinimalRLCore.get_actions(env))
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
            (:total_rews, :total_steps),
            (Float32, Int),
            Dict(
                :total_rews => (rew, steps) -> rew,
                :total_steps => (rew, steps) -> steps,
            )
        )

        prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
        get_showvalues(eps, a) = () -> begin
            [(:episode, eps),
             (:action, a)]
        end

        #=
        Start of _actual_ experiment. 
        =#
        eps = 1
        while sum(logger.data.total_steps) <= num_steps

            max_episode_steps = min(
                max(num_steps - sum(logger[:total_steps]),
                    1),
                10000)
            n = 0
            total_rew, steps = run_episode!(env, agent, max_episode_steps) do (s, a, s′, r)
                if progress
                    next!(prg_bar, showvalues=get_showvalues(eps, a))
                end
                n+=1
            end

            #=
            Log data
            =#
            logger(total_rew, steps)
            eps += 1
        end
        
        save_results = logger.data
        (;save_results = save_results)
    end
end





end
