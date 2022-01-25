module VisualDirectionalTMazeERExperiment


import MinimalRLCore: MinimalRLCore, run_episode!, get_actions
import ActionRNNs: ActionRNNs, ImageDirTMaze, ExpUtils

import .ExpUtils: SimpleLogger, UpdateStateAnalysis, l1_grad, experiment_wrapper
import .ExpUtils: TMazeUtils as TMU, FluxUtils as FLU

import Flux
import JLD2

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

#=
Time: 0:00:21
  episode:    6
  steps:      33
  successes:  0.4
  loss:       0.9570326
  l1:         0.06516068
  action:     3
  preds:      Float32[-1.4597893; -0.6299171; -1.4020246]
=#
function default_config()
    Dict{String,Any}(
        "save_dir" => "tmp/viz_dir_tmaze",

        "seed" => 1,
        "steps" => 2000,
        "size" => 10,


        "cell" => "MAGRU",
        "init_style" => "standard",
        "numhidden" => 64,
        "latent_size" => 128,
        "output_size" => 128,

        "opt" => "ADAM",
        "eta" => 0.00005,
        "rho" =>0.99,

        "replay_size"=>50000,
        "warm_up" => 1000,
        "batch_size"=>16,
        "update_wait"=>4,
        "target_update_wait"=>10000,
        "truncation" => 15,

 
        "hs_learnable" => true,
        
        "gamma"=>0.9)

    
end

function build_ann(in, actions::Int, parsed, rng)
    
    nh = parsed["numhidden"]
    init_func, initb = ActionRNNs.get_init_funcs(rng)


    deep_action = get(parsed, "deep", false)

    cl = Flux.Conv((4, 4), 1 => 4, Flux.relu; stride=2, init=init_func)
    fs = prod(Flux.outdims(cl, in))
    latent_size = parsed["latent_size"]
    output_size = parsed["output_size"]
    
    pre_layers = if deep_action

        internal_a = parsed["internal_a"]
        
        action_stream = Flux.Chain(
            (a)->Flux.onehotbatch(a, 1:actions),
            Flux.Dense(actions, internal_a, Flux.relu, initW=init_func),
        )
        
        obs_stream = Flux.Chain(cl,
                                Flux.flatten,
                                Flux.Dense(fs, latent_size, Flux.relu; initW=init_func),
                                Flux.Dense(latent_size, latent_size, Flux.relu; initW=init_func))
        
        (ActionRNNs.DualStreams(action_stream, obs_stream),
         ActionRNNs.build_rnn_layer(latent_size, internal_a, nh, parsed, rng))
    else
        (cl,
         Flux.flatten,
         Flux.Dense(fs, latent_size, Flux.relu; initW=init_func),
         Flux.Dense(latent_size, latent_size, Flux.relu; initW=init_func),
         ActionRNNs.build_rnn_layer(latent_size, actions, nh, parsed, rng))
    end
    
    Flux.Chain(
        pre_layers...,
        Flux.Dense(nh, output_size, Flux.relu; initW=init_func),
        Flux.Dense(output_size, actions; initW=init_func)
    )


end


function construct_agent(env, parsed, rng)

    γ = Float32(parsed["gamma"])
    τ = parsed["truncation"]

    ap = ActionRNNs.ϵGreedy(0.1, MinimalRLCore.get_actions(env))

    opt = FLU.get_optimizer(parsed)

    chain = build_ann((28, 28, 1, 1), length(get_actions(env)), parsed, rng) |> Flux.gpu

    ActionRNNs.ImageDRQNAgent(chain,
                         opt,
                         τ,
                         ActionRNNs.QLearningHUBER(γ),
                         (28, 28, 1),
                         UInt8,
                         parsed["replay_size"],
                         parsed["warm_up"],
                         parsed["batch_size"],
                         parsed["update_wait"],
                         parsed["target_update_wait"],
                         ap,
                         parsed["hs_learnable"])
end

function main_experiment(parsed = default_config(); progress=false, testing=false, overwrite=false)

    if "cell_numhidden" ∈ keys(parsed)
        parsed["cell"] = parsed["cell_numhidden"][1]
        parsed["numhidden"] = parsed["cell_numhidden"][2]
        delete!(parsed, "cell_numhidden")
    end

    if "numhidden_factors" ∈ keys(parsed)
        parsed["numhidden"] = parsed["numhidden_factors"][1]
        parsed["factors"] = parsed["numhidden_factors"][2]
        delete!(parsed, "numhidden_factors")
    end
    
    experiment_wrapper(parsed; use_git_info=false, testing=testing, overwrite=overwrite) do parsed

        num_steps = parsed["steps"]

        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = ImageDirTMaze(parsed["size"])
        agent = construct_agent(env, parsed, rng)

        
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
            success = false
            max_episode_steps = min(max((num_steps - sum(logger[:total_steps])), 2), 1000)
            n = 0
            total_rew, steps = run_episode!(env, agent, max_episode_steps, rng) do (s, a, s′, r)
                if progress
                    pr_suc = if length(logger.data.successes) <= 1000
                        mean(logger.data.successes)
                    else
                        mean(logger.data.successes[end-1000:end])
                    end
                    next!(prg_bar, showvalues=[(:episode, eps),
                                               (:steps, n),
                                               (:successes, pr_suc),
                                               (:loss, usa[:avg_loss]),
                                               (:l1, usa[:l1]/n),
                                               (:action, a.action),
                                               (:preds, a.preds)])
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
