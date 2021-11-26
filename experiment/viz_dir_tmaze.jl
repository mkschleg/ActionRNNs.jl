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
        "save_dir" => "tmaze",

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

function get_ann(parsed, image_dims, env, rng)

    nh = parsed["numhidden"]
    na = 3#length(get_actions(env))
    init_func = (dims...)->ActionRNNs.glorot_uniform(rng, dims...)

    
    cl = Flux.Conv((4, 4), 1 => 4, Flux.relu; stride=2, init=init_func)
    fs = prod(Flux.outdims(cl, image_dims))
    latent_size = parsed["latent_size"]
    output_size = parsed["output_size"]
    
    if parsed["cell"] ∈ ActionRNNs.fac_rnn_types()

        rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
        factors = parsed["factors"]
        init_style = get(parsed, "init_style", "standard")
        
        init_func = (dims...; kwargs...)->
            ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
        initb = (dims...; kwargs...) -> Flux.zeros(dims...)
        
        Flux.Chain(cl, Flux.flatten, Flux.Dense(fs, latent_size, Flux.relu; initW=init_func),
                   rnn(latent_size, na, nh, factors;
                       init_style=init_style,
                       init=init_func,
                       initb=initb),
                   Flux.Dense(nh, na; initW=init_func))
        
    elseif parsed["cell"] ∈ ActionRNNs.rnn_types()

        rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
        
        init_func = (dims...; kwargs...)->
            ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
        initb = (dims...; kwargs...) -> Flux.zeros(dims...)
        
        Flux.Chain(
            cl, Flux.flatten, Flux.Dense(fs, latent_size, Flux.relu; initW=init_func),
            Flux.Dense(latent_size, latent_size, Flux.relu; initW=init_func),
            rnn(latent_size, na, nh;
                init=init_func,
                initb=initb),
            Flux.Dense(nh, output_size, Flux.relu; initW=init_func),
            Flux.Dense(output_size, na; initW=init_func))

    else
        rnntype = getproperty(Flux, Symbol(parsed["cell"]))
        Flux.Chain(
            cl, Flux.flatten, Flux.Dense(fs, latent_size, Flux.relu; initW=init_func),
            rnntype(latent_size, nh; init=init_func),
            Flux.Dense(nh,
                       length(get_actions(env));
                       initW=init_func))
        
    end
end

function construct_agent(env, parsed, rng)

    # fc = TMU.StandardFeatureCreator{false}()
    # fs = MinimalRLCore.feature_size(fc)

    γ = Float32(parsed["gamma"])
    τ = parsed["truncation"]

    ap = ActionRNNs.ϵGreedy(0.1, MinimalRLCore.get_actions(env))
    # ap = ActionRNNs.ϵGreedyDecay((1.0, 0.05), 50000, 1000, MinimalRLCore.get_actions(env))

    opt = FLU.get_optimizer(parsed)

    chain = get_ann(parsed, (28, 28, 1, 1), env, rng) |> Flux.gpu

    ActionRNNs.ImageDRQNAgent(chain,
                         opt,
                         τ,
                         γ,
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

function main_experiment(parsed = default_config(); working=false, progress=false, verbose=false)

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
    
    experiment_wrapper(parsed, working) do parsed

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
