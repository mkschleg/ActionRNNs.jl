module TMazeERExperiment

# include("../src/ActionRNNs.jl")

import Flux
import JLD2
import LinearAlgebra.Diagonal
import MinimalRLCore
using MinimalRLCore: run_episode!, get_actions
import ActionRNNs

using ActionRNNs: TMaze

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

const TMU = ActionRNNs.TMazeUtils
const FLU = ActionRNNs.FluxUtils

function default_config()
    Dict{String,Any}(
        "save_dir" => "tmaze",

        "seed" => 1,
        "steps" => 200000,
        "size" => 6,

        "cell" => "ARNN",
        "numhidden" => 6,

        "opt" => "RMSProp",
        "eta" => 0.0005,
        "rho" =>0.99,

        "replay_size"=>10000,
        "warm_up" => 1000,
        "batch_size"=>4,
        "update_wait"=>4,
        "truncation" => 8,

        "hs_learnable" => true,
        
        "gamma"=>0.99)

    
end

function get_ann(parsed, fs, env)

    init_func = (dims...)->Flux.glorot_uniform(dims...)
    
    if parsed["cell"] == "FacARNN"
        factors = parsed["factors"]
        nh = parsed["numhidden"]
        Flux.Chain(ActionRNNs.FacARNN(fs, 4, nh, factors; init=init_func),
                   Flux.Dense(nh, length(get_actions(env)); initW=init_func))
    elseif parsed["cell"] == "ARNN"
        nh = parsed["numhidden"]
        Flux.Chain(
            ActionRNNs.ARNN(fs, 4, nh;
                            init=init_func,
                            hs_learnable=parsed["hs_learnable"]),
            Flux.Dense(nh, length(get_actions(env)); initW=init_func))
    elseif parsed["cell"] == "RNN"
        nh = parsed["numhidden"]
        Flux.Chain(
            Flux.RNN(fs, nh;
                     init=init_func, initb=init_func),
                           # hs_learnable=parsed["hs_learnable"]),
            Flux.Dense(nh, length(get_actions(env)); initW=init_func))
    else
        nh = parsed["numhidden"]
        rnntype = getproperty(Flux, Symbol(parsed["cell"]))
        Flux.Chain(rnntype(fs, nh; init=init_func),
                   Flux.Dense(nh,
                              length(get_actions(env));
                              initW=init_func))
    end
end

function construct_agent(env, parsed, rng)

    fc = if parsed["cell"] ∈ ["FacARNN", "ARNN"]
        TMU.StandardFeatureCreator{false}()
    else
        TMU.StandardFeatureCreator{true}()
    end
    fs = MinimalRLCore.feature_size(fc)

    γ = Float32(parsed["gamma"])
    τ = parsed["truncation"]


    ap = ActionRNNs.ϵGreedy(0.1, MinimalRLCore.get_actions(env))
    init_func = (dims...)->ActionRNNs.glorot_uniform(rng, dims...)

    opt = FLU.get_optimizer(parsed)

    chain = get_ann(parsed, fs, env)

    ActionRNNs.ControlERAgent(chain,
                              opt,
                              τ,
                              γ,
                              fc,
                              fs,
                              3,
                              parsed["replay_size"],
                              parsed["warm_up"],
                              parsed["batch_size"],
                              parsed["update_wait"],
                              ap,
                              parsed["hs_learnable"])
end

function main_experiment(parsed::Dict; working=false, progress=false, verbose=false)


    ActionRNNs.experiment_wrapper(parsed, working) do parsed

        num_steps = parsed["steps"]

        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = TMaze(parsed["size"])
        agent = construct_agent(env, parsed, rng)


        logger = ActionRNNs.SimpleLogger(
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
            usa = ActionRNNs.UpdateStateAnalysis(
                (l1 = 0.0f0, loss = 0.0f0, avg_loss = 1.0f0),
                Dict(
                    :l1 => ActionRNNs.l1_grad,
                    :loss => (s, us) -> s + us.loss,
                    :avg_loss => (s, us) -> 0.99 * s + 0.01 * us.loss
                ))
            success = false
            max_episode_steps = min(max((num_steps - sum(logger[:total_steps])), 2), 10000)
            n = 0
            total_rew, steps = run_episode!(env, agent, max_episode_steps, rng) do (s, a, s′, r)
                if progress
                    pr_suc = if length(logger.data.successes) <= 1000
                        mean(logger.data.successes)
                    else
                        mean(logger.data.successes[end-1000:end])
                    end
                    next!(prg_bar, showvalues=[(:episode, eps),
                                               (:successes, pr_suc),
                                               (:loss, usa[:avg_loss]),
                                               (:l1, usa[:l1]/n),
                                               (:action, a.action)])
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
