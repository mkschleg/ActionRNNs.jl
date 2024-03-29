module VariableTMazeERExperiment

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

    chain = begin
        if parsed["cell"] == "FacARNN"
            # throw("You know this doesn't work yet...")
            Flux.Chain(ActionRNNs.FacARNN(fs, 4, parsed["numhidden"], parsed["factors"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
        elseif parsed["cell"] == "ARNN"
            Flux.Chain(
                ActionRNNs.ARNN(fs, 4, parsed["numhidden"]; init=init_func, hs_learnable=parsed["hs_learnable"]),
                Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
        elseif parsed["cell"] == "RNN"
            Flux.Chain(
                ActionRNNs.RNN(fs, parsed["numhidden"]; init=init_func, hs_learnable=parsed["hs_learnable"]),
                Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
        else
            rnntype = getproperty(Flux, Symbol(parsed["cell"]))
            Flux.Chain(rnntype(fs, parsed["numhidden"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
            # end
        end
    end

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

function main_experiment(parsed::Dict = default_config(); working=false, progress=false, verbose=false)


    ActionRNNs.experiment_wrapper(parsed, working) do parsed

        num_steps = parsed["steps"]

        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = ActionRNNs.VariableTMaze(parsed["size"])
        agent = construct_agent(env, parsed, rng)


        logger = ActionRNNs.SimpleLogger(
            (:total_rews, :losses, :successes, :total_steps, :l1),
            (Float32, Float32, Bool, Int, Float32),
            Dict(
                :total_rews => (;rew, kwargs...) -> rew,
                :losses => (;steps, usa, kwargs...) -> usa[:loss]/steps,
                :successes => (;success, kwargs...) -> success,
                :total_steps => (;steps, kwargs...) -> steps,
                :l1 => (;usa, steps, kwargs...) -> usa[:l1]/steps
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
            logger(rew=total_rew, steps=steps, success=success, usa=usa)
            eps += 1
        end
        save_results = logger.data
        (;save_results = save_results)
    end
end



end
