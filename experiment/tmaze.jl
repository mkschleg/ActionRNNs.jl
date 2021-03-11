
module TMazeExperiment

# include("../src/ActionRNNs.jl")

import Flux
import JLD2
import LinearAlgebra.Diagonal
import MinimalRLCore
using MinimalRLCore: run_episode!, get_actions
import ActionRNNs

using ActionRNNs: TMaze


# using ActionRNNs
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

# using Plots

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
        "rho" => 0.99,
        "truncation" => 8,

        "hs_learnable" => true,

        "gamma"=>0.99)

end

function construct_agent(env, parsed, rng)

    num_actions = length(get_actions(env))
    is_actionrnn = parsed["cell"] ∈ ["FacARNN", "ARNN"]
    τ = parsed["truncation"]
    γ = parsed["gamma"]
    opt = FLU.get_optimizer(parsed)
    
    fc = if is_actionrnn
        # States without one hot action encoding.
        TMU.StandardFeatureCreator{false}()
    else
        # States with one hot action encoding
        TMU.StandardFeatureCreator{true}()
    end
    fs = MinimalRLCore.feature_size(fc)

    ap = ActionRNNs.ϵGreedy(0.1, MinimalRLCore.get_actions(env))
    init_func = (dims...)->ActionRNNs.glorot_uniform(rng, dims...)

    chain = begin
        if false
            # ARNN
            Flux.Chain(
                (x)->(Flux.onehot(x[1], 1:4), x[2]),
                ActionRNNs.ActionStateStreams(Flux.Dense(4, parsed["ae_size"], tanh; initW=init_func), identity),
                ActionRNNs.ARNN(fs, parsed["ae_size"], parsed["numhidden"]; init=init_func),
                Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))

            # generic RNN
            Flux.Chain(
                (x)->(x[1:3], x[4:end]),
                ActionRNNs.ActionStateStreams(Flux.param,
                                              Flux.Dense(4, parsed["ae_size"], Flux.sigmoid; initW=init_func)),
                (x)->vcat(x[1], x[2]),
                rnntype(parsed["ae_size"] + fs, parsed["numhidden"]; init=init_func),
                Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
        else
            if parsed["cell"] == "FacARNN"
                # throw("You know this doesn't work yet...")
                Flux.Chain(ActionRNNs.FacARNN(fs, 4, parsed["numhidden"], parsed["factors"]; hs_learnable=parsed["hs_learnable"], init=init_func),
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
            end
        end
    end

    ActionRNNs.ControlOnlineAgent(chain, opt, τ, γ, fc, fs, ap)
end

function main_experiment(parsed::Dict; working=false, progress=false)

    ActionRNNs.experiment_wrapper(parsed, working) do (parsed)

        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        num_steps = parsed["steps"]
        env = TMaze(parsed["size"])
        agent = construct_agent(env, parsed, rng)

        logger = ActionRNNs.SimpleLogger(
            (:total_rews, :losses, :successes, :total_steps),
            (Float32, Float32, Bool, Int),
            Dict(
                :total_rews => (rew, steps, success, usa) -> rew,
                :losses => (rew, steps, success, usa) -> usa[:loss]/steps,
                :successes => (rew, steps, success, usa) -> success,
                :total_steps => (rew, steps, success, usa) -> steps
            )
        )

        prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
        eps = 1
        while sum(logger[:total_steps]) <= num_steps
            success = false
            usa = ActionRNNs.UpdateStateAnalysis(
                (loss = 0.0f0, avg_loss = 1.0f0),
                Dict(
                    :loss => (s, us) -> s + us.loss,
                    :avg_loss => (s, us) -> 0.9 * s + 0.1 * us.loss
                ))
            max_episode_steps = min(max((num_steps - sum(logger[:total_steps])), 2), 10000)
            total_rew, steps = run_episode!(env, agent, max_episode_steps, rng) do (s, a, s′, r)
                success = success || (r == 4.0)
                usa(a.update_state)
                
                if progress
                    pr_suc = if length(logger[:successes]) <= 1000
                        mean(logger[:successes])
                    else
                        mean(logger[:successes][end-1000:end])
                    end
                    next!(prg_bar, showvalues=[(:episode, eps), (:successes, pr_suc), (:loss, usa[:avg_loss])])
                end
            end
            logger(total_rew, steps, success, usa)
            eps += 1
        end
        
        (agent = agent, save_results = logger.data)
    end
end


end
