
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
                throw("You know this doesn't work yet...")
                Flux.Chain(ActionRNNs.FacARNN(fs, 2, parsed["numhidden"], parsed["factors"]; init=init_func),
                           Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
            elseif parsed["cell"] == "ARNN"
                Flux.Chain(
                    ActionRNNs.ARNN(fs, 4, parsed["numhidden"]; init=init_func, islearnable=parsed["hs_learnable"]),
                    Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
            elseif parsed["cell"] == "RNN"
                Flux.Chain(
                    ActionRNNs.RNN(fs, parsed["numhidden"]; init=init_func, islearnable=parsed["hs_learnable"]),
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
        
        total_rews = Float32[]
        successes = Bool[]
        losses = Float32[]
        l1_grads = Float32[]
        total_steps = Int[]
        
        mv_avg_losses = 1.0f0
        
        prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
        eps = 1
        while sum(total_steps) <= num_steps
            success = false
            ℒ = 0.0f0
            l1_grad = 0.0f0
            total_rew, steps =
                run_episode!(env, agent, min(max((num_steps - sum(total_steps)), 2), 10000), rng) do (s, a, s′, r)
                    success = success || (r == 4.0)
                    ℒ += a.loss
                    l1_grad += a.l1_grads
                    mv_avg_losses = 0.9*mv_avg_losses + 0.1*ℒ
                    
                    if progress
                        pr_suc = if length(successes) <= 1000
                            mean(successes)
                        else
                            mean(successes[end-1000:end])
                        end
                        next!(prg_bar, showvalues=[(:episode, eps), (:successes, pr_suc), (:loss, mv_avg_losses)])
                    end
                end
            push!(total_rews, total_rew)
            push!(total_steps, steps)
            push!(successes, success)
            push!(losses, ℒ \ steps)
            push!(l1_grads, l1_grad \ steps)
            eps += 1
        end
        save_results = Dict("total_rews"=>total_rews,
                            "steps"=>total_steps,
                            "successes"=>successes,
                            "losses"=>losses,
                            "l1_grads" => l1_grads)
        
        (agent = agent, save_results = save_results)
    end
end


end
