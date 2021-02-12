module TMazeERExperiment

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
        "eta" => 0.001,
        "rho" =>0.9,

        "replay_size"=>10000,
        "warm_up" => 1000,
        "batch_size"=>32,
        
        "truncation" => 10,

        "synopsis" => false,

        "gamma"=>0.99)

end

function construct_agent(env, parsed, rng)

    fc = if parsed["cell"] ∈ ["FacARNN", "ARNN"]
        TMU.OneHotFeatureCreator{false}()
    else
        TMU.OneHotFeatureCreator{true}()
    end
    fs = MinimalRLCore.feature_size(fc)

    γ = Float32(parsed["gamma"])
    τ = parsed["truncation"]
    

    ap = ActionRNNs.ϵGreedy(0.1, MinimalRLCore.get_actions(env))
    init_func = (dims...)->ActionRNNs.glorot_uniform(rng, dims...)

    opt = FLU.get_optimizer(parsed)
    
    chain = begin
        if parsed["cell"] == "FacARNN"
            throw("You know this doesn't work yet...")
            Flux.Chain(ActionRNNs.FacARNN(fs, 2, parsed["numhidden"], parsed["factors"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
        elseif parsed["cell"] == "ARNN"
            if false#parsed["action_embedding"]
                Flux.Chain(
                    (x)->(Flux.onehot(x[1], 1:4), x[2]),
                    ActionRNNs.ActionStateStreams(Flux.Dense(4, parsed["ae_size"], tanh; initW=init_func), identity),
                    ActionRNNs.ARNN(fs, parsed["ae_size"], parsed["numhidden"]; init=init_func),
                    Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
            else
                Flux.Chain(
                    # (x)->(Flux.onehot(x[1], 1:4), x[2]),
                    # ActionRNNs.ActionStateStreams(Flux.Dense(4, 3, Flux.relu; initW=init_func), identity),
                    ActionRNNs.ARNN(fs, 4, parsed["numhidden"]; init=init_func),
                    Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
            end
        else
            rnntype = getproperty(Flux, Symbol(parsed["cell"]))
            if false#parsed["action_embedding"]
                Flux.Chain(
                    (x)->(x[1:3], x[4:end]),
                    ActionRNNs.ActionStateStreams(Flux.param, Flux.Dense(4, parsed["ae_size"], Flux.sigmoid; initW=init_func)),
                    (x)->vcat(x[1], x[2]),
                    rnntype(parsed["ae_size"] + fs, parsed["numhidden"]; init=init_func),
                    Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
            else
                Flux.Chain(rnntype(fs, parsed["numhidden"]; init=init_func),
                           Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
            end
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
                              ap)
end

function main_experiment(args::Vector{String}; kwargs...)
    as = arg_parse()
    parsed = parse_args(args, as)
    main_experiment(parsed; kwargs...)
end

function main_experiment(parsed::Dict; working=false, progress=false, verbose=false)

    working = get!(parsed, "working", working)
    progress = get!(parsed, "progress", working)
    verbose = get!(parsed, "verbose", working)
    
    savefile = ActionRNNs.save_setup(parsed)
    if isnothing(savefile)
        return
    end

    num_steps = parsed["steps"]

    seed = parsed["seed"]
    rng = Random.MersenneTwister(seed)

    env = TMaze(parsed["size"])
    agent = construct_agent(env, parsed, rng)

    total_rews = Float32[]
    successes = Bool[]
    total_steps = Int[]

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    eps = 1
    while sum(total_steps) <= num_steps
        success = false
        total_rew, steps =
            run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
                if parsed["progress"]
                    pr_suc = if length(successes) <= 1000
                        mean(successes)
                    else
                        mean(successes[end-1000:end])
                    end
                        
                    next!(prg_bar, showvalues=[(:episode, eps), (:successes, pr_suc)])
                end
                success = success || (r == 4.0)
            end
            # episode!(env, agent, rng, num_steps, sum(total_steps), parsed["progress"] ? prg_bar : nothing, nothing, eps)
        push!(total_rews, total_rew)
        push!(total_steps, steps)
        push!(successes, success)
        eps += 1
    end
    save_results = Dict("total_rews"=>total_rews, "steps"=>total_steps, "successes"=>successes)
    ActionRNNs.save_results(parsed, savefile, save_results)
end


end
