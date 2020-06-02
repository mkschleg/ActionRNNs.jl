
module TMazeExperiment

# include("../src/ActionRNN.jl")

import Flux
import Flux.Tracker
import JLD2
import LinearAlgebra.Diagonal
import MinimalRLCore
using MinimalRLCore: run_episode!
import ActionRNN

using ActionRNN: TMaze, step!, start!, is_terminal, get_actions, glorot_uniform

# using ActionRNN
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

# using Plots

const TMU = ActionRNN.TMazeUtils
const FLU = ActionRNN.FluxUtils

function arg_parse(as::ArgParseSettings = ArgParseSettings(exc_handler=Reproduce.ArgParse.debug_handler))

    ActionRNN.exp_settings!(as)
    ActionRNN.env_settings!(as, TMaze)
    ActionRNN.agent_settings!(as, ActionRNN.FluxAgent)
    # RWU.horde_settings!(as, "out")

    @add_arg_table as begin
        "--factors"
        arg_type=Int
        default=0
        "--gamma"
        arg_type=Float32
        default=0.99f0
        "--action_embedding"
        action=:store_true
        "--ae_size"
        arg_type=Int
    end
    
    return as
end

function construct_agent(env, parsed, rng)

    fc = TMU.OneHotFeatureCreator()
    fs = MinimalRLCore.feature_size(fc)
    
    ap = ActionRNN.ϵGreedy(0.01, get_actions(env))

    init_func = (dims...)->glorot_uniform(rng, dims...)

    chain = begin
        if parsed["cell"] == "FacARNN"
            throw("You know this doesn't work yet...")
            Flux.Chain(ActionRNN.FacARNN(fs, 2, parsed["numhidden"], parsed["factors"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
        elseif parsed["cell"] == "ARNN"
            if parsed["action_embedding"]
                Flux.Chain(
                    (x)->(Flux.onehot(x[1], 1:4), x[2]),
                    ActionRNN.ActionStateStreams(Flux.Dense(4, parsed["ae_size"], tanh; initW=init_func), identity),
                    ActionRNN.ARNN(fs, parsed["ae_size"], parsed["numhidden"]; init=init_func),
                    Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
            else
                Flux.Chain(
                    # (x)->(Flux.onehot(x[1], 1:4), x[2]),
                    # ActionRNN.ActionStateStreams(Flux.Dense(4, 3, Flux.relu; initW=init_func), identity),
                    ActionRNN.ARNN(fs, 4, parsed["numhidden"]; init=init_func),
                    Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
            end
        else
            rnntype = getproperty(Flux, Symbol(parsed["cell"]))
            if parsed["action_embedding"]
                Flux.Chain(
                    (x)->(x[1:3], x[4:end]),
                    ActionRNN.ActionStateStreams(Flux.param, Flux.Dense(4, parsed["ae_size"], Flux.sigmoid; initW=init_func)),
                    (x)->vcat(x[1], x[2]),
                    rnntype(parsed["ae_size"] + fs, parsed["numhidden"]; init=init_func),
                    Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
            else
                Flux.Chain(rnntype(fs, parsed["numhidden"]; init=init_func),
                           Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
            end
        end
    end

    ActionRNN.ControlFluxAgent(chain,
                               fc,
                               fs,
                               ap,
                               parsed;
                               rng=rng)
end


function main_experiment(args::Vector{String})

    as = arg_parse()
    parsed = parse_args(args, as)
    
    savefile = ActionRNN.save_setup(parsed, "results.jld2")
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
                    pr_suc = if length(successes) <= 100
                        mean(successes)
                    else
                        mean(successes[end-100:end])
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
    ActionRNN.save_results(parsed, savefile, save_results)
end


end
