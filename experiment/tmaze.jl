
module TMazeExperiment

include("../src/ActionRNN.jl")

import Flux
import Flux.Tracker
import JLD2
import LinearAlgebra.Diagonal
import RLCore
import ActionRNN

using .ActionRNN: TMaze, step!, start!, is_terminal, get_actions, glorot_uniform

# using ActionRNN
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

# using Plots

const TMU = ActionRNN.TMazeUtils
const FLU = ActionRNN.FluxUtils

function results_synopsis(res, ::Val{true})
    rmse = sqrt.(mean(res["err"].^2; dims=2))
    Dict([
        "desc"=>"All operations are on the RMSE",
        "all"=>mean(rmse),
        "end"=>mean(rmse[end-50000:end]),
        "lc"=>mean(reshape(rmse, 1000, :); dims=1)[1,:],
        "var"=>var(reshape(rmse, 1000, :); dims=1)[1,:]
    ])
end

results_synopsis(res, ::Val{false}) = res

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
    end
    
    return as
end

function construct_agent(env, parsed, rng)

    fc = TMU.OneHotFeatureCreator()
    fs = RLCore.feature_size(fc)
    
    ap = ActionRNN.ϵGreedy(0.1, get_actions(env))

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
                    ActionRNN.ActionStateStreams(Flux.Dense(4, 3, Flux.relu; initW=init_func), identity),
                    ActionRNN.ARNN(fs, 3, parsed["numhidden"]; init=init_func),
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
                    ActionRNN.ActionStateStreams(Flux.param, Flux.Dense(4, 3, Flux.relu; initW=init_func)),
                    (x)->vcat(x[1], x[2]),
                    rnntype(6, parsed["numhidden"]; init=init_func),
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
    num_episodes = 10000

    seed = parsed["seed"]
    rng = Random.MersenneTwister(seed)


    env = TMaze(parsed["size"])
    agent = construct_agent(env, parsed, rng)

    total_rews = zeros(Float32, num_episodes)
    successes = fill(false, num_episodes)
    @showprogress "Episode: " for eps in 1:num_episodes
        s_t = start!(env, rng)
        action = start!(agent, s_t, rng)
        step = 1
        while RLCore.is_terminal(env) == false
            trns = step!(env, action, rng)
            action = step!(agent, trns..., rng)
            total_rews[eps] += trns[2]
            if is_terminal(env)
                successes[eps] = trns[2] == 4.0
            end
            step += 1
            if step == 500
                # println("Episode 500")
                break;
            end
        end
    end
    total_rews, successes

end


end
