
module RingWorldFluxExperiment

import Flux
import Flux.Tracker
import JLD2
import LinearAlgebra.Diagonal
import RLCore
import ActionRNN

using DataStructures: CircularBuffer
using ActionRNN: RingWorld, step!, start!, glorot_uniform

# using ActionRNN
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

const RWU = ActionRNN.RingWorldUtils
const FLU = ActionRNN.FluxUtils

function arg_parse(as::ArgParseSettings = ArgParseSettings(exc_handler=Reproduce.ArgParse.debug_handler))

    ActionRNN.exp_settings!(as)
    ActionRNN.env_settings!(as, RingWorld)
    ActionRNN.agent_settings!(as, ActionRNN.FluxAgent)
    
    return as
end

function construct_agent(parsed, rng)
    out_horde = RWU.onestep()
    fc = RWU.OneHotFeatureCreator()
    fs = RLCore.feature_size(fc)
    ap = ActionRNN.RandomActingPolicy([0.5, 0.5])


    init_func = (dims...)->glorot_uniform(rng, dims...)

    chain = begin
        if parsed["cell"] == "ARNN"
            Flux.Chain(ActionRNN.ARNN(fs, 2, parsed["numhidden"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func))
        else
            Flux.Chain(Flux.RNN(fs, parsed["numhidden"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func))
        end
    end
    
    ActionRNN.FluxAgent(out_horde,
                        chain,
                        fc,
                        fs,
                        # 2,
                        ap,
                        parsed;
                        rng=rng,
                        init_func=(dims...)->glorot_uniform(rng, dims...))
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

    env = RingWorld(parsed)

    out_pred_strg = zeros(num_steps, 2)
    out_err_strg = zeros(num_steps, 2)

    err_func! = (env, out_preds, step) -> begin;
        out_pred_strg[step, :] .= Flux.data(out_preds);
        out_err_strg[step, :] = out_pred_strg[step, :] .- RWU.oracle(env, "onestep");

    end;

    agent = construct_agent(parsed, rng)

    ActionRNN.pred_experiment(env, agent, rng, num_steps, parsed, err_func!)

    results = Dict(["pred"=>out_pred_strg, "err"=>out_err_strg])
    ActionRNN.save_results(parsed, savefile, results)
end



Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end
