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

using Plots

const RWU = ActionRNN.RingWorldUtils
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
    ActionRNN.env_settings!(as, RingWorld)
    # ActionRNN.agent_settings!(as, ActionRNN.FluxAgent)
    RWU.horde_settings!(as, "out")

    @add_arg_table as begin
        "--pred_type"
        arg_type = String
        default = "const"
        "--pred_val"
        arg_type = Float32
        default = 0.0f0
    end
    
    return as
end

# function construct_agent(parsed, rng)

    # out_horde = RWU.gammas_term(collect(0.0:0.1:0.9))
#     out_horde = RWU.get_horde(parsed, "out")
#     fc = RWU.OneHotFeatureCreator()
#     fs = RLCore.feature_size(fc)
#     ap = ActionRNN.RandomActingPolicy([0.5, 0.5])


#     init_func = (dims...)->glorot_uniform(rng, dims...)

#     chain = begin
#         if parsed["cell"] == "FacARNN"
#             Flux.Chain(ActionRNN.FacARNN(fs, 2, parsed["numhidden"], parsed["factors"]; init=init_func),
#                        Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func))
#         elseif parsed["cell"] == "ARNN"
#             Flux.Chain(ActionRNN.ARNN(fs, 2, parsed["numhidden"]; init=init_func),
#                        Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func))
#         else
#             Flux.Chain(Flux.RNN(fs, parsed["numhidden"]; init=init_func),
#                        Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func))
#         end
#     end

#     ActionRNN.FluxAgent(out_horde,
#                         chain,
#                         fc,
#                         fs,
#                         # 2,
#                         ap,
#                         parsed;
#                         rng=rng,
#                         init_func=(dims...)->glorot_uniform(rng, dims...))
# end

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

    ap = ActionRNN.RandomActingPolicy([0.5, 0.5])
    
    # agent = construct_agent(parsed, rng)
    out_horde = RWU.get_horde(parsed, "out")
    
    out_pred_strg = zeros(Float32, num_steps, length(out_horde))
    out_err_strg = zeros(Float32, num_steps, length(out_horde))

    
    if parsed["pred_type"] == "const"
        
        out_pred_strg .= parsed["pred_val"]

    elseif parsed["pred_type"] == "rand"

        out_pred_strg .= rand(rng, size(out_pred_strg)...)
        
    else
        throw("pred type not implemented")
    end
    # fill_strategy(out_pred_strg)
    
    err_func! = (step) -> begin;
        out_err_strg[step, :] = out_pred_strg[step, :] .- RWU.oracle(env, parsed["outhorde"]);
    end;


    s_t = start!(env, rng)
    action = ap(nothing, rng)[1]

    # println(action)
    for s ∈ 1:num_steps
        s_tp1, rew, term = step!(env, action, rng)
        action = ap(nothing, rng)[1]
        err_func!(s)
    end
    
    # pred_experiment(env, agent, rng, num_steps, parsed, callback)

    results = Dict(["pred"=>out_pred_strg, "err"=>out_err_strg])
    save_results = results_synopsis(results, Val(parsed["synopsis"]))
    # if parsed["visualize"]
    #     mp4(anim, parsed["cell"]*"_"*string(parsed["numhidden"])*".mp4")
    # end
    ActionRNN.save_results(parsed, savefile, save_results)
end


function pred_experiment(env, agent, rng, num_steps, parsed, callback)

    
    s_t = start!(env, rng)
    action = start!(agent, s_t, rng)
    
    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")

    hs = ActionRNN.get_hidden_state(agent.model)
    hs_strg = CircularBuffer{typeof(hs)}(64)
    
    for step in 1:num_steps

        s_tp1, rew, term = step!(env, action, rng)
        out_preds, action = step!(agent, s_tp1, rew, term, rng)

        callback(env, agent, (s_tp1, rew, term), (out_preds, step))
        
        if parsed["verbose"]
            println(step)
            println(env)
            println(agent)
        end

        if parsed["progress"]
           ProgressMeter.next!(prg_bar)
        end

    end
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end
