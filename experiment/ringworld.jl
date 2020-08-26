
module RingWorldExperiment

import Flux
# import Flux.Tracker
import JLD2
import LinearAlgebra.Diagonal

import ActionRNNs

using DataStructures: CircularBuffer
using ActionRNNs: RingWorld, step!, start!, glorot_uniform

# using ActionRNNs
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random
using MinimalRLCore

# using Plots

const RWU = ActionRNNs.RingWorldUtils
const FLU = ActionRNNs.FluxUtils

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

function default_arg_parse()
    Dict{String,Any}(
        "save_dir" => "ringworld",

        "seed" => 1,
        "steps" => 200000,
        "size" => 6,

        # "features" => "OneHot",
        # "cell" => "ARNN",
        "rnn_config" => "ARNN_OneHot",
        "numhidden" => 6,
        
        "outhorde" => "gammas_term",
        "outgamma" => 0.9,
        
        "opt" => "RMSProp",
        "optparams" => [0.001],
        "truncation" => 3,

        "verbose" => false,
        "synopsis" => false,
        "prev_action_or_not" => false,
        "progress" => true,
        "working" => true,
        "visualize" => false)
end

function get_rnn_config(parsed, out_horde, rng)

    rnn_config_str = parsed["rnn_config"]
    
    cell_str, fc_str = split(rnn_config_str, "_")
    
    fc = if fc_str == "OneHot"
        RWU.OneHotFeatureCreator()
    elseif fc_str == "SansAction"
        RWU.SansActionFeatureCreator()
    else
        throw(fc_str * " not a feature creator.")
    end
    
    fs = MinimalRLCore.feature_size(fc)
    
    init_func = (dims...)->glorot_uniform(rng, dims...)

    chain = begin
        if cell_str == "FacARNN"
            Flux.Chain(ActionRNNs.FacARNN(fs, 2, parsed["numhidden"], parsed["factors"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func))
        elseif cell_str == "ARNN"
            Flux.Chain(ActionRNNs.ARNN(fs, 2, parsed["numhidden"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func))
        elseif cell_str == "RNN"
            Flux.Chain(Flux.RNN(fs, parsed["numhidden"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func))
        else
            throw("Unknown Cell type " * cell_str)
        end
    end

    fc, fs, chain
end

function construct_agent(parsed, rng)


    # out_horde = RWU.gammas_term(collect(0.0:0.1:0.9))
    out_horde = RWU.get_horde(parsed, "out")

    fc, fs, chain = get_rnn_config(parsed, out_horde, rng)
    opt_func = getproperty(Flux, Symbol(parsed["opt"]))
    opt = opt_func(parsed["alpha"])

    
    ap = ActionRNNs.RandomActingPolicy([0.5, 0.5])

    ActionRNNs.FluxAgent(out_horde,
                         chain,
                         opt,
                         fc,
                         fs,
                         ap,
                         parsed)
end

function main_experiment(parsed::Dict{String, Any})

    savefile = ActionRNNs.save_setup(parsed)
    if isnothing(savefile)
        return
    end

    prgs = get(parsed, "progress", false)

    num_steps = parsed["steps"]

    seed = parsed["seed"]
    rng = Random.MersenneTwister(seed)

    env = RingWorld(parsed)
    agent = construct_agent(parsed, rng)

    out_pred_strg = zeros(Float32, num_steps, length(agent.horde))
    out_err_strg = zeros(Float32, num_steps, length(agent.horde))
    hidden_state = zeros(Float32, num_steps, parsed["numhidden"])

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    
    cur_step = 1
    run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
        
        out_preds = a.preds
        out_pred_strg[cur_step, :] .= out_preds
        out_err_strg[cur_step, :] = out_pred_strg[cur_step, :] .- RWU.oracle(env, parsed["outhorde"]);
        hidden_state[cur_step, :] .= a.h[agent.model[1]]

        # @show env
        if prgs
            ProgressMeter.next!(prg_bar)
        end

        cur_step += 1
    end
    

    results = Dict(["pred"=>out_pred_strg, "err"=>out_err_strg, "hidden"=>hidden_state])
    save_results = results_synopsis(results, Val(parsed["synopsis"]))
    ActionRNNs.save_results(parsed, savefile, save_results)
end

end


    # anim = Animation()
    # visualize_callback = if parsed["visualize"]
    #     (agent, step) -> begin
    #         if step > num_steps - 10000
    #             ActionRNNs.reset!(agent.model, agent.hidden_state_init)
    #             agent.model.(agent.state_list)
    #             hs = Flux.data(ActionRNNs.get_hidden_state(agent.model))
    #             push!(hs_strg, hs)
    #             ky = collect(keys(hs_strg[1]))
    #             if length(hs_strg) > 10 && (step%4) == 0
    #                 plot(
    #                     heatmap(hcat(getindex.(hs_strg, ky)...)),
    #                     plot(out_pred_strg[step-64:step, :]),
    #                     plot(mean(out_err_strg[step-64:step, :].^2; dims=2)),
    #                     layout=(3,1),
    #                     legend=false)
    #                 frame(anim)
    #             end
    #         end
    #     end
    # else
    #     (agent, step) -> nothing
    # end
    # err_func! = (env, agent, (s_tp1, rew, term), (out_preds, step)) -> begin;
    #     out_pred_strg[step, :] .= Flux.data(out_preds);
    #     out_err_strg[step, :] = out_pred_strg[step, :] .- RWU.oracle(env, parsed["outhorde"]);
    #     ActionRNNs.reset!(agent.model, agent.hidden_state_init)
    #     agent.model.(agent.state_list)
    #     size(ActionRNNs.get_hidden_state(agent.model[1]))
    #     hidden_state[step, :] .= ActionRNNs.get_hidden_state(agent.model[1])
    # end;
