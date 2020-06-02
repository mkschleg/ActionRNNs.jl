
module RingWorldFluxExperiment

import Flux
# import Flux.Tracker
import JLD2
import LinearAlgebra.Diagonal

import ActionRNN

using DataStructures: CircularBuffer
using ActionRNN: RingWorld, step!, start!, glorot_uniform

# using ActionRNN
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random
using MinimalRLCore

# using Plots

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

function default_arg_parse()
    Dict{String,Any}(
        "save_dir" => "ringworld",

        "seed" => 1,
        "steps" => 200000,
        "size" => 6,

        "cell" => "ARNN",
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

function construct_agent(parsed, rng)


    # out_horde = RWU.gammas_term(collect(0.0:0.1:0.9))
    out_horde = RWU.get_horde(parsed, "out")
    fc = RWU.OneHotFeatureCreator()
    fs = MinimalRLCore.feature_size(fc)
    ap = ActionRNN.RandomActingPolicy([0.5, 0.5])


    init_func = (dims...)->glorot_uniform(rng, dims...)

    chain = begin
        if parsed["cell"] == "FacARNN"
            Flux.Chain(ActionRNN.FacARNN(fs, 2, parsed["numhidden"], parsed["factors"]; init=init_func),
                       Flux.Dense(parsed["numhidden"], length(out_horde); initW=init_func))
        elseif parsed["cell"] == "ARNN"
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
                        ap,
                        parsed)
end

function main_experiment(parsed::Dict{String, Any})

    savefile = ActionRNN.save_setup(parsed, "results.jld2")
    if isnothing(savefile)
        return
    end

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
        out_pred_strg[cur_step, :] .= Flux.data(out_preds)
        out_err_strg[cur_step, :] = out_pred_strg[cur_step, :] .- RWU.oracle(env, parsed["outhorde"]);
        hidden_state[cur_step, :] .= a.h[agent.model[1]]

        ProgressMeter.next!(prg_bar)

        cur_step += 1
    end
    

    results = Dict(["pred"=>out_pred_strg, "err"=>out_err_strg, "hidden"=>hidden_state])
    save_results = results_synopsis(results, Val(parsed["synopsis"]))
    ActionRNN.save_results(parsed, savefile, save_results)
end

end


    # anim = Animation()
    # visualize_callback = if parsed["visualize"]
    #     (agent, step) -> begin
    #         if step > num_steps - 10000
    #             ActionRNN.reset!(agent.model, agent.hidden_state_init)
    #             agent.model.(agent.state_list)
    #             hs = Flux.data(ActionRNN.get_hidden_state(agent.model))
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
    #     ActionRNN.reset!(agent.model, agent.hidden_state_init)
    #     agent.model.(agent.state_list)
    #     size(ActionRNN.get_hidden_state(agent.model[1]))
    #     hidden_state[step, :] .= ActionRNN.get_hidden_state(agent.model[1])
    # end;
