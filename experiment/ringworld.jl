__precompile__(false)

module RingWorldOnlineExperiment

import Flux
# import Flux.Tracker
import JLD2
import LinearAlgebra.Diagonal

# include("../src/ActionRNNs.jl")
import ActionRNNs
using MinimalRLCore

using DataStructures: CircularBuffer
using ActionRNNs: RingWorld, step!, start!, glorot_uniform

# using ActionRNNs
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random


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

function default_args()
    Dict{String,Any}(

        "save_dir" => "ringworld_online",

        "seed" => 1,
#         "steps" => 200000,
        "steps" => 50000,
        "size" => 6,

        # "features" => "OneHot",
#         "cell" => "AARNN",
        "cell" => "MARNN",
#         "rnn_config" => "MARNN_OneHot",
        "numhidden" => 6,
        "hs_learnable" => true,
        
#         "outhorde" => "gammas_term",
        "outhorde" => "onestep",
        "outgamma" => 0.9,

        "action_features"=>false,

#         "alpha" => 0.001,
        "opt" => "RMSProp",
        "eta" => 0.001,
        "rho" => 0.9,
        "truncation" => 3,

        "synopsis" => false)

end

function get_model(parsed, out_horde, fc, rng)


#     rnn_config_str = parsed["rnn_config"]
#
#     cell_str, fc_str = split(rnn_config_str, "_")

#     fc = if fc_str == "OneHot"
#         RWU.OneHotFeatureCreator()
#     elseif fc_str == "SansAction"
#         RWU.SansActionFeatureCreator()
#     else
#         throw(fc_str * " not a feature creator.")
#     end
    
    nh = parsed["numhidden"]
    init_func = (dims...)->glorot_uniform(rng, dims...)
#     fs = MinimalRLCore.feature_size(fc)
    fs = size(fc)
    num_gvfs = length(out_horde)

    chain = begin
        if parsed["cell"] == "FacARNN"

            factors = parsed["factors"]
            Flux.Chain(ActionRNNs.FacARNN(fs, 2, nh, factors,; init=init_func),
                       Flux.Dense(nh, num_gvfs; initW=init_func))
        elseif parsed["cell"] ∈ ActionRNNs.rnn_types()

            rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
        
            init_func = (dims...; kwargs...)->
                ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
            initb = (dims...; kwargs...) -> Flux.zeros(dims...)
        
            m = Flux.Chain(
                rnn(fs, 2, nh;
                    init=init_func,
                    initb=initb),
                Flux.Dense(nh, num_gvfs; initW=init_func))
        else
            rnntype = getproperty(Flux, Symbol(parsed["cell"]))
            Flux.Chain(rnntype(fs, nh; init=init_func),
                       Flux.Dense(nh,
                                  num_gvfs;
                                  initW=init_func))
        end
    end

#     fc, fs, chain
    chain
end

function construct_agent(parsed, rng)

    fc = RWU.StandardFeatureCreator{parsed["action_features"]}()
    fs = size(fc)

    # out_horde = RWU.gammas_term(collect(0.0:0.1:0.9))
    out_horde = RWU.get_horde(parsed, "out")

#     fc, fs, chain = get_rnn_config(parsed, out_horde, rng)
    chain = get_model(parsed, out_horde, fc, rng)
#     opt_func = getproperty(Flux, Symbol(parsed["opt"]))
#     opt = opt_func(parsed["alpha"])
    opt = FLU.get_optimizer(parsed)

    
    ap = ActionRNNs.RandomActingPolicy([0.5, 0.5])
    τ = parsed["truncation"]

    ActionRNNs.PredOnlineAgent(out_horde,
                               chain,
                               opt,
                               τ,
                               fc,
                               ap)
end

function main_experiment(parsed=default_args(); working=false, progress=false, overwrite=false)

    ActionRNNs.experiment_wrapper(parsed, working, overwrite=overwrite) do (parsed)

        num_steps = parsed["steps"]
        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = RingWorld(parsed)
        agent = construct_agent(parsed, rng)

        out_pred_strg, out_err_strg =
            experiment_loop(env, agent, parsed["outhorde"], num_steps, rng; prgs=progress)
        
        results = Dict(["pred"=>out_pred_strg, "err"=>out_err_strg])
        save_results = results_synopsis(results, Val(parsed["synopsis"]))
        (save_results=save_results)
    end
end

# Creating an environment for to run in jupyter.
function experiment_loop(env, agent, outhorde_str, num_steps, rng; prgs=false)

    out_pred_strg = zeros(Float32, num_steps, length(agent.horde))
    out_err_strg = zeros(Float32, num_steps, length(agent.horde))
    out_loss_strg = zeros(Float32, num_steps)

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    
    cur_step = 1
    run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
        
        out_preds = a.preds

        out_pred_strg[cur_step, :] .= out_preds[:, 1]
        out_err_strg[cur_step, :] = out_pred_strg[cur_step, :] .- RWU.oracle(env, outhorde_str);

        out_loss_strg[cur_step] = a.loss

        if prgs
            ProgressMeter.next!(prg_bar)
        end

        cur_step += 1
    end
    
    out_pred_strg, out_err_strg, out_loss_strg
    
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
