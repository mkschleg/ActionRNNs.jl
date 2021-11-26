module RingWorldOnlineExperiment

import Flux
# import Flux.Tracker
import JLD2
import LinearAlgebra.Diagonal

import ActionRNNs: ActionRNNs, ExpUtils, RingWorld, glorot_uniform
import .ExpUtils: RingWorldUtils, FluxUtils, experiment_wrapper
using MinimalRLCore



# using ActionRNNs
using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

# using Plots

const RWU = RingWorldUtils
const FLU = FluxUtils

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

#=
Time: 0:01:28
Dict{String, Matrix{Float32}} with 2 entries:
  "err"  => [0.0 0.0; -1.0 0.0; … ; -0.00895289 0.00520104; 0.00517368 -0.0150787]
  "pred" => [0.0 0.0; 0.0 0.0; … ; -0.00895289 0.00520104; 1.00517 -0.0150787]
=#
function default_args()
    Dict{String,Any}(

        "save_dir" => "tmp/ringworld_online",

        "seed" => 1,
        "steps" => 100000,
        "size" => 6,

        "cell" => "MARNN",
        "numhidden" => 15,

        "outhorde" => "onestep",
        "outgamma" => 0.9,

        "opt" => "RMSProp",
        "eta" => 0.001,
        "rho" => 0.9,
        "truncation" => 4,

        "synopsis" => false)

end

function get_model(parsed, out_horde, fc, rng)

    nh = parsed["numhidden"]
    na = 2
    
    initW = (dims...; kwargs...) -> glorot_uniform(rng, dims...; kwargs...)
    initb = (dims...; kwargs...) -> Flux.zeros(dims...)
    
    fs = size(fc)
    num_gvfs = length(out_horde)

    chain = begin
        if parsed["cell"] ∈ ActionRNNs.fac_rnn_types()

            rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
            factors = parsed["factors"]
            init_style = get(parsed, "init_style", "standard")

            initb = (dims...; kwargs...) -> Flux.zeros(dims...)

            Flux.Chain(rnn(fs, na, nh, factors;
                           init_style=init_style,
                           init=initW,
                           initb=initb),
                       Flux.Dense(nh, num_gvfs; initW=initW))

        elseif parsed["cell"] ∈ ActionRNNs.fac_tuc_rnn_types()

            rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
            action_factors = parsed["action_factors"]
            out_factors = parsed["out_factors"]
            in_factors = parsed["in_factors"]

            Flux.Chain(rnn(fs, 2, nh, action_factors, out_factors, in_factors;
                           init=initW,
                           initb=initb),
                       Flux.Dense(nh, num_gvfs; initW=initW))

        elseif parsed["cell"] ∈ ActionRNNs.rnn_types()

            rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
        
            m = Flux.Chain(
                rnn(fs, na, nh;
                    init=initW,
                    initb=initb),
                Flux.Dense(nh, num_gvfs; initW=initW))
        else
            rnntype = getproperty(Flux, Symbol(parsed["cell"]))
            Flux.Chain(rnntype(fs, nh; init=initW),
                       Flux.Dense(nh,
                                  num_gvfs;
                                  initW=initW))
        end
    end

    chain
end

function construct_agent(parsed, rng)

    fc = RWU.StandardFeatureCreator{false}()

    out_horde = RWU.get_horde(parsed, "out")

    chain = get_model(parsed, out_horde, fc, rng)
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

    experiment_wrapper(parsed, working, overwrite=overwrite) do (parsed)

        num_steps = parsed["steps"]
        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = RingWorld(parsed["size"])
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
