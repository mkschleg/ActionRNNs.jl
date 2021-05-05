module TMazeExperiment

# include("../src/ActionRNNs.jl")

import Flux
import JLD2
import LinearAlgebra.Diagonal
import MinimalRLCore
using MinimalRLCore: run_episode!, get_actions
import ActionRNNs

using ActionRNNs: TMaze

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

const TMU = ActionRNNs.TMazeUtils
const FLU = ActionRNNs.FluxUtils

function default_config()
    Dict{String,Any}(
        "save_dir" => "tmaze_online",

        "seed" => 1,
        "steps" => 80000,
        "size" => 6,

        "cell" => "MARNN",
        "numhidden" => 6,
        
        "opt" => "RMSProp",
        "eta" => 0.0005,
        "rho" => 0.99,
        "truncation" => 8,

#         "hs_learnable" => true,

        "gamma"=>0.99)

end

function get_ann(parsed, fs, env, rng)

    nh = parsed["numhidden"]
    init_func = (dims...)->ActionRNNs.glorot_uniform(rng, dims...)

    if parsed["cell"] == "FacARNN"

        factors = parsed["factors"]

        Flux.Chain(ActionRNNs.FacARNN(fs, 4, nh, factors;
                                      init=init_func,
                                      initb=init_func),
                   Flux.Dense(nh, length(get_actions(env)); initW=init_func))

    elseif parsed["cell"] ∈ ActionRNNs.rnn_types()

        rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))

        init_func = (dims...; kwargs...)->
            ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
        initb = (dims...; kwargs...) -> Flux.zeros(dims...)

        m = Flux.Chain(
            rnn(fs, 4, nh;
                init=init_func,
                initb=initb),
            Flux.Dense(nh, length(get_actions(env)); initW=init_func))


    else

        rnntype = getproperty(Flux, Symbol(parsed["cell"]))
        Flux.Chain(rnntype(fs, nh; init=init_func),
                   Flux.Dense(nh,
                              length(get_actions(env));
                              initW=init_func))
    end

#     chain = begin
#         if false
#             # ARNN
#             Flux.Chain(
#                 (x)->(Flux.onehot(x[1], 1:4), x[2]),
#                 ActionRNNs.ActionStateStreams(Flux.Dense(4, parsed["ae_size"], tanh; initW=init_func), identity),
#                 ActionRNNs.ARNN(fs, parsed["ae_size"], parsed["numhidden"]; init=init_func),
#                 Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
#
#             # generic RNN
#             Flux.Chain(
#                 (x)->(x[1:3], x[4:end]),
#                 ActionRNNs.ActionStateStreams(Flux.param,
#                                               Flux.Dense(4, parsed["ae_size"], Flux.sigmoid; initW=init_func)),
#                 (x)->vcat(x[1], x[2]),
#                 rnntype(parsed["ae_size"] + fs, parsed["numhidden"]; init=init_func),
#                 Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
#         else
#             if parsed["cell"] == "FacARNN"
#                 # throw("You know this doesn't work yet...")
#                 Flux.Chain(ActionRNNs.FacARNN(fs, 4, parsed["numhidden"], parsed["factors"]; hs_learnable=parsed["hs_learnable"], init=init_func),
#                            Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
#             elseif parsed["cell"] == "ARNN"
#                 Flux.Chain(
#                     ActionRNNs.ARNN(fs, 4, parsed["numhidden"]; init=init_func, hs_learnable=parsed["hs_learnable"]),
#                     Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
#             elseif parsed["cell"] == "RNN"
#                 Flux.Chain(
#                     ActionRNNs.RNN(fs, parsed["numhidden"]; init=init_func, hs_learnable=parsed["hs_learnable"]),
#                     Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
#             else
#                 rnntype = getproperty(Flux, Symbol(parsed["cell"]))
#                 Flux.Chain(rnntype(fs, parsed["numhidden"]; init=init_func),
#                            Flux.Dense(parsed["numhidden"], length(get_actions(env)); initW=init_func))
#             end
#         end
#     end
end

function construct_agent(env, parsed, rng)

    num_actions = length(get_actions(env))
    is_actionrnn = parsed["cell"] ∈ ["FacARNN", "ARNN"]

    fc = TMU.StandardFeatureCreator{false}()
    fs = MinimalRLCore.feature_size(fc)

    γ = Float32(parsed["gamma"])
    τ = parsed["truncation"]


    ap = ActionRNNs.ϵGreedy(0.1, MinimalRLCore.get_actions(env))

    opt = FLU.get_optimizer(parsed)
    
    chain = get_ann(parsed, fs, env, rng)

#     fc = if is_actionrnn
#         # States without one hot action encoding.
#         TMU.StandardFeatureCreator{false}()
#     else
#         # States with one hot action encoding
#         TMU.StandardFeatureCreator{true}()
#     end
#     fs = MinimalRLCore.feature_size(fc)

    # TODO: add parsed["hs_learnable"]
    ActionRNNs.ControlOnlineAgent(chain,
                                  opt,
                                  τ,
                                  γ,
                                  fc,
                                  fs,
                                  ap)
end

function main_experiment(parsed=default_config(); working=false, progress=false)

    ActionRNNs.experiment_wrapper(parsed, working) do (parsed)

        num_steps = parsed["steps"]

        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = ActionRNNs.TMaze(parsed["size"])
#         env = TMaze(parsed["size"])
        agent = construct_agent(env, parsed, rng)


        logger = ActionRNNs.SimpleLogger(
            (:total_rews, :losses, :successes, :total_steps, :l1),
            (Float32, Float32, Bool, Int, Float32),
            Dict(
                :total_rews => (rew, steps, success, usa) -> rew,
                :losses => (rew, steps, success, usa) -> usa[:loss]/steps,
                :successes => (rew, steps, success, usa) -> success,
                :total_steps => (rew, steps, success, usa) -> steps,
                :l1 => (rew, steps, success, usa) -> usa[:l1]/steps
            )
        )

        prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
        eps = 1
        while sum(logger[:total_steps]) <= num_steps
            usa = ActionRNNs.UpdateStateAnalysis(
                (l1 = 0.0f0, loss = 0.0f0, avg_loss = 1.0f0),
                Dict(
                    :l1 => ActionRNNs.l1_grad,
                    :loss => (s, us) -> s + us.loss,
                    :avg_loss => (s, us) -> 0.9 * s + 0.1 * us.loss
                ))
            success = false
            max_episode_steps = min(max((num_steps - sum(logger[:total_steps])), 2), 10000)
            n = 0
            total_rew, steps = run_episode!(env, agent, max_episode_steps, rng) do (s, a, s′, r)
                if progress
                    pr_suc = if length(logger[:successes]) <= 1000
                        mean(logger[:successes])
                    else
                        mean(logger[:successes][end-1000:end])
                    end
                    next!(prg_bar, showvalues=[(:episode, eps),
                                               (:successes, pr_suc),
                                               (:loss, usa[:avg_loss]),
                                               (:l1, usa[:l1]/n),
                                               (:action, a.action),
                                               (:preds, a.preds),
                                               (:grad, a.update_state isa Nothing ? 0.0f0 : sum(a.update_state.grads[agent.model[1].cell.Wh]))])
                end
                success = success || (r == 4.0)
                if !(a.update_state isa Nothing)
                    usa(a.update_state)
                end
                n+=1
            end
            logger(total_rew, steps, success, usa)
            eps += 1
        end
         save_results = logger.data
        (;save_results = save_results)
#         (;agent = agent, save_results = logger.data)
    end
end


end
