module LunarLanderExperiment

# include("../src/ActionRNNs.jl")

import Flux
import Flux: gpu
import JLD2
import LinearAlgebra.Diagonal
import MinimalRLCore
using MinimalRLCore: run_episode!, get_actions
import ActionRNNs

# using ActionRNNs: TMaze

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

const FLU = ActionRNNs.FluxUtils

function default_config()
    Dict{String,Any}(
        "save_dir" => "lunar_lander",

        "seed" => 1,
#         "steps" => 1000000,
        "steps" => 100,

        "cell" => "AAGRU",
        "numhidden" => 128,

        "opt" => "RMSProp",
        "eta" => 0.000355,
        "rho" =>0.99,

        "replay_size"=>100000,
        "warm_up" => 1000,
        "batch_size"=>32,
        "update_wait"=>8,
        "target_update_wait"=>1000,
        "truncation" => 16,

        "hs_learnable" => true,
        "encoding_size" => 128,
        "omit_states" => [6],
        "state_conditions" => [2],

        "gamma"=>0.99)

    
end

function get_ann(parsed, fs, env, rng)

    nh = parsed["numhidden"]
    es = parsed["encoding_size"]
    na = length(get_actions(env))
    init_func = (dims...)->ActionRNNs.glorot_uniform(rng, dims...)
    
    rnn_layer = if parsed["cell"] ∈ ActionRNNs.fac_rnn_types()

        rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
        factors = parsed["factors"]
        init_func = (dims...; kwargs...)->
            ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
        initb = (dims...; kwargs...) -> Flux.zeros(dims...)
        
        rnn(es, na, nh, factors;
            init=init_func,
            initb=initb)
        
    elseif parsed["cell"] ∈ ActionRNNs.rnn_types()

        rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
        
        init_func = (dims...; kwargs...)->
            ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
        initb = (dims...; kwargs...) -> Flux.zeros(dims...)
        
        rnn(es, na, nh;
            init=init_func,
            initb=initb)

    else
        rnntype = getproperty(Flux, Symbol(parsed["cell"]))
        rnntype(es, nh; init=init_func)
    end

#     Flux.Chain(Flux.Dense(fs, es; initW=init_func),
#                rnn_layer,
#                Flux.Dense(nh, nh; initW=init_func),
#                Flux.Dense(nh, na; initW=init_func))

   Flux.Chain(Flux.Dense(fs, es, Flux.relu; initW=init_func),
               rnn_layer,
               Flux.Dense(nh, nh, Flux.relu; initW=init_func),
               Flux.Dense(nh, na; initW=init_func))
end

function construct_agent(env, parsed, rng)

    fc = ActionRNNs.LunarLanderUtils.IdentityFeatureCreator()
    fs = MinimalRLCore.feature_size(fc, parsed["omit_states"])

    γ = Float32(parsed["gamma"])
    τ = parsed["truncation"]


    ap = ActionRNNs.ϵGreedyDecay((1.0, 0.05), 10000, 1000, MinimalRLCore.get_actions(env))

    opt = FLU.get_optimizer(parsed)

    chain = get_ann(parsed, fs, env, rng)# |> gpu

    ActionRNNs.DRQNAgent(chain,
                         opt,
                         τ,
                         γ,
                         fc,
                         fs,
                         3,
                         parsed["replay_size"],
                         parsed["warm_up"],
                         parsed["batch_size"],
                         parsed["update_wait"],
                         parsed["target_update_wait"],
                         ap,
                         parsed["hs_learnable"])
end

function main_experiment(parsed = default_config(); working=false, progress=false, verbose=false)

    GC.gc()

    ll_experiment_wrapper(parsed, working) do parsed, data_dir

        num_steps = parsed["steps"]

        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = ActionRNNs.LunarLander(seed, false, parsed["omit_states"], parsed["state_conditions"])
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

        mean_loss = 1.0f0
        
        prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
        eps = 1
        checkpoint = 1
        while sum(logger.data.total_steps) <= num_steps
            if sum(logger.data.total_steps) > checkpoint * 500000
                ActionRNNs.save_results("$(data_dir)/results.jld2", logger.data)

                GC.gc()
                checkpoint += 1
            end
            usa = ActionRNNs.UpdateStateAnalysis(
                (l1 = 0.0f0, loss = 0.0f0, avg_loss = 1.0f0),
                Dict(
                    :l1 => ActionRNNs.l1_grad,
                    :loss => (s, us) -> s + us.loss,
                    :avg_loss => (s, us) -> 0.99 * s + 0.01 * us.loss
                ))
            success = false
            max_episode_steps = min(max((num_steps - sum(logger[:total_steps])), 2), 10000)
            n = 0
            tr = if length(logger.data[:total_rews]) > 100
                mean(logger.data[:total_rews][end-100:end])
            else
                mean(logger.data[:total_rews][1:end])
            end
            total_rew, steps = run_episode!(env, agent, max_episode_steps, rng) do (s, a, s′, r)
                if progress
                    pr_suc = if length(logger.data.successes) <= 1000
                        mean(logger.data.successes)
                    else
                        mean(logger.data.successes[end-1000:end])
                    end
                    # tr += r
                    next!(prg_bar, showvalues=[(:episode, eps),
                                               (:total_rews, tr),
                                               (:loss, usa[:avg_loss]),
                                               (:l1, usa[:l1]/n),
                                               (:action, a.action),
                                               (:preds, a.preds)])

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
    end
end

function ll_experiment_wrapper(exp_func::Function, parsed, working; overwrite=false)
    savefile = ActionRNNs.save_setup(parsed)
    if isfile(savefile) && ActionRNNs.check_save_file_loadable(savefile) && !overwrite
        return
    end

    data_dir = rsplit(savefile, "/"; limit=2)
    ret = exp_func(parsed, data_dir[1])

    if working
        ret
    elseif ret isa NamedTuple
        ActionRNNs.save_results(savefile, ret.save_results)
    else
        ActionRNNs.save_results(savefile, ret)
    end
end


end
