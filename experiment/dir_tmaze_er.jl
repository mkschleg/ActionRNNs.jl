module DirectionalTMazeERExperiment

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
        "save_dir" => "tmaze",

        "seed" => 1,
        "steps" => 80000,
        "size" => 6,

        "cell" => "MARNN",
        "numhidden" => 6,

        "opt" => "RMSProp",
        "eta" => 0.0005,
        "rho" =>0.99,

        "replay_size"=>1000,
        "warm_up" => 1000,
        "batch_size"=>4,
        "update_wait"=>4,
        "target_update_wait"=>1000,
        "truncation" => 8,

        "hs_learnable" => true,
        
        "gamma"=>0.99)

    
end

function get_ann(parsed, fs, env, rng)

    nh = parsed["numhidden"]
    na = length(get_actions(env))
    init_func = (dims...)->ActionRNNs.glorot_uniform(rng, dims...)
    
    
    if parsed["cell"] ∈ ActionRNNs.fac_rnn_types()

        rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
        factors = parsed["factors"]
        init_style = get(parsed, "init_style", "standard")
        
        init_func = (dims...; kwargs...)->
            ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
        initb = (dims...; kwargs...) -> Flux.zeros(dims...)
        
        Flux.Chain(rnn(fs, na, nh, factors;
                       init_style=init_style,
                       init=init_func,
                       initb=initb),
                   Flux.Dense(nh, na; initW=init_func))

     elseif parsed["cell"] ∈ ActionRNNs.fac_tuc_rnn_types()

        rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
        action_factors = parsed["action_factors"]
        out_factors = parsed["out_factors"]
        in_factors = parsed["in_factors"]
        init_func = (dims...; kwargs...)->
            ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
        initb = (dims...; kwargs...) -> Flux.zeros(dims...)

        Flux.Chain(rnn(fs, na, nh, action_factors, out_factors, in_factors;
                       init=init_func,
                       initb=initb),
                   Flux.Dense(nh, na; initW=init_func))
        
    elseif parsed["cell"] ∈ ActionRNNs.rnn_types()

        rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
        
        init_func = (dims...; kwargs...)->
            ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
        initb = (dims...; kwargs...) -> Flux.zeros(dims...)
        
        m = Flux.Chain(
            rnn(fs, na, nh;
                init=init_func,
                initb=initb),
            Flux.Dense(nh, na; initW=init_func))


    else
        rnntype = getproperty(Flux, Symbol(parsed["cell"]))
        Flux.Chain(rnntype(fs, nh; init=init_func),
                   Flux.Dense(nh,
                              length(get_actions(env));
                              initW=init_func))
    end
end

function construct_agent(env, parsed, rng)

    fc = TMU.StandardFeatureCreator{false}()
    fs = MinimalRLCore.feature_size(fc)

    γ = Float32(parsed["gamma"])
    τ = parsed["truncation"]


    ap = ActionRNNs.ϵGreedy(0.1, MinimalRLCore.get_actions(env))

    opt = FLU.get_optimizer(parsed)

    chain = get_ann(parsed, fs, env, rng)

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

    if "cell_numhidden" ∈ keys(parsed)
        parsed["cell"] = parsed["cell_numhidden"][1]
        parsed["numhidden"] = parsed["cell_numhidden"][2]
        delete!(parsed, "cell_numhidden")
    end
    
    ActionRNNs.experiment_wrapper(parsed, working) do parsed

        num_steps = parsed["steps"]

        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = ActionRNNs.DirectionalTMaze(parsed["size"])
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
        while sum(logger.data.total_steps) <= num_steps
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
            total_rew, steps = run_episode!(env, agent, max_episode_steps, rng) do (s, a, s′, r)
                if progress
                    pr_suc = if length(logger.data.successes) <= 1000
                        mean(logger.data.successes)
                    else
                        mean(logger.data.successes[end-1000:end])
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
    end
end



end
