module TMazeOnlineExperiment

import Flux
import JLD2

import MinimalRLCore: MinimalRLCore, run_episode!, get_actions
import ActionRNNs: ActionRNNs, TMaze, ExpUtils, DRQNAgent

import .ExpUtils: SimpleLogger, UpdateStateAnalysis, l1_grad, experiment_wrapper
import .ExpUtils: TMazeUtils as TMU, FluxUtils as FLU

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

#=
Time: 0:01:51
  episode:    5384
  successes:  0.6013986013986014
  loss:       0.99005485
  l1:         0.16161172
  action:     2
  preds:      Float32[1.7402661, 2.2144663, 1.4456049, 1.9616138]
  grad:       0.0032181505
=#
function default_config()
    Dict{String,Any}(
        "save_dir" => "tmp/tmaze_online",

        "seed" => 1,
        "steps" => 80000,
        "size" => 10,

        "cell" => "MARNN",
        "numhidden" => 15,
        "factors" => 10,
        
        "opt" => "RMSProp",
        "eta" => 0.0005,
        "rho" => 0.99,
        "truncation" => 12,

        "gamma"=>0.99)

end

function get_ann(parsed, fs, env, rng)

    nh = parsed["numhidden"]
    na = length(get_actions(env))
    init_func = (dims...)->ActionRNNs.glorot_uniform(rng, dims...)

    if parsed["cell"] ∈ ActionRNNs.fac_rnn_types()

        rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
        factors = parsed["factors"]
        init_func = (dims...; kwargs...)->
            ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
        initb = (dims...; kwargs...) -> Flux.zeros(dims...)

        Flux.Chain(rnn(fs, na, nh, factors;
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

        Flux.Chain(rnn(fs, na, nh;
                init=init_func,
                initb=initb),
            Flux.Dense(nh, na; initW=init_func))


    else

        rnntype = getproperty(Flux, Symbol(parsed["cell"]))
        Flux.Chain(rnntype(fs, nh; init=init_func),
                   Flux.Dense(nh, na; initW=init_func))
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

    ActionRNNs.ControlOnlineAgent(chain,
                                  opt,
                                  τ,
                                  γ,
                                  fc,
                                  fs,
                                  ap)
end

function main_experiment(parsed=default_config(); working=false, progress=false)

    experiment_wrapper(parsed, working) do (parsed)

        num_steps = parsed["steps"]

        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = ActionRNNs.TMaze(parsed["size"])
        agent = construct_agent(env, parsed, rng)


        logger = SimpleLogger(
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
        while sum(logger.data.total_steps) <= num_steps
            usa = UpdateStateAnalysis(
                (l1 = 0.0f0, loss = 0.0f0, avg_loss = 1.0f0),
                Dict(
                    :l1 => l1_grad,
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
