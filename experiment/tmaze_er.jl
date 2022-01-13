module TMazeERExperiment

import Flux
import JLD2

import MinimalRLCore: MinimalRLCore, run_episode!, get_actions
import ActionRNNs: ActionRNNs, TMaze, ExpUtils, DRQNAgent

import .ExpUtils: experiment_wrapper, TMazeUtils, FluxUtils
import .ExpUtils: SimpleLogger, UpdateStateAnalysis, l1_grad

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

const TMU = TMazeUtils
const FLU = FluxUtils

#=
Time: 0:01:19
  episode:    6455
  successes:  0.9600399600399601
  loss:       0.990142
  l1:         0.000502145
  action:     2
  preds:      Float32[0.3016336, 3.6225605, -2.5592222, 1.884988]
  grad:       0.0
=#
function default_config()
    
    Dict{String,Any}(
        "save_dir" => "tmp/tmaze_er",

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

function build_deep_action_rnn_layers(in, actions, out, parsed, rng)

    # Deep actions for RNNs from Zhu et al 2018
    internal_a = parsed["internal_a"]
    internal_o = parsed["internal_o"]
    
    init_func, initb = ActionRNNs.get_init_funcs(rng)
    
    action_stream = Flux.Chain(
        (a)->Flux.onehotbatch(a, 1:actions),
        Flux.Dense(actions, internal_a, Flux.relu, initW=init_func),
    )

    obs_stream = identity
        # Flux.Dense(in, internal_o, Flux.relu, initW=init_func)
    # )
    
    (ActionRNNs.DualStreams(action_stream, obs_stream),
     ActionRNNs.build_rnn_layer(internal_a, internal_o, out, parsed, rng))
end

function build_ann(in, actions, parsed, rng)
    
    nh = parsed["numhidden"]
    init_func, initb = ActionRNNs.get_init_funcs(rng)

    deep_action = get(parsed, "deep", false)
    rnn = if deep_action
        build_deep_action_rnn_layers(in, actions, nh, parsed, rng)
    else
        (ActionRNNs.build_rnn_layer(in, actions, nh, parsed, rng),)
    end

    Flux.Chain(rnn...,
               Flux.Dense(nh, actions; initW=init_func))
    
end

function construct_agent(env, parsed, rng)

    fc = TMU.StandardFeatureCreator{false}()
    fs = MinimalRLCore.feature_size(fc)

    γ = Float32(parsed["gamma"])
    τ = parsed["truncation"]


    ap = ActionRNNs.ϵGreedy(0.1, MinimalRLCore.get_actions(env))

    opt = FLU.get_optimizer(parsed)

    chain = build_ann(fs, length(get_actions(env)), parsed, rng)

    DRQNAgent(chain,
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


    if "numhidden_factors" ∈ keys(parsed)
        parsed["numhidden"] = parsed["numhidden_factors"][1]
        parsed["factors"] = parsed["numhidden_factors"][2]
    end

    experiment_wrapper(parsed, working) do parsed

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

        mean_loss = 1.0f0
        
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
