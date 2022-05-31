module LunarLanderExperiment

# include("../src/ActionRNNs.jl")

import MinimalRLCore: MinimalRLCore, run_episode!, get_actions
import ActionRNNs: ActionRNNs, LunarLander, ExpUtils

import .ExpUtils: SimpleLogger, UpdateStateAnalysis, l1_grad, experiment_wrapper
import .ExpUtils: LunarLanderUtils as LMU, FluxUtils as FLU

import Flux
import Flux: gpu
import JLD2

# using ActionRNNs: TMaze

using Statistics
using Random
using ProgressMeter
using Reproduce
using Random

import LinearAlgebra: BLAS
#=
Time: 0:01:08
  episode:     100
  total_rews:  -249.31464
  loss:        146.8288
  l1:          0.6579351
  action:      4
  preds:       Float32[-18.375952, -17.984146, -17.7312, -17.009594]
=#
function default_config()
    Dict{String,Any}(
        "save_dir" => "tmp/lunar_lander",

        "seed" => 1,
        "steps" => 10000,

        "cell" => "MAGRU",
        "numhidden" => 64,

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


function build_ann(in, actions::Int, parsed, rng)
    
    nh = parsed["numhidden"]
    init_func, initb = ActionRNNs.get_init_funcs(rng)

    es = parsed["encoding_size"]

    deep_action = get(parsed, "deep", false)


    # construction dictated to maintain consistancy from previous experiments. Looks awkward. Is kinda awkward....
    encoding_network = if deep_action

        internal_a = parsed["internal_a"]
        
        rnn_layer = ActionRNNs.build_rnn_layer(es, internal_a, nh, parsed, rng)

        action_stream = Flux.Chain(
            (a)->Flux.onehotbatch(a, 1:actions),
            Flux.Dense(actions, internal_a, Flux.relu, initW=init_func),
        )

        obs_stream = Flux.Chain(
            Flux.Dense(in, es, Flux.relu, initW=init_func)
        )

        (ActionRNNs.DualStreams(action_stream, obs_stream), rnn_layer)
    else
        
        rnn_layer = ActionRNNs.build_rnn_layer(es, actions, nh, parsed, rng)
        
        (Flux.Dense(in, es, Flux.relu; initW=init_func), rnn_layer)
    end



    Flux.Chain(encoding_network...,
               Flux.Dense(nh, nh, Flux.relu; initW=init_func),
               Flux.Dense(nh, actions; initW=init_func))

    
end


function construct_agent(env, parsed, rng)

    fc = LMU.IdentityFeatureCreator()
    fs = MinimalRLCore.feature_size(fc, parsed["omit_states"])

    γ = Float32(parsed["gamma"])
    τ = parsed["truncation"]

    ap = ActionRNNs.ϵGreedyDecay((1.0, 0.05), 10000, 1000, MinimalRLCore.get_actions(env))

    opt = FLU.get_optimizer(parsed)

    chain = build_ann(fs, length(get_actions(env)), parsed, rng)# |> gpu

    ActionRNNs.DRQNAgent(chain,
                         opt,
                         τ,
                         ActionRNNs.QLearningSUM(γ),
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

function main_experiment(parsed = default_config(); progress=false, testing=false, overwrite=false)

    GC.gc()

    if "SLURM_CPUS_PER_TASK" ∈ keys(ENV)
        avail_cores = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
        BLAS.set_num_threads(avail_cores)
    elseif "ARNN_CPUS_PER_TASK" ∈ keys(ENV)
        avail_cores = parse(Int, ENV["ARNN_CPUS_PER_TASK"])
        BLAS.set_num_threads(avail_cores)
    end

    ll_experiment_wrapper(parsed; use_git_info=false, testing=testing, overwrite=overwrite) do parsed, save_setup_ret

        num_steps = parsed["steps"]

        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = ActionRNNs.LunarLander(seed, false, parsed["omit_states"], parsed["state_conditions"])
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
        checkpoint = 1
        while sum(logger.data.total_steps) <= num_steps
            if sum(logger.data.total_steps) > checkpoint * 500000
                # ExpUtils.save_results("$(data_dir)/results.jld2", logger.data)
                # d = copy(logger.data)
                # d["steps"] = checkpoint*500000
                Reproduce.save_results(parsed["_SAVE"], save_setup_ret, logger.data)

                GC.gc()
                checkpoint += 1
            end
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

# function ll_experiment_wrapper(exp_func::Function, parsed, working; overwrite=false)
#     savefile = ExpUtils.save_setup(parsed)
#     if isfile(savefile) && ActionRNNs.check_save_file_loadable(savefile) && !overwrite
#         return
#     end

#     data_dir = rsplit(savefile, "/"; limit=2)
#     ret = exp_func(parsed, data_dir[1])

#     if working
#         ret
#     elseif ret isa NamedTuple
#         ExpUtils.save_results(savefile, ret.save_results)
#     else
#         ExpUtils.save_results(savefile, ret)
#     end
# end

function ll_experiment_wrapper(exp_func::Function, parsed;
                               filter_keys=String[],
                               use_git_info=true,
                               hash_exclude_save_dir=true,
                               testing=false,
                               overwrite=false)

    SAVE_KEY = Reproduce.SAVE_KEY
    save_setup_ret = if SAVE_KEY ∉ keys(parsed)
        if isinteractive() 
            @warn "No arg at \"$(SAVE_KEY)\". Assume testing in repl." maxlog=1
            parsed[SAVE_KEY] = nothing
        elseif testing
            @warn "No arg at \"$(SAVE_KEY)\". Testing Flag Set." maxlog=1
            parsed[SAVE_KEY] = nothing
        else
            @error "No arg found at $(SAVE_KEY). Please use savetypes here."
        end
        nothing
    else
        save_setup_ret = Reproduce.save_setup(parsed;
                                    filter_keys=filter_keys,
                                    use_git_info=use_git_info,
                                    hash_exclude_save_dir=hash_exclude_save_dir)
        
        if Reproduce.check_experiment_done(parsed, save_setup_ret) && !overwrite
            Reproduce.post_save_setup(parsed[SAVE_KEY])
            return
        end
        save_setup_ret
    end

    Reproduce.post_save_setup(parsed[SAVE_KEY])

    ret = exp_func(parsed, save_setup_ret)

    if ret isa NamedTuple
        Reproduce.save_results(parsed[SAVE_KEY], save_setup_ret, ret.save_results)
    else
        Reproduce.save_results(parsed[SAVE_KEY], save_setup_ret, ret)
    end
    
    Reproduce.post_save_results(parsed[SAVE_KEY])
    
    if isinteractive() || testing
        ret
    end
end



end
