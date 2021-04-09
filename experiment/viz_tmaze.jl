module ImageTMazeERExperiment

using Flux
import JLD2
import LinearAlgebra.Diagonal
import MinimalRLCore
using MinimalRLCore: run_episode!, get_actions
import ActionRNNs

using ActionRNNs: ImageTMaze

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
        "steps" => 200000,
        "size" => 6,

        "cell" => "ARNN",
        "numhidden" => 6,

        "opt" => "ADAM",
        "eta" => 0.0005,
        # "rho" =>0.99,

        "replay_size"=>4000,
        "warm_up" => 1000,
        "batch_size"=>16,
        "update_freq"=>4,
        "tn_update_freq"=>1000,
        "truncation" => 8,

        "hs_learnable" => true,
        
        "gamma"=>0.99)

    
end

function get_ann(parsed, image_dims, rng)

    init_func = (dims...; kwargs...)->ActionRNNs.glorot_uniform(rng, dims...; kwargs...)

    cl = Flux.Conv((4, 4), 1 => 4, relu; stride=2, init=init_func)
    fs = prod(Flux.outdims(cl, image_dims))
    println(fs)
    nh = parsed["numhidden"]
    
    if parsed["cell"] == "FacARNN"

        factors = parsed["factors"]
        Flux.Chain(
            cl,
            Flux.flatten,
            ActionRNNs.FacARNN(fs, 4, nh, factors; init=init_func),
            Flux.Dense(nh, 4; initW=init_func))
        
    elseif parsed["cell"] == "ARNN"
        
        Flux.Chain(
            cl,
            Flux.flatten,
            ActionRNNs.MARNN(fs, 4, nh;
                            init=init_func),
            Flux.Dense(nh, 4; initW=init_func))
        
    elseif parsed["cell"] == "RNN"
        
        Flux.Chain(
            cl,
            Flux.flatten,
            Flux.RNN(fs, nh;
                     init=init_func),
                           # hs_learnable=parsed["hs_learnable"]),
            Flux.Dense(nh, 4; initW=init_func))
        
    else
        
        rnntype = getproperty(Flux, Symbol(parsed["cell"]))
        Flux.Chain(
            cl, Flux.flatten,
            rnntype(fs, nh; init=init_func),
            Flux.Dense(nh,
                       4;
                       initW=init_func))
        
    end
end

function construct_agent(env, parsed, rng)

    fc = if parsed["cell"] ∈ ["FacARNN", "ARNN"]
        TMU.StandardFeatureCreator{false}()
    else
        TMU.StandardFeatureCreator{true}()
    end
    fs = MinimalRLCore.feature_size(fc)

    γ = Float32(parsed["gamma"])
    τ = parsed["truncation"]


    # ap = ActionRNNs.ϵGreedy(0.1, MinimalRLCore.get_actions(env))
    ap = ActionRNNs.ϵGreedyDecay((1.0, 0.1), 10000, 1000, MinimalRLCore.get_actions(env))

    opt = FLU.get_optimizer(parsed)
    chain = get_ann(parsed, (28,28, 1, 1), rng) |> gpu

    ActionRNNs.ImageDRQNAgent(chain,
                              opt,
                              τ,
                              γ,
                              (28, 28, 1),
                              UInt8,
                              parsed["replay_size"],
                              parsed["warm_up"],
                              parsed["batch_size"],
                              parsed["update_freq"],
                              parsed["tn_update_freq"],
                              ap,
                              parsed["hs_learnable"])
end

function main_experiment(parsed::Dict=default_config(); working=false, progress=false, verbose=false)


    ActionRNNs.experiment_wrapper(parsed, working) do parsed

        num_steps = parsed["steps"]

        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = ImageTMaze(parsed["size"])
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



end
