module DirectionalTMazeERExperiment

# include("../src/ActionRNNs.jl")

import Flux
import JLD2

import MinimalRLCore: MinimalRLCore, run_episode!, get_actions
import ActionRNNs: ActionRNNs, DirectionalTMaze, ExpUtils

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
Default performance:

Time: 0:02:28
  episode:    5385
  successes:  0.8351648351648352
  loss:       1.0
  l1:         0.0
  action:     2
  preds:      Float32[0.369189, 0.48326853, 0.993273]

=#
function default_config()
    Dict{String,Any}(
        "save_dir" => "tmp/dir_tmaze_er",

        "seed" => 2,
        "steps" => 150000,
        "size" => 10,

        "cell" => "MAGRU",
        "numhidden" => 10,

        "opt" => "RMSProp",
        "eta" => 0.0005,
        "rho" =>0.99,

        "replay_size"=>20000,
        "warm_up" => 1000,
        "batch_size"=>8,
        "update_wait"=>4,
        "target_update_wait"=>1000,
        "truncation" => 12,

        "hs_learnable" => true,
        
        "gamma"=>0.99)
end


function working_config()
    Dict{String,Any}(
        "save_dir" => "tmp/dir_tmaze_er",

        "seed" => 2,
        "steps" => 150000,
        "size" => 10,

        "cell" => "MAGRU",
        "numhidden" => 10,

        "deep" => false,
        "internal_a" => 5,
        "internal_o" => 20,

        "opt" => "RMSProp",
        "eta" => 0.0005,
        "rho" =>0.99,

        "replay_size"=>20000,
        "warm_up" => 1000,
        "batch_size"=>8,
        "update_wait"=>4,
        "target_update_wait"=>1000,
        "truncation" => 12,

        "hs_learnable" => true,
        
        "gamma"=>0.99)

    
end


function get_init_funcs()
    init_func = (dims...; kwargs...)->
        ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
    initb = (dims...; kwargs...) -> Flux.zeros(dims...)
    init_func, initb
end

get_ann(parsed, fs, env, rng) = get_ann(parsed, fs, length(get_actions(env)), rng)

get_rnn_construct_trait(rnn_type) = @error "$(rnn_type) not supported! Please implement `construct_rnn_layer`"

function construct_rnn_layer(in, actions, out, parsed, rng)
    rnn_type = try
        getproperty(ActionRNNs, Symbol(parsed["cell"]))
    catch
        getproperty(Flux, Symbol(parsed["cell"]))
    end

    deep_action = get(parsed, "deep", false)
    if deep_action
        construct_deep_action_rnn_layers(rnn_type, in, actions, out, parsed, rng)
    else
        construct_rnn_layer(rnn_type, in, actions, out, parsed, rng)
    end
end

construct_rnn_layer(rnn_type, in, actions, out, parsed, rng) =
    construct_rnn_layer(get_rnn_build_trait(rnn_type), rnn_type, in, actions, out, parsed, rng)

function construct_deep_action_rnn_layers(rnn_type, in, actions, out, parsed, rng)
    # Deep actions for RNNs from Zhu et al 2018
    
    internal_a = parsed["internal_a"]
    internal_o = parsed["internal_o"]
    
    init_func, initb = get_init_funcs()
    
    action_stream = Flux.Chain(
        (a)->Flux.onehotbatch(a, 1:na),
        Flux.Dense(na, internal_a, Flux.relu, initW=init_func),
    )

    obs_stream = Flux.Chain(
        Flux.Dense(fs, internal_o, Flux.relu, initW=init_func)
    )
    
    (ActionRNNs.DualStreams(action_stream, obs_stream),
     construct_rnn_layer(rnn_type, internal_o, internal_a, out, parsed, rng))
end


struct BuildDeepAction{T}
    rnn_construct::T
end

struct BuildFactored end

get_rnn_construct_trait(::typeof(ActionRNNs.FacMARNN)) = BuildFactored()
get_rnn_construct_trait(::typeof(ActionRNNs.FacMAGRU)) = BuildFactored()

struct BuildTucFactored end

get_rnn_construct_trait(::typeof(ActionRNNs.FacTucMAGRU)) = BuildTucFactored()
get_rnn_construct_trait(::typeof(ActionRNNs.FacTucMARNN)) = BuildTucFactored()

struct BuildActionRNN end

for cell_func in [ActionRNNs.AARNN, ActionRNNs.MARNN, ActionRNNs.AAGRU, ActionRNNs.MAGRU, ActionRNNs.AALSTM, ActionRNNs.MALSTM]
    exp = quote
        function get_rnn_construct_trait(::typeof($cell_func))
            BuildActionRNN()
        end
    end
    eval(exp)
end

struct BuildComboCat end

get_rnn_construct_trait(::typeof(ActionRNNs.CcatRNN)) = BuildComboCat()
get_rnn_construct_trait(::typeof(ActionRNNs.CcatGRU)) = BuildComboCat()

struct BuildComboAdd end

for cell_func in [ActionRNNs.CaddRNN, ActionRNNs.CaddGRU, ActionRNNs.CaddAAGRU, ActionRNNs.CaddMAGRU, ActionRNNs.CaddElRNN]
    exp = quote
        function get_rnn_construct_trait(::typeof($cell_func))
            BuildComboAdd()
        end
    end
    eval(exp)
end

struct BuildMixed end

for cell_func in [ActionRNNs.MixRNN, ActionRNNs.MixElRNN, ActionRNNs.MixElGRU, ActionRNNs.MixGRU]
    exp = quote
        function get_rnn_construct_trait(::typeof($cell_func))
            BuildMixed()
        end
    end
    eval(exp)
end


struct BuildFlux end

for cell_func in [Flux.RNN, Flux.GRU, Flux.LSTM]
        exp = quote
        function get_rnn_construct_trait(::typeof($cell_func))
            BuildFlux()
        end
    end
    eval(exp)
end


function construct_rnn_layer(::BuildFactored, rnn_type, in, actions, out, parsed, rng)

    factors = parsed["factors"]
    init_style = get(parsed, "init_style", "standard")

    init_func, initb = get_init_funcs()

    
    rnn_type(in, actions, out, factors;
             init_style=init_style,
             init=init_func,
             initb=initb)

end

function construct_rnn_layer(::BuildTucFactored, rnn_type, in, actions, out, parsed, rng)

    action_factors = parsed["action_factors"]
    out_factors = parsed["out_factors"]
    in_factors = parsed["in_factors"]
    init_style = get(parsed, "init_style", "standard")

    init_func, initb = get_init_funcs()

    rnn_type(in, actions, out, action_factors, out_factors, in_factors;
             init_style=init_style,
             init=init_func,
             initb=initb)
     
end

function construct_rnn_layer(::Union{BuildActionRNN, BuildComboAdd}, rnn_type, in, actions, out, parsed, rng)
    init_func, initb = get_init_funcs()
    rnn_type(in, actions, out;
             init=init_func,
             initb=initb)
end

function construct_rnn_layer(::BuildComboCat, rnn_type, in, actions, out, parsed, rng)

    @assert out % 2 == 0
    
    init_func, initb = get_init_funcs()
    rnn_type(in, actions, out÷2;
             init=init_func,
             initb=initb)
end

function construct_rnn_layer(::BuildComboAdd, rnn_type, in, actions, out, parsed, rng)

    rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
    
    init_func, initb = get_init_funcs()
    
    rnn_type(in, actions, out;
        init=init_func,
        initb=initb)
    
end

function construct_rnn_layer(::BuildMixed, rnn_type, in, actions, out, parsed, rng)

    init_func, initb = get_init_funcs()
    ne = parsed["num_experts"]
    rnn_type(fs, na, nh, ne;
        init=init_func,
        initb=initb)

end

function construct_rnn_layer(::BuildFlux, rnn_type, in, actions, out, parsed, rng)
    init_func, initb = get_init_funcs()
    rnn_type(fs, nh; init=init_func)
end

function get_ann(parsed, in, actions, rng)
    
    nh = parsed["numhidden"]
    init_func, initb = get_init_funcs()

    rnn = construct_rnn_layer(in, actions, nh, parsed, rng)

    if rnn isa Tuple
        Flux.Chain(rnn...,
                   Flux.Dense(nh, actions; initW=init_func))
    else
        Flux.Chain(rnn,
                   Flux.Dense(nh, actions; initW=init_func))
    end
    
end


# function get_ann(parsed, fs::Int, na::Int, rng)

#     nh = parsed["numhidden"]
#     # na = length(get_actions(env))
#     init_func = (dims...)->ActionRNNs.glorot_uniform(rng, dims...)
    

#     if parsed["cell"] ∈ ActionRNNs.fac_rnn_types()

#         rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
#         factors = parsed["factors"]
#         init_style = get(parsed, "init_style", "standard")
        
#         init_func = (dims...; kwargs...)->
#             ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
#         initb = (dims...; kwargs...) -> Flux.zeros(dims...)
        
#         Flux.Chain(rnn(fs, na, nh, factors;
#                        init_style=init_style,
#                        init=init_func,
#                        initb=initb),
#                    Flux.Dense(nh, na; initW=init_func))

#      elseif parsed["cell"] ∈ ActionRNNs.fac_tuc_rnn_types()

#         rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
#         action_factors = parsed["action_factors"]
#         out_factors = parsed["out_factors"]
#         in_factors = parsed["in_factors"]
#         init_style = get(parsed, "init_style", "standard")

#         init_func = (dims...; kwargs...)->
#             ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
#         initb = (dims...; kwargs...) -> Flux.zeros(dims...)

#         Flux.Chain(rnn(fs, na, nh, action_factors, out_factors, in_factors;
#                        init_style=init_style,
#                        init=init_func,
#                        initb=initb),
#                    Flux.Dense(nh, na; initW=init_func))
        
#     elseif parsed["cell"] ∈ ActionRNNs.rnn_types() && !get(parsed, "deep", false)

#         rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
        
#         init_func = (dims...; kwargs...)->
#             ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
#         initb = (dims...; kwargs...) -> Flux.zeros(dims...)
        
#         m = Flux.Chain(
#             rnn(fs, na, nh;
#                 init=init_func,
#                 initb=initb),
#             Flux.Dense(nh, na; initW=init_func))

#     elseif parsed["cell"] ∈ ActionRNNs.combo_add_rnn_types() && !get(parsed, "deep", false)

#         rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
        
#         init_func = (dims...; kwargs...)->
#             ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
#         initb = (dims...; kwargs...) -> Flux.zeros(dims...)
        
#         m = Flux.Chain(
#             rnn(fs, na, nh;
#                 init=init_func,
#                 initb=initb),
#             Flux.Dense(nh, na; initW=init_func))

#     elseif parsed["cell"] ∈ ActionRNNs.combo_cat_rnn_types() && !get(parsed, "deep", false)

#         rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
        
#         init_func = (dims...; kwargs...)->
#             ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
#         initb = (dims...; kwargs...) -> Flux.zeros(dims...)
        
#         m = Flux.Chain(
#             rnn(fs, na, nh;
#                 init=init_func,
#                 initb=initb),
#             Flux.Dense(nh*2, na; initW=init_func))

#     elseif parsed["cell"] ∈ ActionRNNs.gated_rnn_types()

#         rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))

#         ninternal = parsed["internal"]

#         init_func = (dims...; kwargs...)->
#             ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
#         initb = (dims...; kwargs...) -> Flux.zeros(dims...)

#         m = Flux.Chain(
#             rnn(fs, na, ninternal, nh;
#                 init=init_func,
#                 initb=initb),
#             Flux.Dense(nh, na; initW=init_func))
        
#     elseif parsed["cell"]  ∈ ActionRNNs.rnn_types() && get(parsed, "deep", false)

#         # Deep actions for RNNs from Zhu et al 2018
        
#         rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))
        
#         internal_a = parsed["internal_a"]
#         internal_o = parsed["internal_o"]
        
#         init_func = (dims...; kwargs...)->
#             ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
#         initb = (dims...; kwargs...) -> Flux.zeros(dims...)

#         action_stream = Flux.Chain(
#             (a)->Flux.onehotbatch(a, 1:na),
#             Flux.Dense(na, internal_a, Flux.relu, initW=init_func),
#         )

#         obs_stream = Flux.Chain(
#             Flux.Dense(fs, internal_o, Flux.relu, initW=init_func)
#         )
        
#         m = Flux.Chain(
#             ActionRNNs.DualStreams(action_stream, obs_stream),
#             rnn(internal_o, internal_a, nh;
#                 init=init_func,
#                 initb=initb),
#             Flux.Dense(nh, na; initW=init_func))
        
#     elseif parsed["cell"] ∈ ActionRNNs.mixture_rnn_types()

#         rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))

#         init_func = (dims...; kwargs...)->
#             ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
#         initb = (dims...; kwargs...) -> Flux.zeros(dims...)

#         ne = parsed["num_experts"]
        
#         m = Flux.Chain(
#             rnn(fs, na, nh, ne;
#                 init=init_func,
#                 initb=initb),
#             Flux.Dense(nh, na; initW=init_func))
        
        
#     else
        
#         rnntype = getproperty(Flux, Symbol(parsed["cell"]))
#         Flux.Chain(rnntype(fs, nh; init=init_func),
#                    Flux.Dense(nh,
#                               na;
#                               initW=init_func))
        
#     end
# end

function construct_agent(env, parsed, rng)

    fc = TMU.StandardFeatureCreator{false}()
    fs = MinimalRLCore.feature_size(fc)

    γ = Float32(parsed["gamma"])
    τ = parsed["truncation"]


    ap = ActionRNNs.ϵGreedy(0.1, MinimalRLCore.get_actions(env))

    opt = FLU.get_optimizer(parsed)

    chain = get_ann(parsed, fs, length(get_actions(env)), rng)

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
    
    experiment_wrapper(parsed, working) do parsed

        num_steps = parsed["steps"]

        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = DirectionalTMaze(parsed["size"])
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
                                               (:preds, a.preds)])
                                               # (:grad, a.update_state isa Nothing ? 0.0f0 : sum(a.update_state.grads[agent.model[1].cell.Wh]))])
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
        (;agent = agent, save_results = save_results)
    end
end



end
