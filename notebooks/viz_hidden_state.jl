### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ ecc7ddb6-fef2-11eb-22a2-0d894ec19c99
begin
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ a3d9c068-e2cb-4de2-a138-098632d416ca
begin
	import Flux
	# import Flux.Tracker
	import JLD2
	import LinearAlgebra.Diagonal

	# include("../src/ActionRNNs.jl")
	import ActionRNNs
	import MinimalRLCore

	using DataStructures: CircularBuffer
	using ActionRNNs: RingWorld, step!, start!, glorot_uniform

	# using ActionRNNs
	using Statistics
	using Random
	using ProgressMeter
	using Reproduce
	using Random
	
	using Plots

	# using Plots

	const RWU = ActionRNNs.RingWorldUtils
	const FLU = ActionRNNs.FluxUtils
end

# ╔═╡ 9f82804b-b728-4c1c-9361-4d926ee78693
args = Dict(
	"agent"=>"new",
    "save_dir" => "ringworld",

    "seed" => 1,
    "steps" => 2000,
    "size" => 6,

	"cell" => "MARNN",
	"factors" => 10,
	"numhidden" => 6,
	"hs_learnable" => true,

	"action_factors" => 2,
	"out_factors" => 15,
	"in_factors" => 2,

	"outhorde" => "onestep",
	"outgamma" => 0.9,

	"opt" => "RMSProp",
	"eta" => 0.001,
	"rho" => 0.9,
	"truncation" => 3,

	"replay_size"=>1000,
	"warm_up" => 1000,
	"batch_size"=>4,
	"update_freq"=>4,
	"target_update_freq"=>1000,

	"synopsis" => false)

# ╔═╡ 746ead3a-4066-4790-9298-68d536bdaccf
run = false

# ╔═╡ ddcadc4d-0689-4339-b2c4-873e777748cc
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

# ╔═╡ 422d05d5-3e76-49cb-8ec4-eff1ebfe31de
results_synopsis(res, ::Val{false}) = res

# ╔═╡ 72e9012c-56d7-4a71-a07d-8a1cbd795841
function get_model(parsed, out_horde, fs, rng)

    nh = parsed["numhidden"]
    na = 2
    init_func = (dims...)->glorot_uniform(rng, dims...)
    num_gvfs = length(out_horde)

    chain = begin
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
                       Flux.Dense(nh, num_gvfs; initW=init_func))

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
        elseif parsed["cell"] ∈ ActionRNNs.gated_rnn_types()

        rnn = getproperty(ActionRNNs, Symbol(parsed["cell"]))

        ninternal = parsed["internal"]

        init_func = (dims...; kwargs...)->
            ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
        initb = (dims...; kwargs...) -> Flux.zeros(dims...)

        m = Flux.Chain(
            rnn(fs, na, ninternal, nh;
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

    chain
end

# ╔═╡ 554a1bf7-a7a9-4ee4-95c2-a05552370087
function construct_new_agent(parsed, rng)

    fc = RWU.StandardFeatureCreator{parsed["action_features"]}()
    fs = size(fc)

    out_horde = RWU.get_horde(parsed, "out")
    τ = parsed["truncation"]

    ap = ActionRNNs.RandomActingPolicy([0.5, 0.5])

    opt = FLU.get_optimizer(parsed)

    chain = get_model(parsed, out_horde, fs, rng)

    ActionRNNs.DRTDNAgent(out_horde,
                          chain,
                          opt,
                          τ,
                          fc,
                          fs,
                          1,
                          parsed["replay_size"],
                          parsed["warm_up"],
                          parsed["batch_size"],
                          parsed["update_freq"],
                          parsed["target_update_freq"],
                          ap,
                          parsed["hs_learnable"])

end

# ╔═╡ 4369fb39-267b-4186-8ffc-0f0e933d5f0a
# Creating an environment for to run in jupyter.
function experiment_loop(env, agent, outhorde_str, num_steps, rng; prgs=false)

    out_pred_strg = zeros(Float32, num_steps, length(agent.horde))
    out_err_strg = zeros(Float32, num_steps, length(agent.horde))
    out_loss_strg = zeros(Float32, num_steps)

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
    
    cur_step = 1
    MinimalRLCore.run_episode!(env, agent, num_steps, rng) do (s, a, s′, r)
        
        out_preds = a.preds
        
        out_pred_strg[cur_step, :] .= out_preds[:,1]
        out_err_strg[cur_step, :] = out_pred_strg[cur_step, :] .- RWU.oracle(env, outhorde_str);
        if !(a.update_state isa Nothing)
            out_loss_strg[cur_step] = a.update_state.loss
        end
        
        if prgs
            ProgressMeter.next!(prg_bar)
        end

        cur_step += 1
    end
    
    out_pred_strg, out_err_strg, out_loss_strg
    
end

# ╔═╡ 7b167309-0f1c-4302-a567-1c92ab345dff
function main_experiment(parsed=default_args(); working=false, progress=false, overwrite=false)
    
    ActionRNNs.experiment_wrapper(parsed, working, overwrite=overwrite) do (parsed)
        
        num_steps = parsed["steps"]
        seed = parsed["seed"]
        rng = Random.MersenneTwister(seed)
        
        env = RingWorld(parsed)
        agent = construct_new_agent(parsed, rng)

        out_pred_strg, out_err_strg =
            experiment_loop(env, agent, parsed["outhorde"], num_steps, rng; prgs=progress)
        
        results = Dict(["pred"=>out_pred_strg, "err"=>out_err_strg])
        save_results = results_synopsis(results, Val(parsed["synopsis"]))
        (save_results=save_results)
    end
end

# ╔═╡ 6d3d2c97-6212-487b-b316-04534afe748e
ret = if run
	main_experiment(args; progress=true, working=true)
else
	try
		ret
	catch
		nothing
	end
end

# ╔═╡ 9271215f-df53-4af0-b4db-6272045c98c5
plot(plot((ret["err"].^2)[:, 1]), plot((ret["err"].^2)[:, 2]))

# ╔═╡ Cell order:
# ╠═ecc7ddb6-fef2-11eb-22a2-0d894ec19c99
# ╠═a3d9c068-e2cb-4de2-a138-098632d416ca
# ╠═9f82804b-b728-4c1c-9361-4d926ee78693
# ╠═746ead3a-4066-4790-9298-68d536bdaccf
# ╠═6d3d2c97-6212-487b-b316-04534afe748e
# ╠═9271215f-df53-4af0-b4db-6272045c98c5
# ╠═ddcadc4d-0689-4339-b2c4-873e777748cc
# ╠═422d05d5-3e76-49cb-8ec4-eff1ebfe31de
# ╠═72e9012c-56d7-4a71-a07d-8a1cbd795841
# ╠═554a1bf7-a7a9-4ee4-95c2-a05552370087
# ╠═7b167309-0f1c-4302-a567-1c92ab345dff
# ╟─4369fb39-267b-4186-8ffc-0f0e933d5f0a
