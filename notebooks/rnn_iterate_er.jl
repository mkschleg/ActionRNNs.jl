### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# ╔═╡ fe6ef0da-1f8d-11eb-3ba2-152d65041ed4
using Revise, Statistics, Plots, RollingFunctions, ActionRNNs, Flux

# ╔═╡ 32d54568-1f8e-11eb-3c8c-a99c6ab3a419
include("../experiment/ringworld.jl")

# ╔═╡ 2e18c3a6-1f8e-11eb-15dc-85b4ab2d46d0
import Random

# ╔═╡ 8798b954-1f8e-11eb-03af-cf1e5e25bc7c
RWU = ActionRNNs.RingWorldUtils

# ╔═╡ 3c00d170-1f8e-11eb-1900-0b0d9fb3cbef
function construct_agent(outhorde, fc, rnn, opt, τ)


    fs = MinimalRLCore.feature_size(fc)
    
    ap = ActionRNNs.RandomActingPolicy([0.5, 0.5])

    ActionRNNs.PredERAgent(outhorde,
                           rnn,
                           opt,
                           τ,
                           fc,
                           fs,
                           1, 
                           128, τ, 4, #Replay details
                           ap)
end

# ╔═╡ 6b5ebf7c-1f8e-11eb-3e2e-8b9b3db75a7c
out_pred, out_err, out_loss = begin 
	fc = RWU.OneHotFeatureCreator()
	fs = MinimalRLCore.feature_size(fc)
	num_hidden = 12
	outhorde = RWU.gammas_term(0.0:0.1:0.9)

	Random.seed!(10392)

	rnn = Chain(
    	ARNN(fs, 2, num_hidden), 
	#     Dense(num_hidden, num_hidden, relu),
    	Dense(num_hidden, length(outhorde)))
	opt = ADAM(0.0005)
	τ = 5

	agent = construct_agent(outhorde, fc, rnn, opt, τ)
	env = RingWorld(12)

	RingWorldExperiment.experiment_loop(env, agent, "gammas_term", 750000, Random.GLOBAL_RNG; prgs=true)
end

# ╔═╡ 552fca64-1f95-11eb-0951-956270c79c03
plot(rollmean(sqrt.(mean(out_err.^2;dims=2)[:,1]), 1000)[1:1000:750000])

# ╔═╡ Cell order:
# ╠═fe6ef0da-1f8d-11eb-3ba2-152d65041ed4
# ╠═2e18c3a6-1f8e-11eb-15dc-85b4ab2d46d0
# ╠═32d54568-1f8e-11eb-3c8c-a99c6ab3a419
# ╠═8798b954-1f8e-11eb-03af-cf1e5e25bc7c
# ╠═3c00d170-1f8e-11eb-1900-0b0d9fb3cbef
# ╠═6b5ebf7c-1f8e-11eb-3e2e-8b9b3db75a7c
# ╠═552fca64-1f95-11eb-0951-956270c79c03
