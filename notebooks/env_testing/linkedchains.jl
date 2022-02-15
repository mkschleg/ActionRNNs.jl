### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ 46f0c75a-7ed0-11ec-2b0d-6909f014b867
let
	import Pkg
	Pkg.activate("../../") # ActionRNNs project.
end

# ╔═╡ 4b644d8e-80d4-4aee-8ab5-eab39d0d41cf
using Revise, ActionRNNs #, Plots, PlutoUI

# ╔═╡ 89ddd0e1-680a-4502-810f-5a6203f41fc4
env = ActionRNNs.LinkedChains{:TERM}(5, 4, 2)

# ╔═╡ 1bc3dcb9-52e3-4c68-96da-800313664b5c
ActionRNNs.MinimalRLCore.start!(env)

# ╔═╡ d753dd60-32fe-4cf9-8152-7ba50a6eb8a0
action = 3

# ╔═╡ 93280d30-3e13-46ca-a3d3-f3903e192a06
begin
	ret = ActionRNNs.MinimalRLCore.step!(env, action)
	ret, env
end

# ╔═╡ Cell order:
# ╠═46f0c75a-7ed0-11ec-2b0d-6909f014b867
# ╠═4b644d8e-80d4-4aee-8ab5-eab39d0d41cf
# ╠═89ddd0e1-680a-4502-810f-5a6203f41fc4
# ╠═1bc3dcb9-52e3-4c68-96da-800313664b5c
# ╠═d753dd60-32fe-4cf9-8152-7ba50a6eb8a0
# ╠═93280d30-3e13-46ca-a3d3-f3903e192a06
