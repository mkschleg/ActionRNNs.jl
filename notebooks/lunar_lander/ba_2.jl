### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ c89ee45a-1a82-11ed-2676-579a7ac16da4
let
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ 14716316-045a-40a0-aa99-689b5437af67
using Reproduce, ActionRNNs

# ╔═╡ 1d1def3e-03cb-4f94-9f66-abafb100cccd
using Plots

# ╔═╡ 8b32d68e-d230-4bd5-a318-fe8798970753
env = ActionRNNs.LunarLander(1)

# ╔═╡ d1c57cd3-6918-4ae4-a52c-e7d3f7d7fb3f
# plot([Colors.RGB((env.gym.pyenv.render("rgb_array")[i, j, :]./255)...) for i in 1:400 for j in 1:600])

# ╔═╡ abaa2fa8-f8a6-4024-a647-ed12232b5848
let
	arr = env.gym.pyenv.render("rgb_array")
	[Colors.RGB((arr[i, j, :]./255)...) for i in 1:400 for j in 1:600]
end

# ╔═╡ Cell order:
# ╠═c89ee45a-1a82-11ed-2676-579a7ac16da4
# ╠═14716316-045a-40a0-aa99-689b5437af67
# ╠═1d1def3e-03cb-4f94-9f66-abafb100cccd
# ╠═8b32d68e-d230-4bd5-a318-fe8798970753
# ╠═d1c57cd3-6918-4ae4-a52c-e7d3f7d7fb3f
# ╠═abaa2fa8-f8a6-4024-a647-ed12232b5848
