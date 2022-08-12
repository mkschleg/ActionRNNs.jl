### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ a835fdc6-7096-451e-a0f4-982a5635f8b2
let
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ 331aa026-aa09-49d6-bc08-d0d75ae1d90d
using Reproduce, ActionRNNs, OpenAIGym

# ╔═╡ 5f5799bb-3691-42eb-93b3-aa5c230ceb42
using Plots

# ╔═╡ 2efb3b62-eb93-4bac-85fa-d2404bb1a4bc
Pkg.status()

# ╔═╡ Cell order:
# ╠═a835fdc6-7096-451e-a0f4-982a5635f8b2
# ╠═2efb3b62-eb93-4bac-85fa-d2404bb1a4bc
# ╠═331aa026-aa09-49d6-bc08-d0d75ae1d90d
# ╠═5f5799bb-3691-42eb-93b3-aa5c230ceb42
