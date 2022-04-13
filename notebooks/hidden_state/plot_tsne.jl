### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 8e6e15f0-bb8a-11ec-394f-43e8c4cec434
let
	import Pkg
	Pkg.activate(joinpath(@__DIR__, "../"))
end

# ╔═╡ 50a92605-2ffe-44fe-aa2a-63ebffb61dc6
using Revise, Plots, RollingFunctions, TOML, FileIO

# ╔═╡ 9fb7189b-3c83-4b89-b742-eaeb94214c2f
using ActionRNNs, Random

# ╔═╡ 10e89e56-9a18-40ac-8267-f2bc79b5d523
import ActionRNNs: @data

# ╔═╡ 3c91f02d-2186-4ae7-90e4-d3074cb2276f
color_scheme = [
    colorant"#44AA99",
    colorant"#332288",
    colorant"#DDCC77",
    colorant"#999933",
    colorant"#CC6677",
    colorant"#AA4499",
    colorant"#DDDDDD",
	colorant"#117733",
	colorant"#882255",
	colorant"#1E90FF",
]

# ╔═╡ 3905193b-65d7-4172-ad93-ef83a78aba51
function ActionRNNs.ChoosyDataLoggers.process_data(::Val{:get_hs_1_layer}, data)
	data[collect(keys(data))[1]]
end

# ╔═╡ e5765332-067f-4cd9-b16d-ad486e6ddc6d


# ╔═╡ 90de955a-0aae-47f7-8ebd-d295df1967d2
base_env_loc, toml_file = "../../", "final_runs/ringworld_er_10.toml"

# ╔═╡ 900e3335-6993-4978-94f4-0b71f92524e0
config, static, args = let
	dict = TOML.parsefile(joinpath(base_env_loc, toml_file))
	args = FileIO.load(joinpath(base_env_loc, dict["config"]["arg_file"]))
	dict["config"], dict["static_args"], args
end

# ╔═╡ 7eedd8ec-9eca-4ef6-8a69-d301ba88741a
include(joinpath(base_env_loc, config["exp_file"]))

# ╔═╡ 8a9b1424-a0b0-4693-a7c9-c78e904192de
config

# ╔═╡ 4e68a0ed-e38f-4b3e-b038-b158ad6967ab


# ╔═╡ Cell order:
# ╠═8e6e15f0-bb8a-11ec-394f-43e8c4cec434
# ╠═50a92605-2ffe-44fe-aa2a-63ebffb61dc6
# ╠═9fb7189b-3c83-4b89-b742-eaeb94214c2f
# ╠═10e89e56-9a18-40ac-8267-f2bc79b5d523
# ╠═3c91f02d-2186-4ae7-90e4-d3074cb2276f
# ╠═3905193b-65d7-4172-ad93-ef83a78aba51
# ╠═e5765332-067f-4cd9-b16d-ad486e6ddc6d
# ╠═90de955a-0aae-47f7-8ebd-d295df1967d2
# ╠═900e3335-6993-4978-94f4-0b71f92524e0
# ╠═8a9b1424-a0b0-4693-a7c9-c78e904192de
# ╠═7eedd8ec-9eca-4ef6-8a69-d301ba88741a
# ╠═4e68a0ed-e38f-4b3e-b038-b158ad6967ab
