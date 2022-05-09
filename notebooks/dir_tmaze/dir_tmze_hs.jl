### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 670987a1-1480-4f5f-a5bd-887912014da0
let
	import Pkg; Pkg.activate("..")
end

# ╔═╡ 7add7148-a3bc-11eb-1742-cd08284e30d3
using Reproduce, Plots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 09428771-fa7b-49c9-b357-62c892101c10
Pkg.status()

# ╔═╡ cdfbce1a-bbc0-43d4-b85b-065bf2a6aad8
function ingredients(path::String)
        # this is from the Julia source code (evalfile in base/loading.jl)
        # but with the modification that it returns the module instead of the last object
        name = Symbol(basename(path))
        m = Module(name)
        Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
			 :(include($path))))
        m
end

# ╔═╡ f4b71e55-71fa-407f-998f-b752eec926c2
DTM = ingredients("../experiment/dir_tmaze_er.jl")

# ╔═╡ f5aaf5ba-cf08-477e-926e-4199dcc60c32
args = let
	args = DTM.DirectionalTMazeERExperiment.default_config()
	args["cell"] = "CaddRNN"
	# args["eta"] = 0.5e-7
	args
end

# ╔═╡ 483e3819-a5f5-44af-9bf9-47cd9d069715
ret = DTM.DirectionalTMazeERExperiment.main_experiment(args, working=true, progress=false)

# ╔═╡ Cell order:
# ╠═670987a1-1480-4f5f-a5bd-887912014da0
# ╠═09428771-fa7b-49c9-b357-62c892101c10
# ╠═7add7148-a3bc-11eb-1742-cd08284e30d3
# ╠═cdfbce1a-bbc0-43d4-b85b-065bf2a6aad8
# ╠═f4b71e55-71fa-407f-998f-b752eec926c2
# ╠═f5aaf5ba-cf08-477e-926e-4199dcc60c32
# ╠═483e3819-a5f5-44af-9bf9-47cd9d069715
