### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 7add7148-a3bc-11eb-1742-cd08284e30d3
using Reproduce, Plots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ f4b71e55-71fa-407f-998f-b752eec926c2
DTM = include("../experiment/dir_tmaze_er.jl")

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

# ╔═╡ f5aaf5ba-cf08-477e-926e-4199dcc60c32


# ╔═╡ 483e3819-a5f5-44af-9bf9-47cd9d069715


# ╔═╡ Cell order:
# ╠═7add7148-a3bc-11eb-1742-cd08284e30d3
# ╠═cdfbce1a-bbc0-43d4-b85b-065bf2a6aad8
# ╠═f4b71e55-71fa-407f-998f-b752eec926c2
# ╠═f5aaf5ba-cf08-477e-926e-4199dcc60c32
# ╠═483e3819-a5f5-44af-9bf9-47cd9d069715
