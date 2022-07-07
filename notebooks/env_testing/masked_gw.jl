### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 06ace5d8-8483-11ec-2aea-636a33727c97
let
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ 5c1f4401-7698-4785-b5b6-d810c9d38a2a
using Revise, PlutoUI

# ╔═╡ 2f1be1f1-fd0f-49b0-b89c-c4dfdf1a7498
using ActionRNNs, Plots

# ╔═╡ bb25acf6-940a-41b9-be82-d8e2165ed3cd
env = ActionRNNs.MaskedGridWorld(10, 10, 25, 1; obs_strategy=:aliased, pacman_wrapping=true)

# ╔═╡ 81ea766d-f24e-4a48-8516-c568428e16b0
begin
	x = rand()
	ActionRNNs.MinimalRLCore.start!(env)
end

# ╔═╡ 70a73c1c-fc83-4376-9e02-8a0f418ece87
begin
	action = Ref{Int}()
	nbsp(x) = HTML(join(["&nbsp;" for i in 1:x])[1:end-1])
	s = nbsp(4)
md"""
$s
$(@bind go_north PlutoUI.Button("N")) $(@bind go_west PlutoUI.Button("W")) $(@bind go_east PlutoUI.Button("E"))\
$(nbsp(3)) $(@bind go_south PlutoUI.Button("S"))
"""
end

# ╔═╡ cfbd83a6-ce9a-4f92-9356-5fb49302e881
begin 
	go_north
	action[] = ActionRNNs.Torus2dConst.NORTH
end;

# ╔═╡ 54312522-fe62-4296-b6dc-0cd1792e237e
begin 
	go_south
	action[] = ActionRNNs.Torus2dConst.SOUTH
end;

# ╔═╡ 8cb91d18-3891-4905-9752-c36d60c95453
begin 
	go_east
	action[] = ActionRNNs.Torus2dConst.EAST;
end;

# ╔═╡ d1240339-0a97-4ea4-a573-2dc493445a12
begin 
	go_west
	action[] = ActionRNNs.Torus2dConst.WEST;
end;

# ╔═╡ b8b660f3-4622-47be-873d-b735ed4de434
let
	go_north, go_south, go_east, go_west
	x
	s, r, t = ActionRNNs.MinimalRLCore.step!(env, action[])
	plot(env, 
		title=(o=Int.(ActionRNNs.MinimalRLCore.get_state(env)), r=r, t=t))
	# savefig("../../plots/masked_gw.pdf")
end

# ╔═╡ 6c4f153d-3484-4f3d-bf77-52a6ba648805
import Random

# ╔═╡ c1201e06-e0fe-436b-bade-4f3d893f9b20
Random.randperm

# ╔═╡ Cell order:
# ╠═06ace5d8-8483-11ec-2aea-636a33727c97
# ╠═5c1f4401-7698-4785-b5b6-d810c9d38a2a
# ╠═2f1be1f1-fd0f-49b0-b89c-c4dfdf1a7498
# ╠═bb25acf6-940a-41b9-be82-d8e2165ed3cd
# ╠═81ea766d-f24e-4a48-8516-c568428e16b0
# ╟─cfbd83a6-ce9a-4f92-9356-5fb49302e881
# ╟─54312522-fe62-4296-b6dc-0cd1792e237e
# ╟─8cb91d18-3891-4905-9752-c36d60c95453
# ╟─d1240339-0a97-4ea4-a573-2dc493445a12
# ╟─70a73c1c-fc83-4376-9e02-8a0f418ece87
# ╠═b8b660f3-4622-47be-873d-b735ed4de434
# ╠═6c4f153d-3484-4f3d-bf77-52a6ba648805
# ╠═c1201e06-e0fe-436b-bade-4f3d893f9b20
