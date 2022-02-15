### A Pluto.jl notebook ###
# v0.17.7

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

# ╔═╡ 3dddf630-7d4d-11ec-1214-7fc1b4283a09
begin
	import Pkg
	Pkg.activate("../..")
end

# ╔═╡ 5dd77ba3-a84f-4848-97dd-9c348d091bb3
using Revise, Random, PlutoUI

# ╔═╡ 3ab82140-82d6-46d2-a8b4-566c82ee042a
using ActionRNNs, Plots, RecipesBase

# ╔═╡ c96d8dd8-ddc6-444c-ac0d-8d2695eda4f3
import Pluto: Pluto, HTML

# ╔═╡ bd2ae1fe-6572-47c9-8a47-097207aff491
env = ActionRNNs.Torus2d(10, 10, 10; obs_strategy=:aliased, non_euclidean=true)

# ╔═╡ 3a947a3e-0789-4d46-a923-fa1ff94c5cda
begin
	x = rand()
	ActionRNNs.MinimalRLCore.start!(env)
end

# ╔═╡ 73868c2d-1ab3-4353-a4cd-b04c83fea025
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

# ╔═╡ 73b33c57-b9b5-46f6-8399-2414dacd025c
begin 
	go_north
	action[] = ActionRNNs.Torus2dConst.NORTH
end;

# ╔═╡ 37f52860-69ec-402e-89a8-694838d34527
begin 
	go_south
	action[] = ActionRNNs.Torus2dConst.SOUTH
end;

# ╔═╡ 05c12af7-6585-417e-9d59-2fd6ad090923
begin 
	go_east
	action[] = ActionRNNs.Torus2dConst.EAST;
end;

# ╔═╡ f88d8fde-9fb1-4288-8ff7-3db15e13546e
begin 
	go_west
	action[] = ActionRNNs.Torus2dConst.WEST;
end;

# ╔═╡ dec54cd3-33fd-41b0-910e-1b28a467e140
let
	go_north, go_south, go_east, go_west
	x
	s, r, t = ActionRNNs.MinimalRLCore.step!(env, action[])
	plot(env, 
		title=(o=Int.(ActionRNNs.MinimalRLCore.get_state(env)), r=r, t=t))
end

# ╔═╡ a1ade73b-eee7-4a18-a722-a9eddba18646
env.goal_st

# ╔═╡ a7ce928c-c3aa-47a5-a15a-ef94553a5286
savefig("torus_2d.pdf")

# ╔═╡ ca7d342c-6dbf-477a-a6fd-f924fb5e5853
randperm

# ╔═╡ ea37c881-3051-4355-943e-ef8a029b98e3
12 ÷ 3

# ╔═╡ d95c8038-704c-4e89-b8fe-6c20492feaa3
(x=2, y=2) == (x=1, y=2)

# ╔═╡ Cell order:
# ╠═3dddf630-7d4d-11ec-1214-7fc1b4283a09
# ╠═5dd77ba3-a84f-4848-97dd-9c348d091bb3
# ╠═c96d8dd8-ddc6-444c-ac0d-8d2695eda4f3
# ╠═3ab82140-82d6-46d2-a8b4-566c82ee042a
# ╠═bd2ae1fe-6572-47c9-8a47-097207aff491
# ╠═3a947a3e-0789-4d46-a923-fa1ff94c5cda
# ╟─73b33c57-b9b5-46f6-8399-2414dacd025c
# ╟─37f52860-69ec-402e-89a8-694838d34527
# ╟─05c12af7-6585-417e-9d59-2fd6ad090923
# ╟─f88d8fde-9fb1-4288-8ff7-3db15e13546e
# ╟─73868c2d-1ab3-4353-a4cd-b04c83fea025
# ╟─dec54cd3-33fd-41b0-910e-1b28a467e140
# ╠═a1ade73b-eee7-4a18-a722-a9eddba18646
# ╠═a7ce928c-c3aa-47a5-a15a-ef94553a5286
# ╠═ca7d342c-6dbf-477a-a6fd-f924fb5e5853
# ╠═ea37c881-3051-4355-943e-ef8a029b98e3
# ╠═d95c8038-704c-4e89-b8fe-6c20492feaa3
