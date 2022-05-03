### A Pluto.jl notebook ###
# v0.19.3

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

# ╔═╡ 08118162-6995-45a0-91ed-f94eec767cd0
let
	using Pkg
	Pkg.activate("..")
end

# ╔═╡ f7f500a8-a1e9-11eb-009b-d7afdcade891
using Revise

# ╔═╡ e0d51e67-63dc-45ea-9092-9965f97660b3
using Reproduce, ReproducePlotUtils, StatsPlots, RollingFunctions, Statistics, FileIO, PlutoUI, Pluto

# ╔═╡ 9c649632-95df-4c0b-9501-bd660f92a1f6
using JLD2

# ╔═╡ e53d7b29-788c-469c-9d44-573f996fa5e7
const RPU = ReproducePlotUtils

# ╔═╡ 0c746c1e-ea39-4415-a1b1-d7124b886f98
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
	colorant"#88CCEE",
]

# ╔═╡ 1886bf05-f4be-4160-b61c-edf186a7f3cb
push!(RPU.stats_plot_types, :dotplot)

# ╔═╡ 6243517d-0b59-4329-903c-0c122d14a962
at(dir) = joinpath("../../local_data/dir_tmaze_online/", dir)

# ╔═╡ 834b1cf3-5b22-4e0a-abe9-61e427e6cfda
# function plot_line_from_data_with_params!(
# 		plt, data_col::Vector{PU.LineData}, params; pkwargs...)
#     idx = findfirst(data_col) do (ld)
#         line_params = ld.line_params
# 	all([line_params[i] == params[i] for i ∈ 1:length(line_params)])
#     end
#     d = data_col[idx]
#     if plt isa Nothing
# 		plt = plot(d; pkwargs...)
#     else
# 		plot!(plt, d; pkwargs...)
#     end
#     plt
# end

# ╔═╡ eefd2ccc-ce9b-4cf8-991f-a58f2f932e99
ic_dir_reg, dd_dir_reg = RPU.load_data(at("dir_tmaze_online_rmsprop_size10_1M/"))

# ╔═╡ 4c523f7e-f4a2-4364-a47b-45548878ff51
ic_dir_reg_500k, dd_dir_reg_500k = RPU.load_data(at("dir_tmaze_online_rmsprop_size10_500k_nh15/"))

# ╔═╡ ed00f8a0-f8f5-4340-8c6e-61ecb67e5d18
ic_dir_facmagru, dd_dir_facmagru = RPU.load_data(at("dir_tmaze_online_rmsprop_10_facmagru/"))

# ╔═╡ f32ab9fa-1949-4eae-96ac-0a333f83daaa
ic_dir_facmagru_t16, dd_dir_facmagru_t16 = RPU.load_data(at("dir_tmaze_online_rmsprop_10_facmagru_t16/"))

# ╔═╡ fedb725a-d13d-425f-bde9-531d453d8791
ic_dir_facmarnn, dd_dir_facmarnn = RPU.load_data(at("dir_tmaze_online_rmsprop_10_facmarnn/"))

# ╔═╡ 7d7eee9b-ea92-46d6-87b3-652805dd6468
ic_dir_facmarnn_t16, dd_dir_facmarnn_t16 = RPU.load_data(at("dir_tmaze_online_rmsprop_10_facmarnn_t16/"))

# ╔═╡ da38497d-30db-4125-91f7-3bd94ae9e3da
ic_dir_final, dd_dir_final = RPU.load_data(at("final_dir_tmaze_online_rmsprop_10/"))

# ╔═╡ dd993256-e953-4fcb-bac2-e24dbc76086f
ic_dir_og, dd_dir_og = RPU.load_data(at("dir_tmaze_online_rmsprop_size10/"))

# ╔═╡ 61fc2ec1-94db-4cca-a805-cf3da445e65d
data_og = RPU.get_line_data_for(
	ic_dir_og,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 300))

# ╔═╡ 892ce74d-d148-40cd-afea-019ea3a798ad


# ╔═╡ 35ae89bb-a66d-4fed-be4c-2f134b73afbf
ic_dir_10, dd_dir_10 = ic_dir_og, dd_dir_og

# ╔═╡ eab5c7a0-8052-432d-977e-68b967baf5ca
ic_dir_10[1].parsed_args["steps"]

# ╔═╡ 6761f9ec-8b6b-48d3-9b2b-977799118ae0


# ╔═╡ 55d1416b-d580-4112-827a-30504c21f397
data_10 = RPU.get_line_data_for(
	ic_dir_10,
	["numhidden", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 300))

# ╔═╡ f57df64d-0cd1-491e-8443-b504e217b477
data_500k = RPU.get_line_data_for(
	ic_dir_reg_500k,
	["numhidden", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 300))

# ╔═╡ a8ff395b-4161-4be1-82e3-21316b550af4
data_facmagru = RPU.get_line_data_for(
	ic_dir_facmagru,
	["numhidden_factors"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 300))

# ╔═╡ 9cbe1fd5-1948-4d7c-92dc-e5999ba5c0e5
data_facmarnn = RPU.get_line_data_for(
	ic_dir_facmarnn,
	["numhidden_factors"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 300))

# ╔═╡ 15a642f1-717c-4bff-8b7a-874b31b3bf1b
data_sens_facmagru = RPU.get_line_data_for(
	ic_dir_facmagru_t16,
	["numhidden_factors", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ c531d4fa-89f4-4bb8-8628-f0d9978247c8
md"""
Number Hidden and Factors: $(@bind nh_facs_grus Select(string.(dd_dir_facmagru_t16["numhidden_factors"])))
"""

# ╔═╡ dd543d06-108f-4992-8afd-dc0ada7e7a88
let
	# plt = nothing
	nh_gru = parse(Int, nh_facs_grus[2:3])
	factors_gru = parse(Int, nh_facs_grus[6:7])
	plt = plot()
	plt = plot!(data_sens_facmagru,
	 	  Dict("numhidden_factors"=>[nh_gru, factors_gru]);
	 	  sort_idx="eta",
		  z=1.97, lw=2, xaxis=:log,
		  palette=RPU.custom_colorant, label="none", title="FacMAGRU numhidden_factors: $(nh_facs_grus)")
	plt
end

# ╔═╡ 467ed417-cf69-493d-b21c-3fc4d1fb9907
md"""
Truncation: $(@bind τ_dir_10 Select(string.(dd_dir_og["truncation"])))
NumHidden: $(@bind nh_dir_10 Select(string.(dd_dir_og["numhidden"])))
Cell: $(@bind cells_dir_10 MultiSelect(dd_dir_og["cell"]))
"""

# ╔═╡ 2040d613-0ea8-48ce-937c-9180073812ea
let
	# plt = nothing
	τ = parse(Int, τ_dir_10)
	nh = parse(Int, nh_dir_10)
	plt = plot()
	for cell ∈ cells_dir_10
		plt = plot!(
			  data_og,
			  Dict("numhidden"=>nh, "truncation"=>τ, "cell"=>cell),
			  palette=RPU.custom_colorant, legend=:topleft)
	end
	plt
end

# ╔═╡ a26a302e-bcbd-4671-9dc3-57cd92c0e6f8
md"""
NumHidden and Factors GRU: $(@bind nh_fac_dir_facmagru Select(string.(dd_dir_facmagru["numhidden_factors"])))
NumHidden and Factors RNN: $(@bind nh_fac_dir_facmarnn Select(string.(dd_dir_facmarnn["numhidden_factors"])))
"""

# ╔═╡ d4e9f6a6-f2fe-4145-b2ec-f15600d60f42
let
	# plt = nothing
	#τ = parse(Int, τ_dir_10)
	nh_gru = parse(Int, nh_fac_dir_facmagru[2:3])
	factors_gru = parse(Int, nh_fac_dir_facmagru[6:7]) 
	nh_rnn = parse(Int, nh_fac_dir_facmarnn[2:3])
	factors_rnn = parse(Int, nh_fac_dir_facmarnn[6:7]) 
	plt = plot()
	plt = plot!(
		data_facmagru,
		Dict("numhidden_factors"=>[nh_gru, factors_gru]), label="FacMAGRU $([nh_gru, factors_gru])",
		palette=RPU.custom_colorant, legend=:topleft)
	plt = plot!(
		data_facmarnn,
		Dict("numhidden_factors"=>[nh_rnn, factors_rnn]), label="FacMARNN $([nh_rnn, factors_rnn])",
		palette=RPU.custom_colorant, legend=:topleft)
	plt
end

# ╔═╡ 7c1f8e58-4bfa-4da2-833d-0cc2a4ec74a4
data_10_sens = RPU.get_line_data_for(
	ic_dir_10,
	["numhidden", "cell", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 859b98dd-1024-4fec-8741-40edf72d0287
data_10_sens[1]

# ╔═╡ f5a1e45c-398a-40e3-8c5f-b1d72ead0a89
params = Dict("numhidden" => 20, "truncation" => 20, "cell" => "MARNN")

# ╔═╡ a4763342-d9bf-4d5c-81b9-a735ae3729e3
keys(params)

# ╔═╡ 6b253f85-9005-41b6-8544-e381f46285d8
params_keys = ("numhidden", "truncation", "cell")

# ╔═╡ 094b456c-a7ad-4f93-89a0-c403c73bdc55
data_10_sens[idx[2]].data

# ╔═╡ e2f179e9-5e75-4e6c-917b-07aef8b6f8b7
mean(data_10_sens[idx[2]].data)

# ╔═╡ 4a64f3a7-3879-407a-9654-aacc1861fc42
data_10_sens_ = RPU.get_line_data_for(
	ic_dir_10,
	["numhidden", "truncation", "cell", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 300))

# ╔═╡ d50978f3-f38f-42af-a7d1-1e3b59e68175
md"""
Eta: $(@bind eta_dir_sens Select(string.(dd_dir_10["eta"])))
NumHidden: $(@bind nh_dir_sens Select(string.(dd_dir_10["numhidden"])))
Cell: $(@bind cells_dir_sens MultiSelect(dd_dir_10["cell"]))
"""

# ╔═╡ 7425a013-fdbf-4d50-921d-64306d4c2741
let
	# plt = nothing
	τ = parse(Int, τ_dir_sens)
	nh = parse(Int, nh_dir_sens)
	eta = parse(Float64, eta_dir_sens)
	plt = plot()
	for cell ∈ cells_dir_sens
		plt = plot!(
			  data_10_sens_,
			  Dict("numhidden"=>nh, "truncation"=>τ, "cell"=>cell, "eta"=>eta),
			  palette=RPU.custom_colorant, legend=:topleft)
	end
	plt
end

# ╔═╡ c05c7ffa-53cd-46bf-a661-886784eecc05
plt_args_list = let
	args_list = [
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"MAGRU", "eta"=>0.0003125), #num_params = 1303
#		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"MAGRU", "eta"=>0.0003125), #num_params = 3499
#		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"AAGRU", "eta"=>0.00125), #num_params = 1053
#		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"AAGRU", "eta"=>0.0003125),
#		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AAGRU", "eta"=>0.00125), #num_params = 1703
#		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"GRU", "eta"=>0.00125),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"MARNN", "eta"=>0.0003125), #num_params = 463
#		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AARNN", "eta"=>0.0003125),
#		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"RNN", "eta"=>1.953125e-5)
	]

	FileIO.save("../final_runs/dir_tmaze_10.jld2", "args", args_list)

	plt_keys = ["numhidden", "truncation", "cell"]
	[Dict(k=>args_list[i][k] for k ∈ plt_keys) for i in 1:length(args_list)]
end

# ╔═╡ 90eba5d4-7a1c-4d89-b743-620ee31af023
ic_dir_10

# ╔═╡ fc961ab4-c7c6-4e7b-91be-3e5fbb18d667
data_10_dist = RPU.get_line_data_for(
	ic_dir_10,
	["numhidden", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ f2960c43-c062-4c4a-8472-e0791adae9c6
data_10_dist_500k = RPU.get_line_data_for(
	ic_dir_reg_500k,
	["numhidden", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 9a66b4f9-d10b-478c-bd58-63df5dbef47b
data_10_dist_all = RPU.get_line_data_for(
	ic_dir_og,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 30341781-351f-4b61-80a3-3f3c65f816e2
data_10_dist_facmagru = RPU.get_line_data_for(
	ic_dir_facmagru,
	["numhidden_factors"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ c107cfa9-6ac9-40d4-8cf2-610779421e4f
data_10_dist_facmarnn = RPU.get_line_data_for(
	ic_dir_facmarnn,
	["numhidden_factors"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ cb4d5803-996d-4341-80d0-3ef1bae52dc6
dd_dir_facmagru

# ╔═╡ 9268f797-a997-4165-a57a-8defb4b751dd


# ╔═╡ 08552490-3e28-4f9c-8c98-c6f2a19fca6f
dd_dir_facmarnn

# ╔═╡ c0d46ded-a7e0-4aef-b385-e53cffefedbb
ic_dir_final[1]

# ╔═╡ c171dcdb-0f21-4d63-8755-a3441434028b
FileIO.load(joinpath(ic_dir_final[1].folder_str, "results.jld2"))

# ╔═╡ af855937-4069-4256-8682-f18d3568f8d7
function get_300k(data)
  ns = 0
  idx = 1
  while ns < 300_000
      ns += data["results"][:total_steps][idx]
      idx += 1
  end
  data["results"][:successes][1:idx]
end

# ╔═╡ 19af4769-3e99-4e69-8b9a-6b7ff63ef803
function get_MUE(data, perc)
    mean(data[end-max(1, Int(floor(length(data)*perc))):end])
end

# ╔═╡ 68481a3f-ba2e-4d37-b838-370070b40fc5
function get_MEAN(data)
    mean(data)
end

# ╔═╡ 5456a07a-8d3d-4d4c-8001-b72aac54286e
ic, dd = let
	ic = ItemCollection(at("final_dir_tmaze_online_rmsprop_10_t16/"))
	subic = search(ic) do ld
		ld.parsed_args["cell"][1:3] !== "Fac"
	end
	subic, diff(subic)
end

# ╔═╡ a8c60669-4401-4d56-8f23-39efee7030e4
ic_fac_magru, dd_fac_magru = let
	ic = ItemCollection(at("final_dir_tmaze_online_rmsprop_10_t16/"))
	subic = search(ic, Dict("cell"=>"FacMAGRU"))
	subic, diff(subic)
end

# ╔═╡ 31a0ef43-deb1-41e1-ba7b-d1c25ef07bc8
ic_fac_marnn, dd_fac_marnn = let
	ic = ItemCollection(at("final_dir_tmaze_online_rmsprop_10_t16/"))
	subic = search(ic, Dict("cell"=>"FacMARNN"))
	subic, diff(subic)
end

# ╔═╡ 5deaf434-1441-4cf7-bb20-80e4498c4ba2
ic_joint = ItemCollection([ic.items; ic_fac_magru.items; ic_fac_marnn.items])

# ╔═╡ 66a3490e-a515-45c9-ba97-a308059d6467
data_t16 = RPU.get_line_data_for(
	ic_joint,
	["cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_rolling_mean_line(x, :successes, 1000))

# ╔═╡ 6962749e-0324-4edb-a4bf-f320ce1fb235
dd_joint = diff(ic_joint)

# ╔═╡ c437bc65-65cb-4486-8a98-3bf290ce3162
md"""
NumHidden: $(@bind nh_joint Select(string.(dd_joint["numhidden"])))
Cell: $(@bind cells_joint MultiSelect(dd_joint["cell"]))
"""

# ╔═╡ f93f586c-bd83-4175-87ca-6e4278da5de4
let
	# plt = nothing
	nh = parse(Int, nh_joint)
	plt = plot()
	for cell ∈ cells_joint
		plt = plot!(
			  data_t16,
			  Dict("numhidden"=>nh, "cell"=>cell),
			  palette=RPU.custom_colorant, legend=:topleft)
	end
	plt
end

# ╔═╡ 82dba65f-8792-4c16-8aab-c9de5c660f73
data_dist_final_t16 = RPU.get_line_data_for(
	ic_joint,
	["cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 6e7a223b-9af1-4160-a291-8799e3c963b2
data_dist_final = RPU.get_line_data_for(
	ic_dir_final,
	["cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ c6366560-ef13-4d6e-bc86-4861af5b559d
data_dist_final_300k_t16 = RPU.get_line_data_for(
	ic_joint,
	["cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->get_MUE(get_300k(x), 0.1),
	get_data=(x)->get_MUE(get_300k(x), 0.1))

# ╔═╡ b8bca610-afbb-4000-b603-3c917f823f36
data_dist_final_300k = RPU.get_line_data_for(
	ic_dir_final,
	["cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->get_MUE(get_300k(x), 0.1),
	get_data=(x)->get_MUE(get_300k(x), 0.1))

# ╔═╡ ee0bddea-ce0d-45c0-b5e0-f7202d46cb4f
data_dist_final_mean_t16 = RPU.get_line_data_for(
	ic_joint,
	["cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :successes),
	get_data=(x)->RPU.get_MEAN(x, :successes))

# ╔═╡ 226cbd88-f9f4-4022-bab3-607e066e0f28
data_dist_final_mean = RPU.get_line_data_for(
	ic_dir_final,
	["cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :successes),
	get_data=(x)->RPU.get_MEAN(x, :successes))

# ╔═╡ d41b2113-ba79-4c8b-80b8-71e4be18ff69
data_dist_final_mean_300k_t16 = RPU.get_line_data_for(
	ic_joint,
	["cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->get_MEAN(get_300k(x)),
	get_data=(x)->get_MEAN(get_300k(x)))

# ╔═╡ 78d8672c-c0ad-4f64-b931-308a46b9479c
data_dist_final_mean_300k = RPU.get_line_data_for(
	ic_dir_final,
	["cell"],
	[];
	comp=findmax,
	get_comp_data=(x)->get_MEAN(get_300k(x)),
	get_data=(x)->get_MEAN(get_300k(x)))

# ╔═╡ 99439b9e-49bb-48b8-bf38-3f1f229bcaa1
begin
	args_list_hc = [
		Dict("numhidden"=>46, "truncation"=>16, "cell"=>"RNN", "eta"=>0.0002220),
		Dict("numhidden"=>46, "truncation"=>16, "cell"=>"AARNN", "eta"=>0.0002220),
		Dict("numhidden"=>27, "truncation"=>16, "cell"=>"MARNN", "eta"=>0.0002220), 
		Dict("numhidden"=>46, "factors"=>24, "truncation"=>16, "cell"=>"FacMARNN", "eta"=>0.0009095),
		Dict("numhidden"=>26, "truncation"=>16, "cell"=>"GRU", "eta"=>0.0009095), 
		Dict("numhidden"=>26, "truncation"=>16, "cell"=>"AAGRU", "eta"=>0.0009095),
		Dict("numhidden"=>15, "truncation"=>16, "cell"=>"MAGRU", "eta"=>0.0009095), 
		Dict("numhidden"=>26, "factors"=>21, "truncation"=>16, "cell"=>"FacMAGRU", "eta"=>0.0009095)
		]
	
	# FileIO.save("../final_runs/dir_tmaze_online_10_t16.jld2", "args", args_list_hc)
end

# ╔═╡ f5f0c142-8ef1-4abc-a0c9-40c927436942
f=jldopen(at("final_dir_tmaze_online_rmsprop_10_t16/settings/settings_0x4faa9134464b4a4e.jld2"), "r")

# ╔═╡ 8520162d-6c0f-41af-84eb-63b1edcefb20
f_=jldopen(at("final_dir_tmaze_online_rmsprop_10/settings/settings_0xc0fd784eb4140e93.jld2"), "r")

# ╔═╡ 6170c64c-9bfa-46dd-869c-6a989faf57c5
f1=jldopen(at("final_dir_tmaze_online_rmsprop_10/data/RP_0_0x1a95034fa9423e2f/settings.jld2"), "r")

# ╔═╡ f8388454-8c55-4404-a05f-a5e51ea225ca
data1 = read(f1, keys(f1)[1])

# ╔═╡ 549a936c-e6d3-49a1-9172-f7abfc41956e
data = read(f, keys(f)[1])

# ╔═╡ 5ac45ce9-7074-4df6-bc48-c3db35bb2fe3
data_ = read(f_, keys(f_)[1])

# ╔═╡ c8e944f2-ef02-4b81-8b6d-3527609f0822
data_10_dist_mean = RPU.get_line_data_for(
	ic_dir_10,
	["numhidden", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :successes),
	get_data=(x)->RPU.get_MEAN(x, :successes))

# ╔═╡ f4e6b374-f900-4545-b8a6-d95c85ab3596
data_10_dist_mean_500k = RPU.get_line_data_for(
	ic_dir_reg_500k,
	["numhidden", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :successes),
	get_data=(x)->RPU.get_MEAN(x, :successes))

# ╔═╡ 8a7d820c-7ca9-461d-9160-27c47496436a
data_10_dist_mean_facmagru = RPU.get_line_data_for(
	ic_dir_facmagru,
	["numhidden_factors"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :successes),
	get_data=(x)->RPU.get_MEAN(x, :successes))

# ╔═╡ 04655ac3-0d40-4063-bc8c-0826e13c6722
data_10_dist_mean_facmarnn = RPU.get_line_data_for(
	ic_dir_facmarnn,
	["numhidden_factors"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :successes),
	get_data=(x)->RPU.get_MEAN(x, :successes))

# ╔═╡ 6ef161be-ec61-461d-9c38-60510c08328b
ic_dir_10_s, dd_dir_10_s = RPU.load_data(at("dir_tmaze_online_rmsprop_size10/"))

# ╔═╡ 4f2e1222-9daa-439c-9d57-957f23e44657
data_10_dist_s = RPU.get_line_data_for(
	ic_dir_10_s,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MEAN(x, :successes),
	get_data=(x)->RPU.get_MEAN(x, :successes))

# ╔═╡ b8be3138-98e3-476a-adcf-564196b51ae7
data_10_dist_s_mue = RPU.get_line_data_for(
	ic_dir_10_s,
	["numhidden", "truncation", "cell"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 305f8ac8-8f5f-4ec4-84a6-867f69a8887c
ic_fac, dd_fac = RPU.load_data(at("dir_tmaze_er_fac_rnn_rmsprop_10/"))

# ╔═╡ 533cba3d-7fc5-4d66-b545-b15ffc8ab6d8
data_fac_sens_eta = RPU.get_line_data_for(
	ic_fac,
	["numhidden", "cell", "replay_size", "factors", "eta"],
	[];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ 39752286-a6db-439d-aca0-1be4821bfc2b
let
	args_list = [
		Dict("numhidden"=>15, "replay_size"=>20000, "cell"=>"FacMARNN", "factors"=>10),
		Dict("numhidden"=>15, "replay_size"=>20000, "cell"=>"FacMAGRU", "factors"=>10)]
	plot(data_fac_sens_eta, args_list; sort_idx="eta", labels=["FacMARNN" "FacMAGRU"])
end

# ╔═╡ 7d611b39-f9a8-43e4-951e-9d812cbd4384
data_fac_sens = RPU.get_line_data_for(
	ic_fac,
	["numhidden", "cell", "replay_size", "factors"],
	["eta"];
	comp=findmax,
	get_comp_data=(x)->RPU.get_MUE(x, :successes),
	get_data=(x)->RPU.get_MUE(x, :successes))

# ╔═╡ a8949e02-61f5-456a-abee-2bad91d2df05
let
	args_list = [
		Dict("numhidden"=>15, "replay_size"=>20000, "cell"=>"FacMARNN"),
		Dict("numhidden"=>15, "replay_size"=>20000, "cell"=>"FacMAGRU")]
	plot(data_fac_sens, args_list; sort_idx="factors", labels=["FacMARNN" "FacMAGRU"])
end

# ╔═╡ 9b400057-90ff-4b8b-8abf-f6fe1163f281
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN")
	]
	names = ["GRU", "AAGRU", "MAGRU", "FacGRU", "RNN", "AARNN", "MARNN", "FacRNN"]
	plt = plot()
	for i in 1:length(names)
		plt = violin!(data_dist_final_300k_t16, args_list[i], label=names[i], legend=false, color=cell_colors_[args_list[i]["cell"]], lw=1, linecolor=cell_colors_[args_list[i]["cell"]])
		
		plt = boxplot!(data_dist_final_300k_t16, args_list[i], label=names[i], color=color=cell_colors_[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15)
		
	if i == 4	
		plt = vline!([6], linestyle=:dot, color=:white, lw=2)
	end
		#plt = dotplot!(data_dist_final_300k, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(0.4, 1.0))
		
	end
	plt
	
	savefig("../data/paper_plots/dir_tmaze_online_violin_and_box_plots_300k_steps_MUE_tau_16.pdf")
end

# ╔═╡ e265c8f2-b147-4509-81a6-8c34a7de5457
color_scheme_ = [
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

# ╔═╡ fe50ffef-b691-47b5-acf8-8378fbf860a1
cell_colors = Dict(
	"RNN" => color_scheme_[3],
	"AARNN" => color_scheme_[end],
	"MARNN" => color_scheme_[5],
	"FacMARNN" => color_scheme_[1],
	"GRU" => color_scheme_[4],
	"AAGRU" => color_scheme_[2],
	"MAGRU" => color_scheme_[6],
	"FacMAGRU" => color_scheme_[end-2])

# ╔═╡ 6f7494cb-7fb9-4ddc-a14f-e50364ae624a
let
	# plt = nothing
	plt = plot()
	plt = plot!(
		  data_t16,
		  Dict("cell"=>"GRU"), label="cell: GRU (nh: 26)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(0.4, 1.1), ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors["GRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_t16,
		  Dict("cell"=>"AAGRU"), label="cell: AAGRU (nh: 26)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(0.4, 1.1), ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors["AAGRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_t16,
		  Dict("cell"=>"MAGRU"), label="cell: MAGRU (nh: 15)",
			  palette=RPU.custom_colorant, legend=:bottomright, ylim=(0.4, 1.1), ylabel="Total Reward", xlabel="Episode", lw=2, z=1, color=cell_colors["MAGRU"], fillalpha=0.3)
	
	plt = plot!(
		  data_t16,
		  Dict("cell"=>"FacMAGRU"), label="cell: FacMAGRU (nh: 26, fac: 21)",
			  palette=RPU.custom_colorant, legend=:topleft, ylim=(0.4, 1.1), ylabel="Success Rate", xlabel="Episode", lw=2, z=1, color=cell_colors["FacMAGRU"], title="Dir Tmaze Final Runs, τ: 16, Steps: 500K", fillalpha=0.3)
	plt
end

# ╔═╡ f5c3a38a-cf78-4b90-b371-506cc2997f92
let
	# plt = nothing
	# τ = parse(Int, τ_dir_sens)
	nh = parse(Int, nh_dir_sens)
	plt = plot()
	for cell ∈ cells_dir_sens
		plt = plot!(data_10_sens,
	 	  Dict("numhidden"=>nh, "cell"=>cell);
	 	  sort_idx="eta",
		  z=1.97, lw=2, xaxis=:log,
		  palette=RPU.custom_colorant, label="cell: $(cell)",
		  color=cell_colors[cell], title="numhidden: $(nh)")
	end
	plt
end

# ╔═╡ db83dce1-29d5-42f8-b226-4412ce63c8f1
let
	args_list_l = [
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"GRU"),
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"AAGRU"),
		Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MAGRU"),
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"RNN"),
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"AARNN"),
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"MARNN")]

	boxplot(data_10_dist_all, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="MUE")
	dotplot!(data_10_dist_all, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
end

# ╔═╡ c4d1ce22-d9e5-4505-8e74-c02e6e9ddefe
let
	args_list_l = [
		Dict("numhidden"=>26, "cell"=>"GRU"),
		Dict("numhidden"=>26, "cell"=>"AAGRU"),
		Dict("numhidden"=>15, "cell"=>"MAGRU"),
		Dict("numhidden"=>46, "cell"=>"RNN"),
		Dict("numhidden"=>46, "cell"=>"AARNN"),
		Dict("numhidden"=>27, "cell"=>"MARNN")]
	
	#args_list_l = [
	#	Dict("numhidden"=>20, "truncation"=>12, "cell"=>"GRU"),
	#	Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AAGRU"),
	#	Dict("numhidden"=>10, "truncation"=>20, "cell"=>"MAGRU"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"RNN"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"AARNN"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"MARNN")]
	boxplot(data_10_dist_500k, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="MUE")
	dotplot!(data_10_dist_500k, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ 5096204e-5134-4fda-8894-20e0c1a3e650
let
	args_list = [
		Dict("numhidden_factors"=>[15, 37]),
		Dict("numhidden_factors"=>[15, 75]),
		Dict("numhidden_factors"=>[15, 100]),
		Dict("numhidden_factors"=>[20, 28]),
		Dict("numhidden_factors"=>[20, 75]),
		Dict("numhidden_factors"=>[20, 100]),
		Dict("numhidden_factors"=>[26, 21]),
		Dict("numhidden_factors"=>[26, 75]),
		Dict("numhidden_factors"=>[26, 100])
	]
	boxplot(data_10_dist_facmagru, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMAGRU"] cell_colors["FacMAGRU"] cell_colors["FacMAGRU"]], legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="FacMAGRU")
	dotplot!(data_10_dist_facmagru, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMAGRU"] cell_colors["FacMAGRU"] cell_colors["FacMAGRU"]])

#savefig("../data/paper_plots/dir_tmaze_online_facgru_box_plot.pdf")
end

# ╔═╡ fafaf816-d61e-4f52-8528-9b2b8e9fe971
let
	args_list = [
		Dict("numhidden_factors"=>[27, 40]),
		Dict("numhidden_factors"=>[27, 75]),
		Dict("numhidden_factors"=>[27, 100]),
		Dict("numhidden_factors"=>[36, 31]),
		Dict("numhidden_factors"=>[36, 75]),
		Dict("numhidden_factors"=>[36, 100]),
		Dict("numhidden_factors"=>[46, 24]),
		Dict("numhidden_factors"=>[46, 75]),
		Dict("numhidden_factors"=>[46, 100])
	]
	boxplot(data_10_dist_facmarnn, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMARNN"]], legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="FacMARNN MUE")
	dotplot!(data_10_dist_facmarnn, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMARNN"]])
	
end

# ╔═╡ a6dc6873-8779-4e4e-856c-3dc818268acf
let
	args_list = [
		Dict("numhidden_factors"=>[15, 37]),
		Dict("numhidden_factors"=>[20, 28]),
		Dict("numhidden_factors"=>[26, 21])
	]
	boxplot(data_10_dist_facmagru, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMAGRU"] cell_colors["FacMAGRU"] cell_colors["FacMAGRU"]])
	dotplot!(data_10_dist_facmagru, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMAGRU"] cell_colors["FacMAGRU"] cell_colors["FacMAGRU"]])
	
	args_list_l = [
		Dict("numhidden"=>26, "cell"=>"GRU"),
		Dict("numhidden"=>26, "cell"=>"AAGRU"),
		Dict("numhidden"=>15, "cell"=>"MAGRU"),
		Dict("numhidden"=>46, "cell"=>"RNN"),
		Dict("numhidden"=>46, "cell"=>"AARNN"),
		Dict("numhidden"=>27, "cell"=>"MARNN")]

	boxplot!(data_10_dist_500k, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="MUE")
	dotplot!(data_10_dist_500k, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	
	
	args_list_rnn = [
		Dict("numhidden_factors"=>[27, 40]),
		Dict("numhidden_factors"=>[36, 31]),
		Dict("numhidden_factors"=>[46, 24])
	]
	boxplot!(data_10_dist_facmarnn, args_list_rnn; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMARNN"] cell_colors["FacMARNN"] cell_colors["FacMARNN"]])
	dotplot!(data_10_dist_facmarnn, args_list_rnn; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMARNN"] cell_colors["FacMARNN"] cell_colors["FacMARNN"]])
end

# ╔═╡ c6636dd2-243d-40fa-99ac-a7982d7a0a9c
let
	args_list_1 = [
		Dict("numhidden"=>26, "cell"=>"GRU")]
	boxplot(data_10_dist_500k, args_list_1;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_1, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="MUE")
	dotplot!(data_10_dist_500k, args_list_1;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_1, "cell")), 1, :))
	
	args_list = [
		Dict("numhidden_factors"=>[26, 21])
	]
	boxplot!(data_10_dist_facmagru, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMAGRU"] cell_colors["FacMAGRU"] cell_colors["FacMAGRU"]])
	dotplot!(data_10_dist_facmagru, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMAGRU"] cell_colors["FacMAGRU"] cell_colors["FacMAGRU"]])
	
	args_list_2 = [
		Dict("numhidden"=>26, "cell"=>"AAGRU"),
		Dict("numhidden"=>15, "cell"=>"MAGRU"),
		Dict("numhidden"=>46, "cell"=>"RNN")]
	boxplot!(data_10_dist_500k, args_list_2;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_2, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Lunar Lander MUE, Steps: 500k")
	dotplot!(data_10_dist_500k, args_list_2;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_2, "cell")), 1, :))
	
	
	args_list_rnn = [
		Dict("numhidden_factors"=>[46, 24])
	]
	boxplot!(data_10_dist_facmarnn, args_list_rnn; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMARNN"] cell_colors["FacMARNN"] cell_colors["FacMARNN"]])
	dotplot!(data_10_dist_facmarnn, args_list_rnn; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMARNN"] cell_colors["FacMARNN"] cell_colors["FacMARNN"]])
	
	args_list_3 = [
		Dict("numhidden"=>46, "cell"=>"AARNN"),
		Dict("numhidden"=>27, "cell"=>"MARNN")]
	boxplot!(data_10_dist_500k, args_list_3;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_3, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, Steps: 500k")
	dotplot!(data_10_dist_500k, args_list_3;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_3, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ f0068edb-4762-4e7b-850c-0c37e90ceb00
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN")
	]
	names = ["GRU", "AAGRU", "MAGRU", "FacGRU", "RNN", "AARNN", "MARNN", "FacRNN"]
	plt = plot()
	for i in 1:length(names)
		if i == 1
			plt = boxplot(data_dist_final_mean_300k_t16, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, τ: 16, Steps: 500k")
		else
			plt = boxplot!(data_dist_final_mean_300k_t16, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MEAN, τ: 16, Steps: 300k")
		end
		plt = dotplot!(data_dist_final_mean_300k_t16, [args_list[i]];
			label=names[i],
			color=cell_colors[args_list[i]["cell"]])
	end
	plt
end

# ╔═╡ 59912f1e-6bf1-40b3-9c34-03a3b4542bd6
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN")
	]
	names = ["GRU", "AAGRU", "MAGRU", "FacGRU", "RNN", "AARNN", "MARNN", "FacRNN"]
	plt = plot()
	for i in 1:length(names)
		if i == 1
			plt = boxplot(data_dist_final_300k_t16, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, τ: 16, Steps: 300k")
		else
			plt = boxplot!(data_dist_final_300k_t16, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, τ: 16, Steps: 300k")
		end
		plt = dotplot!(data_dist_final_300k_t16, [args_list[i]];
			label=names[i],
			color=cell_colors[args_list[i]["cell"]])
	end
	plt
end

# ╔═╡ 413bfa10-4039-4b51-8ac9-f04006adc074
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN")
	]
	names = ["GRU", "AAGRU", "MAGRU", "FacGRU", "RNN", "AARNN", "MARNN", "FacRNN"]
	plt = plot()
	for i in 1:length(names)
		if i == 1
			plt = boxplot(data_dist_final_mean_300k, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, τ: 16, Steps: 300k")
		else
			plt = boxplot!(data_dist_final_mean_300k, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MEAN, τ: 20, Steps: 300k")
		end
		plt = dotplot!(data_dist_final_mean_300k, [args_list[i]];
			label=names[i],
			color=cell_colors[args_list[i]["cell"]])
	end
	plt
end

# ╔═╡ c4c36ce4-aa29-4ba5-bba4-2b88362ed744
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN")
	]
	names = ["GRU", "AAGRU", "MAGRU", "FacGRU", "RNN", "AARNN", "MARNN", "FacRNN"]
	plt = plot()
	for i in 1:length(names)
		if i == 1
			plt = boxplot(data_dist_final_300k, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, τ: 16, Steps: 300k")
		else
			plt = boxplot!(data_dist_final_300k, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, τ: 20, Steps: 300k")
		end
		plt = dotplot!(data_dist_final_300k, [args_list[i]];
			label=names[i],
			color=cell_colors[args_list[i]["cell"]])
	end
	plt
end

# ╔═╡ 74639e00-1076-4600-9565-3574a839d915
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN")
	]
	names = ["GRU", "AAGRU", "MAGRU", "FacGRU", "RNN", "AARNN", "MARNN", "FacRNN"]
	plt = plot()
	for i in 1:length(names)
		if i == 1
			plt = boxplot(data_dist_final_mean_t16, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, τ: 16, Steps: 500k")
		else
			plt = boxplot!(data_dist_final_mean_t16, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MEAN, τ: 16, Steps: 500k")
		end
		plt = dotplot!(data_dist_final_mean_t16, [args_list[i]];
			label=names[i],
			color=cell_colors[args_list[i]["cell"]])
	end
	plt
end

# ╔═╡ 98b89884-37f6-4476-bf41-e3aae3b40248
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN")
	]
	names = ["GRU", "AAGRU", "MAGRU", "FacGRU", "RNN", "AARNN", "MARNN", "FacRNN"]
	plt = plot()
	for i in 1:length(names)
		if i == 1
			plt = boxplot(data_dist_final_t16, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, τ: 16, Steps: 500k")
		else
			plt = boxplot!(data_dist_final_t16, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, τ: 16, Steps: 500k")
		end
		plt = dotplot!(data_dist_final_t16, [args_list[i]];
			label=names[i],
			color=cell_colors[args_list[i]["cell"]])
	end
	plt
end

# ╔═╡ 41aef310-21d6-4e9f-8eb1-28005a736487
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN")
	]
	names = ["GRU", "AAGRU", "MAGRU", "FacGRU", "RNN", "AARNN", "MARNN", "FacRNN"]
	plt = plot()
	for i in 1:length(names)
		if i == 1
			plt = boxplot(data_dist_final_mean, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, τ: 16, Steps: 500k")
		else
			plt = boxplot!(data_dist_final_mean, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MEAN, τ: 20, Steps: 500k")
		end
		plt = dotplot!(data_dist_final_mean, [args_list[i]];
			label=names[i],
			color=cell_colors[args_list[i]["cell"]])
	end
	plt
end

# ╔═╡ 0718582b-811f-4227-9f8c-2368881ac8dc
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN")
	]
	names = ["GRU", "AAGRU", "MAGRU", "FacGRU", "RNN", "AARNN", "MARNN", "FacRNN"]
	plt = plot()
	for i in 1:length(names)
		if i == 1
			plt = boxplot(data_dist_final, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, τ: 16, Steps: 500k")
		else
			plt = boxplot!(data_dist_final, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, τ: 20, Steps: 500k")
		end
		plt = dotplot!(data_dist_final, [args_list[i]];
			label=names[i],
			color=cell_colors[args_list[i]["cell"]])
	end
	plt
end

# ╔═╡ 6a8ea54b-86e6-4c38-8bda-7e39c6ef1605
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"FacMARNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN")
	]
	boxplot(data_dist_final, args_list;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, Steps: 500k")
	dotplot!(data_dist_final, args_list;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ 398e5230-e8d0-4261-85a9-5b049702a8ce
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"FacMARNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN")
	]
	boxplot(data_dist_final_300k, args_list;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MUE, Steps: 300k")
	dotplot!(data_dist_final_300k, args_list;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ 8af2e436-a097-4fbb-81f5-17018f02b2fd
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"FacMARNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN")
	]
	boxplot(data_dist_final_mean, args_list;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MEAN, Steps: 500k")
	dotplot!(data_dist_final_mean, args_list;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ 090d428e-f2c1-4032-878d-df6de3063266
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"FacMARNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN")
	]
	boxplot(data_dist_final_mean_300k, args_list;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Dir TMaze Online MEAN, Steps: 300k")
	dotplot!(data_dist_final_mean_300k, args_list;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ f0390e50-13a6-496f-b0ab-97ed1b2b18e7
let
	args_list = [
		Dict("numhidden_factors"=>[15, 37]),
		Dict("numhidden_factors"=>[15, 75]),
		Dict("numhidden_factors"=>[15, 100]),
		Dict("numhidden_factors"=>[20, 28]),
		Dict("numhidden_factors"=>[20, 75]),
		Dict("numhidden_factors"=>[20, 100]),
		Dict("numhidden_factors"=>[26, 21]),
		Dict("numhidden_factors"=>[26, 75]),
		Dict("numhidden_factors"=>[26, 100])
	]
	boxplot(data_10_dist_mean_facmagru, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMAGRU"] cell_colors["FacMAGRU"] cell_colors["FacMAGRU"]], legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="FacMAGRU MEAN")
	dotplot!(data_10_dist_mean_facmagru, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMAGRU"] cell_colors["FacMAGRU"] cell_colors["FacMAGRU"]])
	
end

# ╔═╡ ec7f5f9a-94be-473d-9542-8780c91f9e8e
let
	args_list = [
		Dict("numhidden_factors"=>[27, 40]),
		Dict("numhidden_factors"=>[27, 75]),
		Dict("numhidden_factors"=>[27, 100]),
		Dict("numhidden_factors"=>[36, 31]),
		Dict("numhidden_factors"=>[36, 75]),
		Dict("numhidden_factors"=>[36, 100]),
		Dict("numhidden_factors"=>[46, 24]),
		Dict("numhidden_factors"=>[46, 75]),
		Dict("numhidden_factors"=>[46, 100])
	]
	boxplot(data_10_dist_mean_facmarnn, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMARNN"]], legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="FacMARNN MEAN")
	dotplot!(data_10_dist_mean_facmarnn, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMARNN"]])
	
end

# ╔═╡ 588cd0d2-c598-4e50-a1a0-93f9eddbc10e
let
	args_list = [
		Dict("numhidden_factors"=>[15, 37]),
		Dict("numhidden_factors"=>[20, 28]),
		Dict("numhidden_factors"=>[26, 21])
	]
	boxplot(data_10_dist_mean_facmagru, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMAGRU"] cell_colors["FacMAGRU"] cell_colors["FacMAGRU"]])
	dotplot!(data_10_dist_mean_facmagru, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMAGRU"] cell_colors["FacMAGRU"] cell_colors["FacMAGRU"]])
	
	args_list_l = [
		Dict("numhidden"=>26, "cell"=>"GRU"),
		Dict("numhidden"=>26, "cell"=>"AAGRU"),
		Dict("numhidden"=>15, "cell"=>"MAGRU"),
		Dict("numhidden"=>46, "cell"=>"RNN"),
		Dict("numhidden"=>46, "cell"=>"AARNN"),
		Dict("numhidden"=>27, "cell"=>"MARNN")]

	boxplot!(data_10_dist_mean, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="MEAN")
	dotplot!(data_10_dist_mean, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	
	
	args_list_rnn = [
		Dict("numhidden_factors"=>[27, 40]),
		Dict("numhidden_factors"=>[36, 31]),
		Dict("numhidden_factors"=>[46, 24])
	]
	boxplot!(data_10_dist_mean_facmarnn, args_list_rnn; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMARNN"] cell_colors["FacMARNN"] cell_colors["FacMARNN"]])
	dotplot!(data_10_dist_mean_facmarnn, args_list_rnn; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMARNN"] cell_colors["FacMARNN"] cell_colors["FacMARNN"]])
end

# ╔═╡ e0a3851d-3db5-440e-b67e-75ca7d9517e1
let
	args_list_l = [
		Dict("numhidden"=>26, "cell"=>"GRU"),
		Dict("numhidden"=>26, "cell"=>"AAGRU"),
		Dict("numhidden"=>15, "cell"=>"MAGRU"),
		Dict("numhidden"=>46, "cell"=>"RNN"),
		Dict("numhidden"=>46, "cell"=>"AARNN"),
		Dict("numhidden"=>27, "cell"=>"MARNN")]
	
	#args_list_l = [
	#	Dict("numhidden"=>20, "truncation"=>12, "cell"=>"GRU"),
	#	Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AAGRU"),
	#	Dict("numhidden"=>10, "truncation"=>20, "cell"=>"MAGRU"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"RNN"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"AARNN"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"MARNN")]
	boxplot(data_10_dist_mean, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Mean")
	dotplot!(data_10_dist_mean, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ ce18cc80-89da-4608-9f01-3ba94609049d
let
	args_list_l = [
		Dict("numhidden"=>26, "cell"=>"GRU"),
		Dict("numhidden"=>26, "cell"=>"AAGRU"),
		Dict("numhidden"=>15, "cell"=>"MAGRU"),
		Dict("numhidden"=>46, "cell"=>"RNN"),
		Dict("numhidden"=>46, "cell"=>"AARNN"),
		Dict("numhidden"=>27, "cell"=>"MARNN")]
	
	#args_list_l = [
	#	Dict("numhidden"=>20, "truncation"=>12, "cell"=>"GRU"),
	#	Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AAGRU"),
	#	Dict("numhidden"=>10, "truncation"=>20, "cell"=>"MAGRU"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"RNN"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"AARNN"),
	#	Dict("numhidden"=>20, "truncation"=>20, "cell"=>"MARNN")]
	boxplot(data_10_dist_mean_500k, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="Mean")
	dotplot!(data_10_dist_mean_500k, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ 4d5dc03a-8b9b-4b5a-b002-6fc48a580b04
let
	args_list_l = [
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"GRU"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"AAGRU"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"MAGRU"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"RNN"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"AARNN"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"MARNN")]
	boxplot(data_10_dist_s, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="MEAN")
	dotplot!(data_10_dist_s, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ f9e81f7b-26f3-4ce5-bca7-5054c8ddf44a
let
	args_list_l = [
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"GRU"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"AAGRU"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"MAGRU"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"RNN"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"AARNN"),
		Dict("numhidden"=>20, "truncation"=>16, "cell"=>"MARNN")]
	boxplot(data_10_dist_s_mue, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="MUE")
	dotplot!(data_10_dist_s_mue, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
		# legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
end

# ╔═╡ 2dbcb518-2fda-44c4-bfc0-b422a8da9c35
let
	args_list = [
		Dict("numhidden"=>15, "replay_size"=>20000, "cell"=>"FacMARNN", "factors"=>10),
		Dict("numhidden"=>15, "replay_size"=>20000, "cell"=>"FacMAGRU", "factors"=>10)]
	violin(data_fac_sens, args_list; label_idx="cell", color = [cell_colors["FacMARNN"] cell_colors["FacMAGRU"]])
	dotplot!(data_fac_sens, args_list; label_idx="cell", color = [cell_colors["FacMARNN"] cell_colors["FacMAGRU"]])
	args_list_l = [
		Dict("numhidden"=>17, "truncation"=>12, "cell"=>"GRU"),
		Dict("numhidden"=>17, "truncation"=>12, "cell"=>"AAGRU"),
		Dict("numhidden"=>10, "truncation"=>12, "cell"=>"MAGRU"),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"RNN"),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AARNN"),
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"MARNN")]
	violin!(data_10_dist, args_list_l;
		label_idx="cell",
		color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),
		legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
	dotplot!(data_10_dist, args_list_l; label_idx="cell",
				color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :),)
	# dotplot!(data_10_dist, args_list_l;
	# 	label_idx="cell",
	# 	color=reshape(getindex.([cell_colors], getindex.(args_list_l, "cell")), 1, :))
end

# ╔═╡ 47c828a3-c468-4260-8ad7-17e6a40cf8fd
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN")
	]
	names = ["GRU", "AAGRU", "MAGRU", "FacGRU", "RNN", "AARNN", "MARNN", "FacRNN"]
	plt = plot()
	for i in 1:length(names)
		if i == 1
			plt = boxplot(data_dist_final_300k_t16, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
		else
			plt = boxplot!(data_dist_final_300k_t16, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
		end
		plt = dotplot!(data_dist_final_300k_t16, [args_list[i]];
			label=names[i],
			color=cell_colors[args_list[i]["cell"]], tickfontsize=11)
	end
	plt
				savefig("../data/paper_plots/dir_tmaze_online_box_plots_300k_steps_MUE_tau_16.pdf")
end

# ╔═╡ c4b9a22b-6954-46e9-aaa2-8000fe870006
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN")
	]
	names = ["GRU", "AAGRU", "MAGRU", "FacGRU", "RNN", "AARNN", "MARNN", "FacRNN"]
	plt = plot()
	for i in 1:length(names)
		if i == 1
			plt = boxplot(data_dist_final_300k, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
		else
			plt = boxplot!(data_dist_final_300k, [args_list[i]];
				label=names[i],
				color=cell_colors[args_list[i]["cell"]],
			legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false)
		end
		plt = dotplot!(data_dist_final_300k, [args_list[i]];
			label=names[i],
			color=cell_colors[args_list[i]["cell"]], tickfontsize=11)
	end
	plt
	
	savefig("../data/paper_plots/dir_tmaze_online_box_plots_300k_steps_MUE_tau_20.pdf")
end

# ╔═╡ c08d99c0-680c-495a-b7ae-46c907e8288e
let	
	names = ["GRU", "AAGRU", "MAGRU", "RNN", "AARNN", "MARNN"]
	
	args_list = [
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"GRU"),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AAGRU"),
		Dict("numhidden"=>15, "truncation"=>12, "cell"=>"MAGRU"),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"RNN"),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"AARNN"),
		Dict("numhidden"=>20, "truncation"=>12, "cell"=>"MARNN")]
	plt = plot()
	for i in 1:length(names)
		plt = violin!(data_10_dist_s_mue, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
		plt = boxplot!(data_10_dist_s_mue, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 12", ylim=(0.3, 1))

		
		#plt = dotplot!(data_dist_final_300k, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(0.4, 1.0))
		
	end
	plt
	
		savefig("../data/paper_plots/dir_tmaze_online_violin_and_box_plots_300k_steps_MUE_tau_12.pdf")
end

# ╔═╡ 99d4fbfa-61f6-444d-a525-eafc7e1bf55f
let	
	args_list = [
		Dict("cell"=>"GRU"),
		Dict("cell"=>"AAGRU"),
		Dict("cell"=>"MAGRU"),
		Dict("cell"=>"FacMAGRU"),
		Dict("cell"=>"RNN"),
		Dict("cell"=>"AARNN"),
		Dict("cell"=>"MARNN"),
		Dict("cell"=>"FacMARNN")
	]
	names = ["GRU", "AAGRU", "MAGRU", "FacGRU", "RNN", "AARNN", "MARNN", "FacRNN"]
	plt = plot()
	for i in 1:length(names)
		plt = violin!(data_dist_final_300k_t16, args_list[i], label=names[i], legend=false, color=cell_colors[args_list[i]["cell"]], lw=2, linecolor=cell_colors[args_list[i]["cell"]])
		
		plt = boxplot!(data_dist_final_300k_t16, args_list[i], label=names[i], color=color=cell_colors[args_list[i]["cell"]], fillalpha=0.75, outliers=true, lw=2, linecolor=:black, tickfontsize=10, grid=false, tickdir=:out, xguidefontsize=14, yguidefontsize=14, legendfontsize=10, titlefontsize=15, ylabel="Success Rate", title="Online Directional TMaze, τ: 16", ylim=(0.3, 1))
	if i == 4	
		plt = vline!([6], linestyle=:dot, color=:white, lw=2)
	end
		
		#plt = dotplot!(data_dist_final_300k, args_list[i], label=names[i], color=:black, tickdir=:out, grid=false, tickfontsize=11, ylims=(0.4, 1.0))
		
	end
	plt
	
	#savefig("../data/paper_plots/dir_tmaze_online_violin_and_box_plots_300k_steps_MUE_tau_16.pdf")
end

# ╔═╡ 76e66b40-5e08-48ef-ab84-780b238a15f1
let
	args_list = [
		Dict("numhidden_factors"=>[15, 37]),
		Dict("numhidden_factors"=>[15, 75]),
		Dict("numhidden_factors"=>[15, 100]),
		Dict("numhidden_factors"=>[20, 28]),
		Dict("numhidden_factors"=>[20, 75]),
		Dict("numhidden_factors"=>[20, 100]),
		Dict("numhidden_factors"=>[26, 21]),
		Dict("numhidden_factors"=>[26, 75]),
		Dict("numhidden_factors"=>[26, 100])
	]
	violin(data_10_dist_facmagru, args_list, make_label_string=true, label_idx="numhidden_factors", legend=false, color=[cell_colors["FacMAGRU"] cell_colors["FacMAGRU"] cell_colors["FacMAGRU"]], lw=2, linecolor=[cell_colors["FacMAGRU"] cell_colors["FacMAGRU"] cell_colors["FacMAGRU"]])
	boxplot!(data_10_dist_facmagru, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMAGRU"] cell_colors["FacMAGRU"] cell_colors["FacMAGRU"]], legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="FacGRU", fillalpha=0.75, outliers=true,)
	
	dotplot!(data_10_dist_facmagru, args_list, color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0), make_label_string=true, label_idx="numhidden_factors", xlabel="[hidden size, factors]", ylabel="Success Rate")

savefig("../data/paper_plots/dir_tmaze_online_facgru_box_plot.pdf")
end

# ╔═╡ bac07523-9f76-430b-b0e3-ec6d7e799640
let
	args_list = [
		Dict("numhidden_factors"=>[27, 40]),
		Dict("numhidden_factors"=>[27, 75]),
		Dict("numhidden_factors"=>[27, 100]),
		Dict("numhidden_factors"=>[36, 31]),
		Dict("numhidden_factors"=>[36, 75]),
		Dict("numhidden_factors"=>[36, 100]),
		Dict("numhidden_factors"=>[46, 24]),
		Dict("numhidden_factors"=>[46, 75]),
		Dict("numhidden_factors"=>[46, 100])
	]
	
		violin(data_10_dist_facmarnn, args_list, make_label_string=true, label_idx="numhidden_factors", legend=false, color=[cell_colors["FacMARNN"] cell_colors["FacMARNN"] cell_colors["FacMARNN"]], lw=2, linecolor=[cell_colors["FacMARNN"] cell_colors["FacMARNN"] cell_colors["FacMARNN"]])
	boxplot!(data_10_dist_facmarnn, args_list; make_label_string=true, label_idx="numhidden_factors", color = [cell_colors["FacMARNN"] cell_colors["FacMARNN"] cell_colors["FacMARNN"]], legend=false, lw=1.5, ylims=(0.4, 1.0), tickdir=:out, grid=false, title="FacRNN", fillalpha=0.75, outliers=true,)
	
	dotplot!(data_10_dist_facmarnn, args_list, color=:black, tickdir=:out, grid=false, tickfontsize=9, ylims=(0.4, 1.0), make_label_string=true, label_idx="numhidden_factors", xlabel="[hidden size, factors]", ylabel="Success Rate")
	
	savefig("../data/paper_plots/dir_tmaze_online_facrnn_box_plot.pdf")
end

# ╔═╡ Cell order:
# ╠═08118162-6995-45a0-91ed-f94eec767cd0
# ╠═f7f500a8-a1e9-11eb-009b-d7afdcade891
# ╠═e0d51e67-63dc-45ea-9092-9965f97660b3
# ╠═e53d7b29-788c-469c-9d44-573f996fa5e7
# ╟─0c746c1e-ea39-4415-a1b1-d7124b886f98
# ╠═1886bf05-f4be-4160-b61c-edf186a7f3cb
# ╠═fe50ffef-b691-47b5-acf8-8378fbf860a1
# ╠═6243517d-0b59-4329-903c-0c122d14a962
# ╟─834b1cf3-5b22-4e0a-abe9-61e427e6cfda
# ╠═eefd2ccc-ce9b-4cf8-991f-a58f2f932e99
# ╠═4c523f7e-f4a2-4364-a47b-45548878ff51
# ╠═ed00f8a0-f8f5-4340-8c6e-61ecb67e5d18
# ╠═f32ab9fa-1949-4eae-96ac-0a333f83daaa
# ╠═fedb725a-d13d-425f-bde9-531d453d8791
# ╠═7d7eee9b-ea92-46d6-87b3-652805dd6468
# ╠═da38497d-30db-4125-91f7-3bd94ae9e3da
# ╠═dd993256-e953-4fcb-bac2-e24dbc76086f
# ╠═61fc2ec1-94db-4cca-a805-cf3da445e65d
# ╠═892ce74d-d148-40cd-afea-019ea3a798ad
# ╠═35ae89bb-a66d-4fed-be4c-2f134b73afbf
# ╠═eab5c7a0-8052-432d-977e-68b967baf5ca
# ╠═6761f9ec-8b6b-48d3-9b2b-977799118ae0
# ╠═55d1416b-d580-4112-827a-30504c21f397
# ╠═f57df64d-0cd1-491e-8443-b504e217b477
# ╠═a8ff395b-4161-4be1-82e3-21316b550af4
# ╠═9cbe1fd5-1948-4d7c-92dc-e5999ba5c0e5
# ╠═66a3490e-a515-45c9-ba97-a308059d6467
# ╠═c437bc65-65cb-4486-8a98-3bf290ce3162
# ╠═f93f586c-bd83-4175-87ca-6e4278da5de4
# ╟─6f7494cb-7fb9-4ddc-a14f-e50364ae624a
# ╠═15a642f1-717c-4bff-8b7a-874b31b3bf1b
# ╠═c531d4fa-89f4-4bb8-8628-f0d9978247c8
# ╠═dd543d06-108f-4992-8afd-dc0ada7e7a88
# ╠═467ed417-cf69-493d-b21c-3fc4d1fb9907
# ╠═2040d613-0ea8-48ce-937c-9180073812ea
# ╟─a26a302e-bcbd-4671-9dc3-57cd92c0e6f8
# ╠═d4e9f6a6-f2fe-4145-b2ec-f15600d60f42
# ╠═7c1f8e58-4bfa-4da2-833d-0cc2a4ec74a4
# ╠═859b98dd-1024-4fec-8741-40edf72d0287
# ╠═f5a1e45c-398a-40e3-8c5f-b1d72ead0a89
# ╠═a4763342-d9bf-4d5c-81b9-a735ae3729e3
# ╠═6b253f85-9005-41b6-8544-e381f46285d8
# ╠═094b456c-a7ad-4f93-89a0-c403c73bdc55
# ╠═e2f179e9-5e75-4e6c-917b-07aef8b6f8b7
# ╠═4a64f3a7-3879-407a-9654-aacc1861fc42
# ╠═d50978f3-f38f-42af-a7d1-1e3b59e68175
# ╠═7425a013-fdbf-4d50-921d-64306d4c2741
# ╠═f5c3a38a-cf78-4b90-b371-506cc2997f92
# ╠═c05c7ffa-53cd-46bf-a661-886784eecc05
# ╠═90eba5d4-7a1c-4d89-b743-620ee31af023
# ╠═fc961ab4-c7c6-4e7b-91be-3e5fbb18d667
# ╠═f2960c43-c062-4c4a-8472-e0791adae9c6
# ╠═9a66b4f9-d10b-478c-bd58-63df5dbef47b
# ╠═db83dce1-29d5-42f8-b226-4412ce63c8f1
# ╠═c4d1ce22-d9e5-4505-8e74-c02e6e9ddefe
# ╠═30341781-351f-4b61-80a3-3f3c65f816e2
# ╠═c107cfa9-6ac9-40d4-8cf2-610779421e4f
# ╠═cb4d5803-996d-4341-80d0-3ef1bae52dc6
# ╠═9268f797-a997-4165-a57a-8defb4b751dd
# ╠═5096204e-5134-4fda-8894-20e0c1a3e650
# ╠═08552490-3e28-4f9c-8c98-c6f2a19fca6f
# ╠═fafaf816-d61e-4f52-8528-9b2b8e9fe971
# ╠═a6dc6873-8779-4e4e-856c-3dc818268acf
# ╠═c6636dd2-243d-40fa-99ac-a7982d7a0a9c
# ╠═c0d46ded-a7e0-4aef-b385-e53cffefedbb
# ╠═c171dcdb-0f21-4d63-8755-a3441434028b
# ╠═af855937-4069-4256-8682-f18d3568f8d7
# ╠═19af4769-3e99-4e69-8b9a-6b7ff63ef803
# ╠═68481a3f-ba2e-4d37-b838-370070b40fc5
# ╠═5456a07a-8d3d-4d4c-8001-b72aac54286e
# ╠═a8c60669-4401-4d56-8f23-39efee7030e4
# ╠═31a0ef43-deb1-41e1-ba7b-d1c25ef07bc8
# ╠═6962749e-0324-4edb-a4bf-f320ce1fb235
# ╠═5deaf434-1441-4cf7-bb20-80e4498c4ba2
# ╠═82dba65f-8792-4c16-8aab-c9de5c660f73
# ╠═6e7a223b-9af1-4160-a291-8799e3c963b2
# ╠═c6366560-ef13-4d6e-bc86-4861af5b559d
# ╠═b8bca610-afbb-4000-b603-3c917f823f36
# ╠═ee0bddea-ce0d-45c0-b5e0-f7202d46cb4f
# ╠═226cbd88-f9f4-4022-bab3-607e066e0f28
# ╠═d41b2113-ba79-4c8b-80b8-71e4be18ff69
# ╠═78d8672c-c0ad-4f64-b931-308a46b9479c
# ╟─f0068edb-4762-4e7b-850c-0c37e90ceb00
# ╠═59912f1e-6bf1-40b3-9c34-03a3b4542bd6
# ╟─413bfa10-4039-4b51-8ac9-f04006adc074
# ╠═c4c36ce4-aa29-4ba5-bba4-2b88362ed744
# ╟─74639e00-1076-4600-9565-3574a839d915
# ╟─98b89884-37f6-4476-bf41-e3aae3b40248
# ╟─41aef310-21d6-4e9f-8eb1-28005a736487
# ╟─0718582b-811f-4227-9f8c-2368881ac8dc
# ╠═6a8ea54b-86e6-4c38-8bda-7e39c6ef1605
# ╟─398e5230-e8d0-4261-85a9-5b049702a8ce
# ╟─8af2e436-a097-4fbb-81f5-17018f02b2fd
# ╠═090d428e-f2c1-4032-878d-df6de3063266
# ╠═99439b9e-49bb-48b8-bf38-3f1f229bcaa1
# ╠═9c649632-95df-4c0b-9501-bd660f92a1f6
# ╠═f5f0c142-8ef1-4abc-a0c9-40c927436942
# ╠═8520162d-6c0f-41af-84eb-63b1edcefb20
# ╠═6170c64c-9bfa-46dd-869c-6a989faf57c5
# ╠═f8388454-8c55-4404-a05f-a5e51ea225ca
# ╟─549a936c-e6d3-49a1-9172-f7abfc41956e
# ╟─5ac45ce9-7074-4df6-bc48-c3db35bb2fe3
# ╠═c8e944f2-ef02-4b81-8b6d-3527609f0822
# ╠═f4e6b374-f900-4545-b8a6-d95c85ab3596
# ╠═8a7d820c-7ca9-461d-9160-27c47496436a
# ╠═04655ac3-0d40-4063-bc8c-0826e13c6722
# ╠═f0390e50-13a6-496f-b0ab-97ed1b2b18e7
# ╠═ec7f5f9a-94be-473d-9542-8780c91f9e8e
# ╠═588cd0d2-c598-4e50-a1a0-93f9eddbc10e
# ╟─e0a3851d-3db5-440e-b67e-75ca7d9517e1
# ╠═ce18cc80-89da-4608-9f01-3ba94609049d
# ╠═6ef161be-ec61-461d-9c38-60510c08328b
# ╠═4f2e1222-9daa-439c-9d57-957f23e44657
# ╠═4d5dc03a-8b9b-4b5a-b002-6fc48a580b04
# ╠═b8be3138-98e3-476a-adcf-564196b51ae7
# ╠═f9e81f7b-26f3-4ce5-bca7-5054c8ddf44a
# ╠═305f8ac8-8f5f-4ec4-84a6-867f69a8887c
# ╠═533cba3d-7fc5-4d66-b545-b15ffc8ab6d8
# ╠═39752286-a6db-439d-aca0-1be4821bfc2b
# ╠═7d611b39-f9a8-43e4-951e-9d812cbd4384
# ╠═a8949e02-61f5-456a-abee-2bad91d2df05
# ╠═2dbcb518-2fda-44c4-bfc0-b422a8da9c35
# ╠═47c828a3-c468-4260-8ad7-17e6a40cf8fd
# ╠═c4b9a22b-6954-46e9-aaa2-8000fe870006
# ╠═9b400057-90ff-4b8b-8abf-f6fe1163f281
# ╠═e265c8f2-b147-4509-81a6-8c34a7de5457
# ╠═c08d99c0-680c-495a-b7ae-46c907e8288e
# ╠═99d4fbfa-61f6-444d-a525-eafc7e1bf55f
# ╠═76e66b40-5e08-48ef-ab84-780b238a15f1
# ╠═bac07523-9f76-430b-b0e3-ec6d7e799640
