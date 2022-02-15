### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ 17054590-968b-46e7-90ef-11658ee77b45
using Statistics, ProgressLogging

# ╔═╡ 4a1a0874-8abf-11ec-1376-61d3d6a661ea
using MySQL, DBInterface

# ╔═╡ 413fb9e0-a463-4139-b529-a6ed6960295b
using DataFrames

# ╔═╡ aa30dad0-760f-4c03-ab25-ad29d2ebe4a7
using Query

# ╔═╡ ff20fbaa-89a8-4a18-937d-567b377f9c93


# ╔═╡ 3821a118-dbde-4a8c-81bc-d39a5f7fc8b3
# conn = DBInterface.connect(
# 	MySQL.Connection, 
# 	"127.0.0.1", 
# 	"mkschleg", 
# 	"jlli=+CvSTw6rNqo", 
# 	port=8000)

# ╔═╡ fa412d9d-b2d0-422b-aeee-bbb8caf53ad9
begin
	conn = DBInterface.connect(MySQL.Connection, "", "", option_file=joinpath(homedir(), ".my.cnf"))
	DBInterface.execute(conn, "use mkschleg_masked_gw_baseline;");
end;

# ╔═╡ 2687b558-352b-4992-9f9d-6d209f6705f7
DataFrame(DBInterface.execute(conn, "show databases;"))

# ╔═╡ b57f1a0b-e7b6-45bc-982c-5f26f0a6473f
DataFrame(DBInterface.execute(conn, "show tables;"))

# ╔═╡ fa0509d8-6145-429e-8b10-cc39c7133771
DBInterface.execute(conn, "use mkschleg_masked_gw_baseline;");

# ╔═╡ 8c1a0800-3455-4077-a58a-8ee30a6f2314
clms = DataFrame(DBInterface.execute(conn, 
	"SELECT COLUMN_NAME
	  FROM INFORMATION_SCHEMA.COLUMNS
	  WHERE TABLE_SCHEMA = 'mkschleg_masked_gw_baseline' AND TABLE_NAME = 'params';"))

# ╔═╡ 2c9ca959-8aa3-4dfc-9c83-0f07bd11420c
diff_dict = let
	dfs = Dict{String, Any}()
	for clm in filter((clm)-> (clm != "_GIT_INFO") && (clm != "_HASH"), clms[!, :COLUMN_NAME])
		df = DataFrame(
			DBInterface.execute(
				conn,
				"SELECT DISTINCT($(clm)) FROM params"))
		if length(df[:, clm]) > 1
			dfs[clm] = df[:, clm]
		end
	end
	dfs
end

# ╔═╡ ef57a791-a694-4602-a416-859151854bcb
params = DataFrame(DBInterface.execute(conn, "SELECT * FROM params"))

# ╔═╡ c0667f22-e002-4cd5-b497-9b3fc9ee5912
results_total_rews = DataFrame(DBInterface.execute(conn, "SELECT * FROM results_total_rews"))

# ╔═╡ 0fa27631-d1b4-463d-ab79-e91b884c4839
size(results_total_rews)

# ╔═╡ 555edd9e-82d4-4fe6-8711-83a1c0c6e7bb
q1 = @from i in params begin
	@where i[:cell] == "MARNN" && i.batch_size == 8
	@select {i._HASH}
	@collect DataFrame
end

# ╔═╡ 48c4ecbc-149b-49b0-b318-08f5f966a506
function get_hashes(df, search_dict)
	get_hashes(df) do (i)
		all([i[Symbol(k)] == v for (k, v) in search_dict])
	end
end

# ╔═╡ cc3b2db9-9a73-4d76-91d9-c49411d310d6
function get_hashes(func::Function, df::DataFrame)
	@from i in df begin
		@where func(i)
		@select {i._HASH}
		@collect DataFrame
	end
end

# ╔═╡ 1e60c40a-17b4-44a3-a65e-f0c8fef61106
get_hashes(params, 
		   Dict(k=>v[1] for (k, v) in filter((k) -> k.first != "seed", diff_dict)))

# ╔═╡ d6dececf-7993-41b9-9ce1-44bbe6747589
let
	sans_seed_dd = filter((k) -> k.first != "seed", diff_dict)
	df = DataFrame(;[Symbol(k) => eltype(v)[] for (k, v) in sans_seed_dd]...)
	ks = keys(sans_seed_dd)
	x = Tuple{Float32, Float32}[]
	@progress for vs in collect(Iterators.product(values(sans_seed_dd)...))
		hashes = get_hashes(params, 
			Dict(k=>vs[i] for (i, k) in enumerate(ks))
		)[!, :_HASH]
		d = Float32[]
		for hsh in hashes
			df2 = DataFrame(DBInterface.execute(conn, 
				"""
				SELECT data 
				FROM results_total_rews
				WHERE _HASH=$(hsh)
				ORDER BY step ASC;
				"""
			))
			push!(d, mean(df2[end-100:end, :data]))
		end
		push!(x, (mean(d), var(d)))
	end
	x
end

# ╔═╡ d6214513-abd2-48fa-835c-9b7c1f9f2e9b
function sql_step_avg_over_hashes(hashes, vec_table_name; )
	
end

# ╔═╡ 015fdc90-03ff-42c3-982c-4010490b02ca
avg_df = let
	# df = DataFrame(_HASH=UInt[], mean=Float32[])
	μ = zeros(Float32, length(params[!, :_HASH]))
	sql_stmt = """
			SELECT AVG(A.data)
			FROM (
				select data
				from results_total_rews
				WHERE _HASH=?
				ORDER BY step DESC LIMIT 100) A;
			"""
	stmt = DBInterface.prepare(conn, sql_stmt)
	@progress for (i, hsh) in enumerate(params[!, :_HASH])
		μ[i] = DataFrame(
			DBInterface.execute(stmt, [hsh]))[1, "AVG(A.data)"]
	end
	DBInterface.close!(stmt)
	DataFrame(_HASH=params[:, :_HASH], mean_end=μ)
end

# ╔═╡ 1652e818-730c-42be-94bf-4c4ac2d8dfa0
(hsh=params[1:100, :_HASH],)

# ╔═╡ 12b67ff3-435a-4546-a0b6-34db1a4bd6dd
let
	# df = DataFrame(_HASH=UInt[], mean=Float32[])
	μ = zeros(Float32, length(params[!, :_HASH]))
	@progress for (i, hsh) in enumerate(params[1:100, :_HASH])
		μ[i] = mean(DataFrame(DBInterface.execute(conn, 
				"""
				select data
				from results_total_rews
				WHERE _HASH=$(hsh);
				"""))[end-99:end, :data])
	end
	DataFrame(_HASH=params[:, :_HASH], mean_end=μ)
end

# ╔═╡ d5995a29-c0c8-435b-a576-3720bba78ff5
eltype(params[!, :_HASH])

# ╔═╡ 40649832-17f0-4861-8ee3-c448823f794d
let # proc data
	df2 = DataFrame(DBInterface.execute(conn, 
				"""
				select *
				from results_total_rews
				group by _HASH
				ORDER BY step DESC;
				"""))
end

# ╔═╡ ff39ab7e-d945-4653-ac97-0b278aebca84


# ╔═╡ 57da9788-8eef-4e1c-9d99-759c4876c2a8
let
df2 = mean(DataFrame(DBInterface.execute(conn, 
				"""
				SELECT data
				FROM results_total_rews
				WHERE _HASH=14554276979958612940
				ORDER BY step DESC LIMIT 100;
				"""
))[!, :data])
end

# ╔═╡ 1d5574e6-7ea1-4ea3-a405-714e1148c46d
	df2 = DataFrame(DBInterface.execute(conn, 
				"""
				SELECT data 
				FROM results_total_rews
				WHERE _HASH=14554276979958612940
				ORDER BY step;
				"""
			))

# ╔═╡ 1a7a37b3-27a5-4c1e-b012-678e8f809fc7
get_hashes(params, 
			Dict(k=>diff_dict[k][end] for (i, k) in enumerate(keys(diff_dict))))

# ╔═╡ 6504e16a-4df5-4e50-b3cf-f00248846006
let
df2 = DataFrame(DBInterface.execute(conn, 
				"""
				SELECT data, step
				FROM results_total_rews
				WHERE _HASH=14554276979958612940
				ORDER BY step ASC;
				"""
			))
	mean(df2[!, :data])
end

# ╔═╡ d9ff0083-2c1d-45ce-a0a3-7d7a97b5787e
# let
# 	df = DataFrame()
# 	sans_seed_dd = filter((k) -> k.first != "seed", diff_dict)
# 	DataFrame(;[Symbol(k) => eltype(v)[] for (k, v) in sans_seed_dd]...)
# 	ks = keys(sans_seed_dd)
# 	x = Any[]
# 	for vs in collect(Iterators.product(values(sans_seed_dd)...))[1:2]
# 		hashes = get_hashes(params, 
# 			Dict(k=>vs[i] for (i, k) in enumerate(ks))
# 		)[!, :_HASH]
# 		d = []
# 		for hsh in hashes
# 			df = DataFrame(DBInterface.execute(conn, 
# 				"""
# 				SELECT data 
# 				FROM results_total_rews
# 				WHERE _HASH='$(hsh)';
# 				"""
# 			))
# 			push!(d, mean(df[!, :data][end-100:end]))
# 		end
# 		push!(x, (mean(d), variance(d)))
# 	end
# 	x
# end

# ╔═╡ d061a6bc-0af9-4c93-8e2d-e8b4dea14e6b
DataFrame(;[Symbol(k) => eltype(v)[] for (k, v) in diff_dict]...)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DBInterface = "a10d1c49-ce27-4219-8d33-6db1a4562965"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
MySQL = "39abe10b-433b-5dbd-92d4-e302a9df00cd"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Query = "1a8c2f83-1ff3-5112-b086-8aa67b057ba1"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DBInterface = "~2.5.0"
DataFrames = "~1.3.2"
MySQL = "~1.2.1"
ProgressLogging = "~0.1.4"
Query = "~1.0.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f9982ef575e19b0e5c7a98c6e75ee496c0f73a93"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.12.0"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[DBInterface]]
git-tree-sha1 = "9b0dc525a052b9269ccc5f7f04d5b3639c65bca5"
uuid = "a10d1c49-ce27-4219-8d33-6db1a4562965"
version = "2.5.0"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "ae02104e835f219b8930c7664b8012c93475c340"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DecFP]]
deps = ["DecFP_jll", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a836d5799ae9772d74861755d95df306a3056cac"
uuid = "55939f99-70c6-5e9b-8bb0-5071ed7d61fd"
version = "1.2.0"

[[DecFP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9982cddeac476ed4ead831baff7b3f9d084579c1"
uuid = "47200ebd-12ce-5be5-abb7-8e082af23329"
version = "2.0.3+0"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterableTables]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Requires", "TableTraits", "TableTraitsUtils"]
git-tree-sha1 = "70300b876b2cebde43ebc0df42bc8c94a144e1b4"
uuid = "1c8ee90f-4401-5389-894e-7a04a3dc0f4d"
version = "1.0.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[MariaDB_Connector_C_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "Libiconv_jll", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "30523a011413b08ee38a6ed6a18e84c363d0ce79"
uuid = "aabc7e14-95f1-5e66-9f32-aea603782360"
version = "3.1.12+0"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MySQL]]
deps = ["BinaryProvider", "DBInterface", "Dates", "DecFP", "Libdl", "MariaDB_Connector_C_jll", "Parsers", "Tables"]
git-tree-sha1 = "6aef9b11d71dfb0cd8c70ce37c703d904a3b5c0c"
uuid = "39abe10b-433b-5dbd-92d4-e302a9df00cd"
version = "1.2.1"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "648107615c15d4e09f7eca16307bc821c1f718d8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.13+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0b5cfbb704034b5b4c1869e36634438a047df065"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.1"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[Query]]
deps = ["DataValues", "IterableTables", "MacroTools", "QueryOperators", "Statistics"]
git-tree-sha1 = "a66aa7ca6f5c29f0e303ccef5c8bd55067df9bbe"
uuid = "1a8c2f83-1ff3-5112-b086-8aa67b057ba1"
version = "1.0.0"

[[QueryOperators]]
deps = ["DataStructures", "DataValues", "IteratorInterfaceExtensions", "TableShowUtils"]
git-tree-sha1 = "911c64c204e7ecabfd1872eb93c49b4e7c701f02"
uuid = "2aef5ad7-51ca-5a8f-8e88-e75cf067b44b"
version = "0.9.3"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "8d0c8e3d0ff211d9ff4a0c2307d876c99d10bdf1"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.2"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableShowUtils]]
deps = ["DataValues", "Dates", "JSON", "Markdown", "Test"]
git-tree-sha1 = "14c54e1e96431fb87f0d2f5983f090f1b9d06457"
uuid = "5e66a065-1f0a-5976-b372-e0b8c017ca10"
version = "0.2.5"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[TableTraitsUtils]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Missings", "TableTraits"]
git-tree-sha1 = "78fecfe140d7abb480b53a44f3f85b6aa373c293"
uuid = "382cd787-c1b6-5bf2-a167-d5b971a19bda"
version = "1.0.2"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═17054590-968b-46e7-90ef-11658ee77b45
# ╠═4a1a0874-8abf-11ec-1376-61d3d6a661ea
# ╠═413fb9e0-a463-4139-b529-a6ed6960295b
# ╠═aa30dad0-760f-4c03-ab25-ad29d2ebe4a7
# ╠═ff20fbaa-89a8-4a18-937d-567b377f9c93
# ╠═3821a118-dbde-4a8c-81bc-d39a5f7fc8b3
# ╠═fa412d9d-b2d0-422b-aeee-bbb8caf53ad9
# ╠═2687b558-352b-4992-9f9d-6d209f6705f7
# ╠═b57f1a0b-e7b6-45bc-982c-5f26f0a6473f
# ╠═fa0509d8-6145-429e-8b10-cc39c7133771
# ╠═8c1a0800-3455-4077-a58a-8ee30a6f2314
# ╠═2c9ca959-8aa3-4dfc-9c83-0f07bd11420c
# ╠═ef57a791-a694-4602-a416-859151854bcb
# ╠═c0667f22-e002-4cd5-b497-9b3fc9ee5912
# ╠═0fa27631-d1b4-463d-ab79-e91b884c4839
# ╠═555edd9e-82d4-4fe6-8711-83a1c0c6e7bb
# ╠═48c4ecbc-149b-49b0-b318-08f5f966a506
# ╠═cc3b2db9-9a73-4d76-91d9-c49411d310d6
# ╠═1e60c40a-17b4-44a3-a65e-f0c8fef61106
# ╠═d6dececf-7993-41b9-9ce1-44bbe6747589
# ╠═d6214513-abd2-48fa-835c-9b7c1f9f2e9b
# ╠═015fdc90-03ff-42c3-982c-4010490b02ca
# ╠═1652e818-730c-42be-94bf-4c4ac2d8dfa0
# ╠═12b67ff3-435a-4546-a0b6-34db1a4bd6dd
# ╠═d5995a29-c0c8-435b-a576-3720bba78ff5
# ╠═40649832-17f0-4861-8ee3-c448823f794d
# ╠═ff39ab7e-d945-4653-ac97-0b278aebca84
# ╠═57da9788-8eef-4e1c-9d99-759c4876c2a8
# ╠═1d5574e6-7ea1-4ea3-a405-714e1148c46d
# ╠═1a7a37b3-27a5-4c1e-b012-678e8f809fc7
# ╠═6504e16a-4df5-4e50-b3cf-f00248846006
# ╠═d9ff0083-2c1d-45ce-a0a3-7d7a97b5787e
# ╠═d061a6bc-0af9-4c93-8e2d-e8b4dea14e6b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
