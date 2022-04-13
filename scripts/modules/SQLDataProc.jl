module SQLDataProc


# using TerminalLoggers
using DataFrames, Query
using Statistics, ProgressLogging
using MySQL, DBInterface
using MacroTools
using FileIO

function connect(database=nothing)
    conn = DBInterface.connect(MySQL.Connection, "", "", option_file=joinpath(homedir(), ".my.cnf"))
    if !isnothing(database)
        DBInterface.execute(conn, "use $(database);");
    end
    conn
end

function use_database(conn, database)
    cur_db = DataFrame(DBInterface.execute(conn, "select DATABASE();"))[1, 1]
    if ismissing(cur_db) || cur_db != database
        DBInterface.execute(conn, "use $(database);")
    end
end

function list_databases(conn)
    DataFrame(DBInterface.execute(conn, "show databases;"))
end

function list_tables(conn)
    DataFrame(DBInterface.execute(conn, "show tables;"))
end

function get_param_table(conn)
    DataFrame(DBInterface.execute(conn, "select * from params;"))
end

function get_tables(conn)
    
end

function create_indexes(table_selector::Function, conn)
    
end


function proc_vec_data(conn, table, hashes=UInt[]; _pre_name="", kwargs...)

    if length(hashes) == 0
        hashes = DataFrame(DBInterface.execute(conn, "select _HASH from params;"))[!, :_HASH]
    end
    μ = zeros(Float32, length(hashes))
    sql_stmt = """select data
                  from $(table)
                  WHERE _HASH=?
                  ORDER BY step;
               """
    stmt = DBInterface.prepare(conn, sql_stmt)
    ret = DataFrame(DBInterface.execute(stmt, [hashes[1]]))[!, :data]
    ret_proc = [(k, f(ret)) for (k, f) in kwargs]
    get_initial = (x)->if x isa Number
        zeros(typeof(x), length(hashes))
    elseif x isa AbstractArray
        Vector{typeof(x)}(undef, length(hashes))
    end
    ret_strg = Dict(
        r[1]=>get_initial(r[2]) for r in ret_proc
    )

    @progress for (i, hsh) in enumerate(hashes)
        curs = DBInterface.execute(stmt, [hsh])
	data = DataFrame(curs)[!, :data]
        curs = nothing
        for (k, f) in kwargs
            ret_strg[k][i] = f(data)
        end
    end
    DBInterface.close!(stmt)
    DataFrame(;_HASH=hashes, (Symbol(_pre_name*string(k))=>v for (k, v) in ret_strg)...)
end

function get_res_data(conn, hashes=UInt[]; _pre_name="", kwargs...)
    
    if length(hashes) == 0
        hashes = DataFrame(DBInterface.execute(conn, "select _HASH from params;"))[!, :_HASH]
    end
    μ = zeros(Float32, length(hashes))
    sql_stmt = """select *
                  from results
                  WHERE _HASH=?;
               """
    stmt = DBInterface.prepare(conn, sql_stmt)
    ret = DataFrame(DBInterface.execute(stmt, [hashes[1]]))[!, :data]
    ret_proc = [(k, f(ret)) for (k, f) in kwargs]
    get_initial = (x)->if x isa Number
        zeros(typeof(x), length(hashes))
    elseif x isa AbstractArray
        Vector{typeof(x)}(undef, length(hashes))
    end
    ret_strg = Dict(
        r[1]=>get_initial(r[2]) for r in ret_proc
    )

    @progress for (i, hsh) in enumerate(hashes)
        curs = DBInterface.execute(stmt, [hsh])
	data = DataFrame(curs)[!, :data]
        curs = nothing
        for (k, f) in kwargs
            ret_strg[k][i] = f(data)
        end
    end
    DBInterface.close!(stmt)
    DataFrame(;_HASH=hashes, (Symbol(_pre_name*string(k))=>v for (k, v) in ret_strg)...)

end

function proc_control_data(database; save=nothing, vector_tables=["stps"=>"results_total_steps", "rews"=>"results_total_rews"])
    conn = connect(database)
    params = get_param_table(conn)
    
    results = [
        SQLDataProc.proc_vec_data(
            conn,
            vt;
            _pre_name=vt_n * "_",
            avg=(d)->mean(d),
            var=(d)->var(d),
            cnt=(d)->length(d), 
            avg_end=(d)->end_mean(d, 100),
            var_end=(d)->end_var(d, 100),
            cnt_end=(d)->end_count(d, 100)) for (vt_n, vt) in vector_tables]

    # results_rews = SQLDataProc.proc_vec_data(
    #     conn,
    #     "results_total_rews";
    #     _pre_name="rews_",
    #     avg=(d)->mean(d),
    #     var=(d)->var(d),
    #     avg_end=(d)->end_mean(d, 100),
    #     var_end=(d)->end_var(d, 100))

    DBInterface.close!(conn)
    params_and_results = DataFrames.innerjoin(params, results..., on=:_HASH)

    params_and_results = SQLDataProc.simplify_dataframe((d)->[collect(d)], params_and_results)
    if !isnothing(save)
        FileIO.save(save, "params_and_results", params_and_results)
    end
    params_and_results
end

function proc_pred_data(database; save=nothing)
    conn = connect(database)
    params = get_param_table(conn)
    results = get_results_table
    # SQLDataProc.proc_vec_data()
    
    DBInterface.close!(conn)
    params, results
end

function end_agg(agg::Function, d, num_steps::Int)
    if length(d) < num_steps
        agg(d)
    else
        agg(d[end-(num_steps-1):end])
    end
end

function end_agg(agg::Function, d, perc::AbstractFloat)
    @assert 0.0 >= perc <= 1.0
    num_steps = round(Int, length(d)*perc)
    if length(d) < num_steps
        agg(d)
    else
        agg(d[end-(num_steps-1):end])
    end
end

end_mean(d, num_steps_or_perc) = end_agg(mean, d, num_steps_or_perc)
end_var(d, num_steps_or_perc) = end_agg(var, d, num_steps_or_perc)
end_std_err(d, num_steps_or_perc) = end_agg(d, num_steps_or_perc) do (d)
    sqrt(var(d)/length(d))
end
end_count(d, num_steps_or_perc) = end_agg(length, d, num_steps_or_perc)


collapse_group(grp_df) = collapse_group_agg(grp_df) do c
    [collect(c)]
end

function collapse_group_agg(agg::Function, grp_df)
    values = []
    nms = names(grp_df)
    for n in nms
        vs = unique(grp_df[!, Symbol(n)])
        if length(vs) != 1
            push!(values, agg(skipmissing(grp_df[!, n])))
        else
            push!(values, vs[1])
        end
    end
    DataFrame(;(Symbol(n)=>v for (n, v) in zip(nms, values))...)
end

function simplify_dataframe(agg::Function, df; special_keys=["_HASH", "_GIT_INFO", "seed"])
    nms = filter((d)->d ∉ special_keys, names(df, Between(1, :_GIT_INFO)))
    grps = groupby(df, Symbol.(nms))
    reduce(vcat, [collapse_group_agg(agg, grp) for grp in grps])
end

function get_diff_dict(params)
    dfs = Dict{String, Any}()
    for clm in filter((clm)-> (clm != "_GIT_INFO") && (clm != "_HASH"), params[!, :COLUMN_NAME])
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


function best_from_sweep_param(o, df, sweep_params, avg_params=["seed"])
    nms = filter((d)->d ∉ avg_params && d ∉ sweep_params, names(df, Between(1, :_HASH))[1:end-1])
    grps = groupby(df, Symbol.(nms))
    reduce(vcat, [DataFrame(sort(grp, o)[1, :]) for grp in grps])
end    


end


