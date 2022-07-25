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

function list_databases(conn = nothing)
    close = false
    if isnothing(conn)
        conn = connect()
        close = true
    end
    df = DataFrame(DBInterface.execute(conn, "show databases;"))
    if close
        DBInterface.close!(conn)
    end
    df
end

function list_tables(conn)
    DataFrame(DBInterface.execute(conn, "show tables;"))
end

function get_param_table(conn)
    DataFrame(DBInterface.execute(conn, "select * from params;"))
end

function get_results_table(conn)
    DataFrame(DBInterface.execute(conn, "select * from results;"))
end

function get_tables(conn)
    
end

function create_indexes(table_selector::Function, conn)
    
end


function proc_matrix_data(proc, conn, table, hashes=UInt[]; _pre_name="")

    if length(hashes) == 0
        hashes = DataFrame(DBInterface.execute(conn, "select _HASH from params;"))[!, :_HASH]
    end
    μ = zeros(Float32, length(hashes))
    sql_stmt = """select data, step_1, step_2
                  from $(table)
                  WHERE _HASH=?
                  ORDER BY step_1, step_2;
               """
    stmt = DBInterface.prepare(conn, sql_stmt)
    dat = DataFrame(DBInterface.execute(stmt, [hashes[1]]))
    cols = Int(maximum(dat[!, :step_1]))
    proc_dat = proc(reshape(dat[!, :data], :, cols))
    get_initial = (x)->if x isa Number
        zeros(typeof(x), length(hashes))
    elseif x isa AbstractArray
        Vector{typeof(x)}(undef, length(hashes))
    end

    ret_strg = Dict(
        r[1]=>get_initial(r[2]) for r in proc_dat
    )

    @progress for (i, hsh) in enumerate(hashes)
        curs = DBInterface.execute(stmt, [hsh])
        dat = DataFrame(DBInterface.execute(stmt, hsh))
        cols = Int(maximum(dat[!, :step_1]))
        proc_dat = proc(reshape(dat[!, :data], :, cols))
        curs = nothing
        for (k, d) ∈ proc_dat
            ret_strg[k][i] = d
        end
    end
    DBInterface.close!(stmt)
    DataFrame(;_HASH=hashes, (Symbol(_pre_name*string(k))=>v for (k, v) in ret_strg)...)
end

function proc_nested_vec_data(proc, conn, table, hashes=UInt[]; _pre_name="")

    if length(hashes) == 0
        hashes = DataFrame(DBInterface.execute(conn, "select _HASH from params;"))[!, :_HASH]
    end
    μ = zeros(Float32, length(hashes))
    sql_stmt = """select data, step_1, step_2
                  from $(table)
                  WHERE _HASH=?
                  ORDER BY step_1, step_2;
               """
    stmt = DBInterface.prepare(conn, sql_stmt)
    dat = DataFrame(DBInterface.execute(stmt, [hashes[1]]))
    cols = Int(maximum(dat[!, :step_1]))
    proc_dat = proc([dat[dat.step_1 .== i, :data] for i in 1:cols])

    get_initial = (x)->if x isa Number
        zeros(typeof(x), length(hashes))
    elseif x isa AbstractArray
        Vector{typeof(x)}(undef, length(hashes))
    end

    ret_strg = Dict(
        r[1]=>get_initial(r[2]) for r in proc_dat
    )

    @progress for (i, hsh) in enumerate(hashes)
        curs = DBInterface.execute(stmt, [hsh])
        dat = DataFrame(DBInterface.execute(stmt, hsh))
        cols = Int(maximum(dat[!, :step_1]))
        proc_dat = proc([dat[dat.step_1 .== i, :data] for i in 1:cols])
        # proc_dat = proc(reshape(dat[!, :data], :, cols))
        curs = nothing
        for (k, d) ∈ proc_dat
            ret_strg[k][i] = d
        end
    end
    DBInterface.close!(stmt)
    DataFrame(;_HASH=hashes, (Symbol(_pre_name*string(k))=>v for (k, v) in ret_strg)...)
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

function get_vec_data(proc::Function, conn, table, hashes=UInt[]; _pre_name="", kwargs...)

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
            avg_end=(d)->end_mean(d, 0.1),
            var_end=(d)->end_var(d, 0.1),
            cnt_end=(d)->end_count(d, 0.1)) for (vt_n, vt) in vector_tables]


    DBInterface.close!(conn)
    params_and_results = DataFrames.innerjoin(params, results..., on=:_HASH)

    params_and_results = SQLDataProc.simplify_dataframe((d)->begin
                                                        c = collect(d)
                                                        # @info typeof(c)
                                                        typeof(c)[collect(d)]
                                                        end, params_and_results)
    if !isnothing(save)
        FileIO.save(save, "params_and_results", params_and_results)
    end
    params_and_results
end

function proc_data(database, vec_tables, non_vec_tables, save=nothing)
    conn = connect(database)
    params = get_param_table(conn)
    results = get_results_table(conn)

    vector_tables = ["lc"=>"results_lc"] #, "var"=>"results_var"]
    
    # proc_vec_data
    results = [
        SQLDataProc.proc_vec_data(
            db_conn,
            vt;
            _pre_name=vt_n * "_",
            identity=(d)->Float32.(d)) for (vt_n, vt) in vector_tables]
    res_numbers = DataFrame(DBInterface.execute(db_conn, "SELECT _HASH, avg_all, avg_end from results order by _HASH"))

    mid = DataFrames.innerjoin(params, res_numbers, results..., on=:_HASH)
    params_and_results = SQLDataProc.simplify_dataframe((d)->[collect(d)], mid)
    
    
    SQLDataProc.proc_vec_data()
    
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
    @assert 0.0 <= perc <= 1.0
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


