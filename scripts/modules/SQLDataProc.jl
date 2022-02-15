module SQLDataProc

using DataFrames, Query
using TerminalLoggers
using Statistics, ProgressLogging
using MySQL, DBInterface
using MacroTools

function connect(database)
    conn = DBInterface.connect(MySQL.Connection, "", "", option_file=joinpath(homedir(), ".my.cnf"))
    DBInterface.execute(conn, "use $(database);");
    conn
end

function get_param_table(conn)
    DBInterface.execute(conn, "select * from params;")
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
    ret_strg = Dict(
        r[1]=>zeros(typeof(r[2]), length(hashes)) for r in ret_proc
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


function collapse_group(grp_df)
    for n in names(grp_df)
        grp_df[]
    end
    
end


end


