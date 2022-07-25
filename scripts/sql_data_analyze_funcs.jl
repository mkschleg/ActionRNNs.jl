using Statistics

import SQLDataProc
import ItemColToDataFrame
import FileIO
import DBInterface
import DataFrames

import StatsBase: StatsBase, percentile


function analyze_dir_tmaze(db, savepath)
    vector_tables = ["stps"=>"results_total_steps",
                     "rews"=>"results_total_rews",
                     "successes"=>"results_successes"]

    SQLDataProc.proc_control_data(db;
                      save=savepath,
                      vector_tables=vector_tables)
end

function analyze_dir_tmaze_ic(dir, savepath=nothing;
                              result_vectors = ["stps"=>:total_steps,
                                                "rews"=>:total_rews,
                                                "successes"=>:successes])
    df = ItemColToDataFrame.proc_to_data_frame(dir) do res
        ret = Pair{String}[]
        for rv in result_vectors
            d = res["results"][rv.second]
            _pre_name=rv.first * "_"
            
            pd = [_pre_name*"avg"=>mean(d),
                  _pre_name*"var"=>var(d),
                  _pre_name*"cnt"=>length(d),
                  _pre_name*"avg_end"=>SQLDataProc.end_mean(d, 0.1),
                  _pre_name*"var_end"=>SQLDataProc.end_var(d, 0.1),
                  _pre_name*"cnt_end"=>SQLDataProc.end_count(d, 0.1)]
            append!(ret, pd)
            
        end
        ret
    end

    params_and_results = SQLDataProc.simplify_dataframe((d)->[collect(d)], df)

    if !isnothing(savepath)
        FileIO.save(savepath, "params_and_results", params_and_results)
    end
    params_and_results
end

function analyze_mgw(db, savepath;
                     vector_tables = ["stps"=>"results_total_steps",
                                      "rews"=>"results_total_rews",
                                      "test_stps"=>"results_test_total_steps",
                                      "test_rews"=>"results_test_total_rews"])

    SQLDataProc.proc_control_data(db;
                                  save=savepath,
                                  vector_tables=vector_tables)
end


function analyze_ring_world(database, savepath=nothing)
    
    conn = SQLDataProc.connect(database)
    params = SQLDataProc.get_param_table(conn)

    vector_tables = ["lc"=>"results_lc"] #, "var"=>"results_var"]
    
    # proc_vec_data
    results = [
        SQLDataProc.proc_vec_data(
            conn,
            vt;
            _pre_name=vt_n * "_",
            identity=(d)->Float32.(d)) for (vt_n, vt) in vector_tables]
    res_numbers = SQLDataProc.DataFrame(
        SQLDataProc.DBInterface.execute(
            conn,
            "SELECT _HASH, avg_all, avg_end from results order by _HASH"))

    SQLDataProc.DBInterface.close!(conn)
    
    mid = SQLDataProc.DataFrames.innerjoin(params, res_numbers, results..., on=:_HASH)
    params_and_results = SQLDataProc.simplify_dataframe((d)->[collect(d)], mid)

    if !isnothing(savepath)
        SQLDataProc.FileIO.save(savepath, "params_and_results", params_and_results)
    end
    params_and_results
end

function analyze_ringworld_ic(dir, savepath=nothing)
    df = ItemColToDataFrame.proc_to_data_frame(dir) do res
        d = res["results"]
        ["lc_identity"=>[d["lc"]],
         "avg_end"=>d["end"],
         "avg_all"=>d["all"]]
    end

    params_and_results = SQLDataProc.simplify_dataframe((d)->[collect(d)], df)

    if !isnothing(savepath)
        FileIO.save(savepath, "params_and_results", params_and_results)
    end
    params_and_results
end


function analyze_lunar_lander_ic(dir, savepath=nothing)
    
    df = ItemColToDataFrame.proc_to_data_frame(dir) do res
        ret = Pair{String}[]
        for rv in ["rews"=>:total_rews, "steps"=>:total_steps]
            d = res["results"][rv.second]
            _pre_name=rv.first * "_"

            pd = [_pre_name*"avg"=>mean(d),
                  _pre_name*"avg_end"=>SQLDataProc.end_mean(d, 0.1),
                  _pre_name*"identity"=>[d]]
            
            append!(ret, pd)
        end

        ret
    end

    params_and_results = SQLDataProc.simplify_dataframe((d)->[collect(d)], df)

    if !isnothing(savepath)
        FileIO.save(savepath, "params_and_results", params_and_results)
    end
    params_and_results
end


# function analyze_dir_tmaze(db, savepath;
#                            vector_tables = ["stps"=>"results_total_steps",
#                                             "rews"=>"results_total_rews",
#                                             "successes"=>"results_successes"])

#     SQLDataProc.proc_control_data(db;
#                       save=savepath,
#                       vector_tables=vector_tables)
# end
function analyze_dir_tmaze_intervention(db, savepath=nothing)

    conn = SQLDataProc.connect(db)
    params = SQLDataProc.get_param_table(conn)
    
    results = [SQLDataProc.proc_matrix_data(conn, "results_inter_total_steps", _pre_name="total_steps_") do mat
        int_mat = Int.(mat)
        [
            ("identity", int_mat),
            ("mean", mean(int_mat; dims=1)),
            ("var", var(int_mat; dims=1)),
            ("median", mean(int_mat; dims=1)),
            ("25_perc", [percentile(col, 25) for col in eachcol(int_mat)]),
            ("75_perc", [percentile(col, 75) for col in eachcol(int_mat)])
        ]
    end,
    SQLDataProc.proc_matrix_data(conn, "results_inter_successes", _pre_name="successes_") do mat
        bool_mat = Bool.(mat)
        [
            ("mean", mean(bool_mat; dims=1))
        ]
    end]

    DBInterface.close!(conn)
    params_and_results = DataFrames.innerjoin(params, results..., on=:_HASH)

    params_and_results = SQLDataProc.simplify_dataframe((d)->begin
                                                        c = collect(d)
                                                        # @info typeof(c)
                                                        typeof(c)[collect(d)]
                                                        end, params_and_results)
    if !isnothing(savepath)
        FileIO.save(savepath, "params_and_results", params_and_results)
    end
    params_and_results
end

function analyze_dir_tmaze_intervention(db, savepath=nothing)

    conn = SQLDataProc.connect(db)
    params = SQLDataProc.get_param_table(conn)
    
    results = [SQLDataProc.proc_matrix_data(conn, "results_inter_total_steps", _pre_name="total_steps_") do mat
        int_mat = Int.(mat)
        [
            ("identity", int_mat),
            ("mean", mean(int_mat; dims=1)),
            ("var", var(int_mat; dims=1)),
            ("median", mean(int_mat; dims=1)),
            ("25_perc", [percentile(col, 25) for col in eachcol(int_mat)]),
            ("75_perc", [percentile(col, 75) for col in eachcol(int_mat)])
        ]
    end,
    SQLDataProc.proc_matrix_data(conn, "results_inter_successes", _pre_name="successes_") do mat
        bool_mat = Bool.(mat)
        [
            ("mean", mean(bool_mat; dims=1)),
            ("identity", bool_mat)
        ]
    end]

    DBInterface.close!(conn)
    params_and_results = DataFrames.innerjoin(params, results..., on=:_HASH)

    params_and_results = SQLDataProc.simplify_dataframe((d)->begin
                                                        c = collect(d)
                                                        # @info typeof(c)
                                                        typeof(c)[collect(d)]
                                                        end, params_and_results)
    if !isnothing(savepath)
        FileIO.save(savepath, "params_and_results", params_and_results)
    end
    params_and_results
end

function analyze_dir_tmaze_intervention_steps(db, savepath=nothing)

    conn = SQLDataProc.connect(db)
    params = SQLDataProc.get_param_table(conn)
    
    results = [SQLDataProc.proc_nested_vec_data(conn, "results_inter_total_steps", _pre_name="total_steps_") do vecs
        int_mat = [Int.(vec) for vec in vecs]
        [
            ("identity", int_mat),
        ]
    end,
    SQLDataProc.proc_nested_vec_data(conn, "results_inter_successes", _pre_name="successes_") do vecs
        bool_mat = [Bool.(vec) for vec in vecs]
        [
            ("identity", bool_mat)
        ]
    end]

    DBInterface.close!(conn)
    params_and_results = DataFrames.innerjoin(params, results..., on=:_HASH)

    params_and_results = SQLDataProc.simplify_dataframe((d)->begin
                                                        c = collect(d)
                                                        # @info typeof(c)
                                                        typeof(c)[collect(d)]
                                                        end, params_and_results)
    if !isnothing(savepath)
        FileIO.save(savepath, "params_and_results", params_and_results)
    end
    params_and_results
end


function analyze_dir_tmaze_combo_sm(db, savepath=nothing)

    conn = SQLDataProc.connect(db)
    params = SQLDataProc.get_param_table(conn)
    
    results = [SQLDataProc.proc_matrix_data(conn, "results_AgentModel_rnn_sm_w_a", _pre_name="sm_w_a_") do mat
               int_mat = Float32.(mat)
               [
                   ("identity", identity(int_mat))
               ]
               end,
               SQLDataProc.proc_matrix_data(conn, "results_AgentModel_rnn_sm_w_m", _pre_name="sm_w_m") do mat
               int_mat = Float32.(mat)
               [
                   ("identity", int_mat)
               ]
               end]

    DBInterface.close!(conn)
    params_and_results = DataFrames.innerjoin(params, results..., on=:_HASH)

    params_and_results = SQLDataProc.simplify_dataframe((d)->begin
                                                        c = collect(d)
                                                        # @info typeof(c)
                                                        typeof(c)[collect(d)]
                                                        end, params_and_results)
    if !isnothing(savepath)
        FileIO.save(savepath, "params_and_results", params_and_results)
    end
    params_and_results
end
