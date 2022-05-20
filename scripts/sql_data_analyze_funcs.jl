

import SQLDataProc

function analyze_dir_tmaze(db, savepath;
                           vector_tables = ["stps"=>"results_total_steps",
                                            "rews"=>"results_total_rews",
                                            "successes"=>"results_successes"])

    SQLDataProc.proc_control_data(db;
                      save=savepath,
                      vector_tables=vector_tables)
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

