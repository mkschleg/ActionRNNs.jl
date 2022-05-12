


import SQLDataProc

function analyze_dir_tmaze(db, savepath;
                           vector_tables = ["stps"=>"results_total_steps",
                                            "rews"=>"results_total_rews",
                                            "successes"=>"results_successes"])

    SQLDataProc.proc_control_data(db;
                      save=savepath,
                      vector_tables=vector_tables)
end


