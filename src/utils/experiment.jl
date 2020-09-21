

using JLD2
import ProgressMeter
# using Reproduce
using FileIO
import Reproduce

function save_setup(parsed;
                    save_dir_key="save_dir",
                    def_save_file="results.jld2",
                    filter_keys=["verbose",
                                 "working",
                                 "exp_loc",
                                 "visualize",
                                 "progress",
                                 "synopsis"])
    savefile = def_save_file
    working = get(parsed, "working", false)
    if !(working)
        Reproduce.create_info!(parsed,
                               parsed[save_dir_key];
                               filter_keys=filter_keys)
        savepath = Reproduce.get_save_dir(parsed)
        savefile = joinpath(savepath, def_save_file)
        if isfile(savefile)
            return nothing
        end
    end
    savefile
end

function pred_experiment(env, agent, rng, num_steps, parsed, err_func)
    s_t = start!(env, rng)
    action = start!(agent, s_t, rng)
    
    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")

    for step in 1:num_steps

        s_tp1, rew, term = step!(env, action, rng)
        out_preds, action = step!(agent, s_tp1, rew, term, rng)

        err_func(env, out_preds, step)
        
        if parsed["verbose"]
            println(step)
            println(env)
            println(agent)
        end

        if parsed["progress"]
           ProgressMeter.next!(prg_bar)
        end

        if parsed["visualize"]
            
        end
    end
end

function save_results(parsed::Dict, savefile, results)
    working = get(parsed, "working", false)
    if !(working)
        JLD2.@save savefile results
    else
        return results
    end
end
