

using JLD2
import ProgressMeter
# using Reproduce
import Reproduce

macro save_setup(parsed, def_save_file="results.jld2")
    quote
        local savefile = $def_save_file
        if !($parsed["working"])
            Reproduce.create_info!($parsed, $parsed["exp_loc"]; filter_keys=["verbose", "working", "exp_loc"])
            local savepath = Reproduce.get_save_dir($parsed)
            local savefile = joinpath(savepath, "results.jld2")
            if isfile(savefile)
                return
            end
        end
        savefile
    end
end


function save_setup(parsed, def_save_file="results.jld2")
    savefile = def_save_file
    if !(parsed["working"])
        create_info!(parsed, parsed["exp_loc"]; filter_keys=["verbose", "working", "exp_loc"])
        savepath = Reproduce.get_save_dir(parsed)
        savefile = joinpath(savepath, "results.jld2")
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
    end
end

function save_results(parsed::Dict, savefile, results)
    if !parsed["working"]
        JLD2.@save savefile results
    else
        return results
    end
end
