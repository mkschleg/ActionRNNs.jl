#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH -o tmaze.out # Standard output
#SBATCH -e tmaze.err # Standard error
#SBATCH --mem-per-cpu=3000M # Memory request of 3 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=256
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

using Reproduce

const save_loc = "tmaze_results"
const exp_file = "experiment/tmaze.jl"
const exp_module_name = :TMazeExperiment
const exp_func_name = :main_experiment
const optimizer = "RMSProp"
const alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01]
const truncations = [1, 2, 6, 10, 15, 20, 30]

const tmaze_sizes = [6, 10, 20]
const hidden_state_sizes = [3, 5, 6, 10, 15, 20, 30, 40]

function make_arguments(args::Dict)
    alpha = args["alpha"]
    cell = args["cell"]
    truncation = args["truncation"]
    seed = args["seed"]
    rw_size = args["size"]
    hs = args["hidden"]
    new_args=["--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--cell", cell, "--seed", seed, "--size", rw_size, "--numhidden", hs]
    return new_args
end

function main()

    as = ArgParseSettings()
    @add_arg_table as begin
        "numworkers"
        arg_type=Int64
        default=5
        "--jobloc"
        arg_type=String
        default=joinpath(save_loc, "jobs")
        "--numjobs"
        action=:store_true
        "--numsteps"
        arg_type=Int64
        default=200000
        "--numruns"
        arg_type=Int64
        default=10
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]

    arg_dict = Dict("alpha"=>alphas,
                    "truncation"=>truncations,
                    "cell"=>["RNN", "GRU", "LSTM", "ARNN"],
                    "size"=>tmaze_sizes,
                    "hidden"=>hidden_state_sizes,
                    "seed"=>collect(1:parsed["numruns"]))
    arg_list = ["size", "cell", "hidden", "alpha", "truncation", "seed"]

    static_args = ["--steps", string(parsed["numsteps"]), "--exp_loc", save_loc]
    args_iterator = ArgIterator(arg_dict, static_args; arg_list=arg_list, make_args=make_arguments)

    if parsed["numjobs"]
        @info "This experiment has $(length(collect(args_iterator))) jobs."
        println(collect(args_iterator)[num_workers])
        exit(0)
    end

    experiment = Experiment(save_loc,
                            exp_file,
                            exp_module_name,
                            exp_func_name,
                            args_iterator)

    create_experiment_dir(experiment)
    add_experiment(experiment; settings_dir="settings")
    ret = job(experiment; num_workers=num_workers, job_file_dir=parsed["jobloc"])
    post_experiment(experiment, ret)
end


main()
