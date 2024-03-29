#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH -o ringworld_rnn_gammas.out # Standard output
#SBATCH -e ringworld_rnn_gammas.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=06:00:00 # Running time of 12 hours
#SBATCH --ntasks=64
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

using Reproduce

<<<<<<< HEAD
const save_loc = "ringworld_arnn_sweep_cell_size_sgd"
const exp_file = "experiment/ringworld_flux_agent.jl"
const exp_module_name = :RingWorldFluxExperiment
const exp_func_name = :main_experiment
const optimizer = "Descent"
const alphas = clamp.(0.1*1.5.^(-6:2:6), 0.0, 1.0)
# const alphas = [0.01,  0.1]
const truncations = 1:2:15

const ringworld_sizes = [10, 15, 20]
=======
const save_loc = "ringworld_rnn_sweep_rmsprop"
const exp_file = "experiment/ringworld_flux_agent.jl"
const exp_module_name = :RingWorldFluxExperiment
const exp_func_name = :main_experiment
const optimizer = "RMSProp"
# const alphas = clamp.(0.1*1.5.^(-6:6), 0.0, 1.0)
const alphas = [0.0005, 0.001, 0.005, 0.01]
const truncations = [1, 2, 4, 6]

const ringworld_sizes = [6, 10, 20]
const hidden_state_sizes = [3, 6, 9, 12]
>>>>>>> 327e1ad7ca915b44538bf0e7d0f7ed0a49ffd895

function make_arguments(args::Dict)
    alpha = args["alpha"]
    cell = args["cell"]
    truncation = args["truncation"]
    seed = args["seed"]
    rw_size = args["size"]
    hs = args["hidden"]
    # save_file = "$(save_loc)/$(horde)/$(cell)/$(optimizer)_alpha_$(alpha)_truncation_$(truncation)/run_$(seed).jld2"
    new_args=["--truncation", truncation, "--opt", optimizer, "--optparams", alpha, "--cell", cell, "--seed", seed, "--size", rw_size, "--numhidden", hs]
    return new_args
end

function main()

    as = ArgParseSettings()
    @add_arg_table as begin
        "numworkers"
        arg_type=Int64
        default=1
        "--jobloc"
        arg_type=String
        default=joinpath(save_loc, "jobs")
        "--numjobs"
        action=:store_true
        "--numsteps"
        arg_type=Int64
        default=300000
        "--numruns"
        arg_type=Int64
        default=10
    end
    parsed = parse_args(as)
    num_workers = parsed["numworkers"]

    arg_dict = Dict([
        "alpha"=>alphas,
        "truncation"=>truncations,
        "cell"=>["ARNN"],
        "size"=>ringworld_sizes,
<<<<<<< HEAD
        "hsize"=>[1, 2, 4, 6, 8, 10, 12, 15, 20],
=======
        "hidden"=>hidden_state_sizes,
>>>>>>> 327e1ad7ca915b44538bf0e7d0f7ed0a49ffd895
        "seed"=>collect(1:parsed["numruns"])
    ])
    arg_list = ["size", "cell", "hidden", "alpha", "truncation", "seed"]

    static_args = ["--steps", string(parsed["numsteps"]), "--exp_loc", save_loc, "--outhorde", "gammas_term"]
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
