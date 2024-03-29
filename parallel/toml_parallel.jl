using Pkg
Pkg.activate(".")

using Reproduce, ActionRNNs
using ArgParse

function main()

    as = ArgParseSettings()
    @add_arg_table as begin
        "config"
        arg_type=String
        "--path"
        arg_type=String
        default=""
        "--numworkers"
        arg_type=Int
        default=4
        "--numjobs"
        action=:store_true
    end
    parsed = parse_args(as)
    
    experiment = Reproduce.parse_experiment_from_config(parsed["config"], parsed["path"])

    pre_experiment(experiment)
    ret = job(experiment; num_workers=parsed["numworkers"])
    post_experiment(experiment, ret)

end

main()
