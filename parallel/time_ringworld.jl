


using BSON: @save
using ProgressMeter

include("../experiment/ringworld.jl")

truncs = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]
# truncs = [1, 2, 3]
num_runs = 10

timings = zeros(length(truncs), num_runs)
args = RingWorldExperiment.default_arg_parse()
args["steps"] = 100000

args["rnn_config"] = "ARNN_OneHot"
# args["outhorde"] = "gammas_term"
args["outhorde"] = "onestep"
RingWorldExperiment.main_experiment(args)
args["progress"] = false

pb_τ = Progress(length(truncs), 0.5, "Truncations")


for (τ_idx, τ) ∈ enumerate(truncs)
    pb_r = Progress(length(truncs), 0.5, "Runs"; offset=1)
    for r ∈ 1:num_runs
        args["truncation"] = τ
        args["seed"] = r
        timings[τ_idx, r] = @elapsed RingWorldExperiment.main_experiment(args)
        next!(pb_r)
    end
    next!(pb_τ)
end

rnn_config = args["rnn_config"]
outhorde = args["outhorde"]
@save "timings_ringworld_$(rnn_config)_$(outhorde).bson" truncs timings


