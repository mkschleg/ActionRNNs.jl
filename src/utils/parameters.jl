using ActionRNNs
using Random
using Flux 
include("../../experiment/dir_tmaze_er.jl")

function default_config()
    [
        Dict{String,Any}(
            "cell" => "AARNN",
            "numhidden" => 30,
            "deep" => false,
        )
        Dict{String,Any}(
            "cell" => "AAGRU",
            "numhidden" => 17,
            "deep" => false,
        )
        Dict{String,Any}(
            "cell" => "AARNN",
            "numhidden" => 30,
            "deep" => true,
            "internal_a" => 6,
            "internal_o" => 12,
        )
        Dict{String,Any}(
            "cell" => "AAGRU",
            "numhidden" => 17,
            "deep" => true,
            "internal_a" => 6,
            "internal_o" => 12,
        )
    ]
end


function main(env_name)
    if env_name == "dir_tmaze_er"
        parsed = DirectionalTMazeERExperiment.default_config()
        env = DirectionalTMaze(10)
        rng = Random.MersenneTwister(1)
        args_list = default_config()
        for args in args_list
            for (key, value) in args
                parsed[key] = value
            end
            agent = DirectionalTMazeERExperiment.construct_agent(env, parsed, rng)
            num_params = size(Flux.destructure(agent.model)[1])
            println("parsed: $(args)")
            println("number of parameters: $(num_params)")
        end
    end
end
