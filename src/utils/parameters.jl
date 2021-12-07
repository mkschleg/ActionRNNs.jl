using ActionRNNs
using Random
using Flux 

function default_config()
    [
        Dict{String,Any}(
            "cell" => "MAGRU",
            "numhidden" => 11,
        )
        Dict{String,Any}(
            "cell" => "MARNN",
            "numhidden" => 17,
        )
        Dict{String,Any}(
            "cell" => "MARNN",
            "numhidden" => 17,
            "deep" => true,
            "internal_a" => 10,
            "internal_o" => 17,
        )
    ]
end


function main(env_name)
    if env_name == "dir_tmaze_er"
        include("../../experiment/dir_tmaze_er.jl")
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
