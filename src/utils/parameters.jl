using ActionRNNs
using Random
using Flux 
include("../../experiment/dir_tmaze_er.jl")
include("../../experiment/ringworld_er.jl")
include("../../experiment/lunar_lander.jl")
include("../../experiment/viz_dir_tmaze.jl")

function dir_tmaze_er_args()
    [
        Dict{String,Any}(
            "cell" => "AARNN",
            "numhidden" => 30,
            "deep" => false,
        )
        Dict{String,Any}(
            "cell" => "MARNN",
            "numhidden" => 18,
            "deep" => false,
        )
        Dict{String,Any}(
            "cell" => "AAGRU",
            "numhidden" => 17,
            "deep" => false,
        )
        Dict{String,Any}(
            "cell" => "MAGRU",
            "numhidden" => 10,
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
            "cell" => "MARNN",
            "numhidden" => 18,
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
        Dict{String,Any}(
            "cell" => "MAGRU",
            "numhidden" => 10,
            "deep" => true,
            "internal_a" => 6,
            "internal_o" => 12,
        )
    ]
end

function ringworld_er_args()
    [
        Dict{String,Any}(
            "cell" => "AARNN",
            "numhidden" => 15,
            "deep" => false,
        )
        Dict{String,Any}(
            "cell" => "MARNN",
            "numhidden" => 12,
            "deep" => false,
        )
        Dict{String,Any}(
            "cell" => "AAGRU",
            "numhidden" => 12,
            "deep" => false,
        )
        Dict{String,Any}(
            "cell" => "MAGRU",
            "numhidden" => 9,
            "deep" => false,
        )
        Dict{String,Any}(
            "cell" => "AARNN",
            "numhidden" => 15,
            "deep" => true,
            "internal_a" => 6,
            "internal_o" => 12,
        )
        Dict{String,Any}(
            "cell" => "MARNN",
            "numhidden" => 12,
            "deep" => true,
            "internal_a" => 6,
            "internal_o" => 12,
        )
        Dict{String,Any}(
            "cell" => "AAGRU",
            "numhidden" => 12,
            "deep" => true,
            "internal_a" => 6,
            "internal_o" => 12,
        )
        Dict{String,Any}(
            "cell" => "MAGRU",
            "numhidden" => 9,
            "deep" => true,
            "internal_a" => 6,
            "internal_o" => 12,
        )
    ]
end

function lunar_lander_args()
    [
        Dict{String,Any}(
            "cell" => "AAGRU",
            "numhidden" => 152,
            "encoding_size" => 128,
            "deep" => false,
        )
        Dict{String,Any}(
            "cell" => "AAGRU",
            "numhidden" => 152,
            "encoding_size" => 128,
            "deep" => true,
            "internal_a" => 64,
        )
        Dict{String,Any}(
            "cell" => "MAGRU",
            "numhidden" => 64,
            "encoding_size" => 128,
            "deep" => false,
        )
        Dict{String,Any}(
            "cell" => "MAGRU",
            "numhidden" => 64,
            "encoding_size" => 128,
            "deep" => true,
            "internal_a" => 64,
        )

    ]
end

function viz_dir_tmaze_args()
    [
        Dict{String,Any}(
            "cell" => "AAGRU",
            "numhidden" => 70,
            "latent_size" => 128,
            "output_size" => 128,
            "deep" => false,
        )
        Dict{String,Any}(
            "cell" => "AAGRU",
            "numhidden" => 70,
            "latent_size" => 128,
            "output_size" => 128,
            "deep" => true,
            "internal_a" => 64,
        )
    ]
end


function main(env_name)
    if env_name == "dir_tmaze_er"
        module_name = DirectionalTMazeERExperiment
        env = DirectionalTMaze(10)
        rng = Random.MersenneTwister(1)
        args_list = dir_tmaze_er_args()
        parsed = module_name.default_config()
        for args in args_list
            for (key, value) in args
                parsed[key] = value
            end
            agent = module_name.construct_agent(env, parsed, rng)
            num_params = size(Flux.destructure(agent.model)[1])
            println("parsed: $(args)")
            println("number of parameters: $(num_params)")
        end

    elseif env_name == "ringworld_er"
        module_name = RingWorldERExperiment 
        rng = Random.MersenneTwister(1)
        args_list = ringworld_er_args()
        parsed = module_name.default_config()
        for args in args_list
            for (key, value) in args
                parsed[key] = value
            end
            agent = module_name.construct_agent(parsed, rng)
            num_params = size(Flux.destructure(agent.model)[1])
            println("parsed: $(args)")
            println("number of parameters: $(num_params)")
        end

    elseif env_name == "lunar_lander"
        module_name = LunarLanderExperiment 
        env = ActionRNNs.LunarLander(1, false)
        rng = Random.MersenneTwister(1)
        args_list = lunar_lander_args()
        parsed = module_name.default_config()
        for args in args_list
            for (key, value) in args
                parsed[key] = value
            end
            agent = module_name.construct_agent(env, parsed, rng)
            num_params = size(Flux.destructure(agent.model)[1])
            println("parsed: $(args)")
            println("number of parameters: $(num_params)")
        end

    elseif env_name == "viz_dir_tmaze"
        module_name = VisualDirectionalTMazeERExperiment 
        env = ImageDirTMaze(10)
        rng = Random.MersenneTwister(1)
        args_list = viz_dir_tmaze_args()
        parsed = module_name.default_config()
        for args in args_list
            for (key, value) in args
                parsed[key] = value
            end
            agent = module_name.construct_agent(env, parsed, rng)
            num_params = size(Flux.destructure(agent.model)[1])
            println("parsed: $(args)")
            println("number of parameters: $(num_params)")
        end
    end

end
