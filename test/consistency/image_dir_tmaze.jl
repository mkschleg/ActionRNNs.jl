

include("../../experiment/viz_dir_tmaze.jl")

IMGDIRTMAZE_BASE_CONFIG = Dict{String,Any}(
        "save_dir" => "tmp/viz_dir_tmaze",

        "seed" => 1,
        "steps" => 2000,
        "size" => 10,

        "numhidden" => 64,
        "latent_size" => 128,
        "output_size" => 128,

        "opt" => "ADAM",
        "eta" => 0.00005,
        "rho" =>0.99,

        "replay_size"=>100,
        "warm_up" => 100,
        "batch_size"=>16,
        "update_wait"=>4,
        "target_update_wait"=>500,
        "truncation" => 15,

        "hs_learnable" => true,
        "gamma"=>0.9)

image_dir_tmaze_rew_check(results, value) = sum(results.save_results.total_rews) == value
image_dir_tmaze_loss_check(results, value) = sum(results.save_results.losses) == value

@testset "Consistency.ImageDirTMaze" begin
    
    @testset "AAGRU" begin
        ret = @run_experiment VisualDirectionalTMazeERExperiment "AAGRU" IMGDIRTMAZE_BASE_CONFIG
        @test image_dir_tmaze_rew_check(ret, -1987.0f0) && image_dir_tmaze_loss_check(ret, 3.8237302f0)
    end

    @testset "MAGRU" begin
        ret = @run_experiment VisualDirectionalTMazeERExperiment "MAGRU" IMGDIRTMAZE_BASE_CONFIG
        @test image_dir_tmaze_rew_check(ret, -2008.0f0) && image_dir_tmaze_loss_check(ret, 2.0507567f0)
    end

    @testset "FacMAGRU" begin
        ret = @run_experiment VisualDirectionalTMazeERExperiment "FacMAGRU" IMGDIRTMAZE_BASE_CONFIG factors=8 init_style="tensor"
        @test image_dir_tmaze_rew_check(ret, -2000.0f0) && image_dir_tmaze_loss_check(ret, 4.3609962f0)
    end

end
