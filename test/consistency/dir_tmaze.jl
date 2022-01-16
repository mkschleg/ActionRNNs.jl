
# module DirTMazeConsistency

# import ..ActionRNNsTests: @run_experiment, CONS_SAVE_DIR



include("../../experiment/dir_tmaze_er.jl")
# const main_experiment = DirectionalTMazeERExperiment.main_experiment

DTMAZE_ER_BASE_CONFIG = 
    Dict{String,Any}(
        "save_dir" => joinpath(CONS_SAVE_DIR, "dir_tmaze_er"),
        # "_SAVE" => nothing,
        "seed" => 2,
        "steps" => 15000,
        "size" => 10,

        "cell" => "MAGRU",
        "numhidden" => 10,
        
        "opt" => "RMSProp",
        "eta" => 0.0005,
        "rho" =>0.99,
        
        "replay_size"=>10000,
        "warm_up" => 1000,
        "batch_size"=>8,
        "update_wait"=>4,
        "target_update_wait"=>1000,
        "truncation" => 12,
        
        "hs_learnable" => true,
        "gamma"=>0.99)

@testset "Consistency.DirTMazeER" begin

    dir_tmaze_check(results, value) = sum(results.save_results.total_rews) == value
    @testset "RNN" begin
        ret = @run_experiment DirectionalTMazeERExperiment  "RNN" Consistency.DTMAZE_ER_BASE_CONFIG
        @test dir_tmaze_check(ret, -1368.7032f0)
    end

    @testset "AARNN" begin
        ret = @run_experiment DirectionalTMazeERExperiment  "AARNN" Consistency.DTMAZE_ER_BASE_CONFIG
        @test dir_tmaze_check(ret, -1465.8024f0)
    end

    @testset "MARNN" begin
        ret = @run_experiment DirectionalTMazeERExperiment  "MARNN" Consistency.DTMAZE_ER_BASE_CONFIG
        @test dir_tmaze_check(ret, -1157.3124f0)
    end

    @testset "GRU" begin
        ret = @run_experiment DirectionalTMazeERExperiment  "GRU" Consistency.DTMAZE_ER_BASE_CONFIG
        @test dir_tmaze_check(ret, -849.3139f0)
    end

    @testset "AAGRU" begin
        ret = @run_experiment DirectionalTMazeERExperiment  "AAGRU" Consistency.DTMAZE_ER_BASE_CONFIG
        @test dir_tmaze_check(ret, -1297.6978f0)
    end

    @testset "MAGRU" begin
        ret = @run_experiment DirectionalTMazeERExperiment  "MAGRU" Consistency.DTMAZE_ER_BASE_CONFIG
        @test dir_tmaze_check(ret, -1173.9962f0)
    end


# end
end



