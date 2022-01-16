

include("../../experiment/ringworld_er.jl")
# const main_experiment = DirectionalTMazeERExperiment.main_experiment

RINGWORLD_ER_BASE_CONFIG =
    Dict{String,Any}(
        "save_dir" => joinpath(CONS_SAVE_DIR, "ringworld_er"),
        "seed" => 1,
        "synopsis" => false,
        
        "steps" => 10000,
        "size" => 6,

        # Network
        "numhidden" => 6,

        # Problem
        "outhorde" => "onestep",
        "outgamma" => 0.9,

        # Opt
        "opt" => "RMSProp",
        "eta" => 0.001,
        "rho" => 0.9,

        # BPTT
        "truncation" => 3,

        # Replay
        "replay_size"=>1000,
        "warm_up" => 1000,
        "batch_size"=>4,
        "update_freq"=>4,
        "target_update_freq"=>1000,
        "hs_learnable"=>true)

@testset "Consistency.RingworldER" begin

    ringworld_check(results, value) = sum(results["err"]) == value
    
    @testset "RNN" begin
        ret = @run_experiment RingWorldERExperiment  "RNN" Consistency.RINGWORLD_ER_BASE_CONFIG
        @test ringworld_check(ret, -785.4614f0)
    end

    @testset "AARNN" begin
        ret = @run_experiment RingWorldERExperiment  "AARNN" Consistency.RINGWORLD_ER_BASE_CONFIG
        @test ringworld_check(ret, -812.2295f0)
    end

    @testset "MARNN" begin
        ret = @run_experiment RingWorldERExperiment  "MARNN" Consistency.RINGWORLD_ER_BASE_CONFIG
        @test ringworld_check(ret, -802.99347f0)
    end

    @testset "GRU" begin
        ret = @run_experiment RingWorldERExperiment  "GRU" Consistency.RINGWORLD_ER_BASE_CONFIG
        @test ringworld_check(ret, -679.2683f0)
    end

    @testset "AAGRU" begin
        ret = @run_experiment RingWorldERExperiment  "AAGRU" Consistency.RINGWORLD_ER_BASE_CONFIG
        @test ringworld_check(ret, -778.9388f0)
    end

    @testset "MAGRU" begin
        ret = @run_experiment RingWorldERExperiment  "MAGRU" Consistency.RINGWORLD_ER_BASE_CONFIG
        @test ringworld_check(ret, -761.51f0)
    end

    @testset "FacMARNN" begin
        ret = @run_experiment RingWorldERExperiment  "FacMARNN" Consistency.RINGWORLD_ER_BASE_CONFIG factors=3
        @test ringworld_check(ret, -681.4659f0)
    end
    
    @testset "FacMAGRU" begin
        ret = @run_experiment RingWorldERExperiment  "FacMAGRU" Consistency.RINGWORLD_ER_BASE_CONFIG factors=3
        @test ringworld_check(ret, -640.77185f0)
    end

    
end
