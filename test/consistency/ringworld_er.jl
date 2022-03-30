

# include("../../experiment/ringworld_er.jl")
# const main_experiment = DirectionalTMazeERExperiment.main_experiment
import RingWorldERExperiment

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

    ringworld_check(results, value) = begin
        @test sum(results.save_results[:out_err]) == value
    end
    
    @testset "RNN" begin
        ret = @run_experiment RingWorldERExperiment  "RNN" RINGWORLD_ER_BASE_CONFIG
        ringworld_check(ret, -785.46136f0)
    end

    @testset "AARNN" begin
        ret = @run_experiment RingWorldERExperiment  "AARNN" RINGWORLD_ER_BASE_CONFIG
        ringworld_check(ret, -812.2294f0)
    end

    @testset "MARNN" begin
        ret = @run_experiment RingWorldERExperiment  "MARNN" RINGWORLD_ER_BASE_CONFIG
        ringworld_check(ret, -802.9936f0)
    end

    @testset "GRU" begin
        ret = @run_experiment RingWorldERExperiment  "GRU" RINGWORLD_ER_BASE_CONFIG
        ringworld_check(ret, -679.2681f0)
    end

    @testset "AAGRU" begin
        ret = @run_experiment RingWorldERExperiment  "AAGRU" RINGWORLD_ER_BASE_CONFIG
        ringworld_check(ret, -778.93884f0)
    end

    @testset "MAGRU" begin
        ret = @run_experiment RingWorldERExperiment  "MAGRU" RINGWORLD_ER_BASE_CONFIG
        ringworld_check(ret, -761.51f0)
    end

    @testset "FacMARNN" begin
        ret = @run_experiment RingWorldERExperiment  "FacMARNN" RINGWORLD_ER_BASE_CONFIG factors=3
        ringworld_check(ret, -681.4658f0)
    end
    
    @testset "FacMAGRU" begin
        ret = @run_experiment RingWorldERExperiment  "FacMAGRU" RINGWORLD_ER_BASE_CONFIG factors=3
        ringworld_check(ret, -640.77185f0)
    end

    @testset "CaddRNN" begin
        ret = @run_experiment RingWorldERExperiment  "CaddRNN" RINGWORLD_ER_BASE_CONFIG
        ringworld_check(ret, -637.20636f0)
    end

    @testset "CcatRNN" begin
        ret = @run_experiment RingWorldERExperiment  "CcatRNN" RINGWORLD_ER_BASE_CONFIG
        ringworld_check(ret, -1139.1787f0)
    end

    
    
end
