module ActionRNNsConsistency

using ReTest, ActionRNNs

# jld2_save_manager(save_dir) = Reproduce.FileSave(save_dir, Reproduce.JLD2Manager())
const CONS_SAVE_DIR = "cons_test"

macro run_experiment(mod, cell_type, baseargs, args...)

    aakws = Pair{Symbol,Any}[]
    for el in args
        if Meta.isexpr(el, :(=))
            push!(aakws, Pair(el.args...))
        end
    end
    
    quote
        args = copy($baseargs)
        args["cell"] = String($cell_type)
        for (k, v) in $aakws
            args[String(k)] = v
        end
        ret = $(mod).main_experiment(args, testing=true, overwrite=true)
    end
end


include("consistency/ringworld_er.jl")
include("consistency/dir_tmaze.jl")
include("consistency/image_dir_tmaze.jl")


end

# @testset "RingWorldER" begin
#     include("../experiment/ringworld_er.jl")
#     @testset "MAGRU" begin
#         @test test_dir_tmaze()
#     end

#     @testset "AAGRU" begin
#         @test test_dir_tmaze()
#     end
# end
