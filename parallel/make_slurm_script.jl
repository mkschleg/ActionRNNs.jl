import Pkg
Pkg.activate(".")


using Reproduce

function main()
    @add_arg_table as begin
        "config"
        arg_type=String
        "--path"
        arg_type=String
        default=""
        "--tasks"
        action=:store_true
        "--mem-per-task"
        arg_type=String
        default="4000M"
        "--ntasks"
        args_type=Int
        default=-1
        "--node"
        action=:store_true
        "--"
    end

    
end


slurm_header(email) = """
#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.2/bin/julia
#SBATCH --mail-user=$(email)
#SBATCH --mail-type=ALL
#SBATCH -o $().out # Standard output
#SBATCH -e $().err # Standard error
#SBATCH --time=5:00:00 # Running time of 24 hours
#SBATCH --account=rrg-whitem
"""


node_job = """
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=0
"""

ntasks_job() = """
#SBATCH --mem-per-cpu=4000M # Max memory per cpu
#SBATCH --ntasks=$()
"""


function construct_slurm_script()
    
end

