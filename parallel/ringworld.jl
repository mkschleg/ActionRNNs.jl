#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.3.0/bin/julia
#SBATCH --mail-user=mkschleg@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o ring_lstm.out # Standard output
#SBATCH -e ring_lstm.err # Standard error
#SBATCH --mem-per-cpu=2000M # Memory request of 2 GB
#SBATCH --time=12:00:00 # Running time of 12 hours
#SBATCH --ntasks=128
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

env_submit_dir =  get(ENV, "SLURM_SUBMIT_DIR", pwd())
include(joinpath(env_submit_dir, "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/ringworld.toml")

