#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/bin/julia
#SBATCH --mail-user=vtkachuk@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o ringworld_er.out # Standard output
#SBATCH -e ringworld_er.err # Standard error
#SBATCH --mem-per-cpu=3000M # Memory request of 2 GB
#SBATCH --time=5:00:00 # Running time of 24 hours
#SBATCH --ntasks=128
#SBATCH --account=def-whitem

using Pkg
Pkg.activate(".")

env_submit_dir =  get(ENV, "SLURM_SUBMIT_DIR", pwd())
include(joinpath(env_submit_dir, "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/ringworld_er/ringworld_er_10.toml";
                            save_path="/home/vladtk/scratch/ActionRNNs")
