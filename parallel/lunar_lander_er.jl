#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/bin/julia
#SBATCH --mail-user=vtkachuk@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o lunar_lander.out # Standard output
#SBATCH -e lunar_lander.err # Standard error
#SBATCH --mem-per-cpu=3000M # Memory request of 3 GB
#SBATCH --time=20:00:00 # Running time of 24 hours
#SBATCH --ntasks=256
#SBATCH --account=def-whitem

using Pkg
Pkg.activate(".")

env_submit_dir =  get(ENV, "SLURM_SUBMIT_DIR", pwd())
include(joinpath(env_submit_dir, "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/lunar_lander_er.toml";
save_path="/home/mkschleg/scratch/ActionRNNs")
