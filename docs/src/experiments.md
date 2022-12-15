# Experiments

There are many experiments run in the paper associated with this paper. All the experiments are run using [Reproduce.jl][https://github.com/mkschleg/Reproduce.jl]. To run a particular config (see the configs and final_runs folders).

To run a specific config on your own machine from the ActionRNNs.jl root folder: 
```
julia --project parallel/toml_parallel.jl <<config>>
```

- configs: This includes configs for all sweeps run in this empirical analysis. If a config doesn't run 
- final_runs: This includes all the configs and hyperparameters for gathering the final runs after the hyperparameter sweep. 







