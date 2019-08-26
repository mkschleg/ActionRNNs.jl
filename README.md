# ActionRNN.jl


## Setup

Clone repository locally, install Julia v1.1.x. In the ActionRNN folder.

Start the julia repl with `julia`

```julia
julia> ]add Revise, Plots
julia> ]activate .
julia> ]instantiate
```


## To run the example experiment

```julia
julia> using Revise; includet("experiment/ringworld_action_rnn.jl")
julia> ret = RingWorldRNNExperiment.main_experiment(["--truncation", "5", "--opt", "Descent", "--optparams", "0.1", "--cell", "RNN", "--seed", "1", "--steps", "300000", "--numhidden", "7", "--exp_loc", "ringworld_rnn_action_sweep_sgd", "--working", "--progress"])
```

This should run a Ring World experiment with the action RNN. This return a dictionary containing predictions and errors. You should easily be able to analyze the error and see relatively good performance here.

To analyze the data:

```julia
julia> using Statistics
julia> rmse = sqrt.(mean(ret["err"].^2; dims=2))
julia> mean(rmse)   # You should get 0.006968971...
```

To plot:

```julia
julia> using Plots
julia> gr()
julia> plot(rmse) # this will plot all the data points, and will be noisy
julia> plot(mean(reshape(rmse, 1000, 300); dims=1)') #this will plot a windowed average of points
```

