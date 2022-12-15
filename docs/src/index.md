# ActionRNNs.jl

Documentation for ActionRNNs.jl. See the right for more details on all the different aspects of the project.

- General Documentation: All the library code necessary for building deep recurrent Q agents.
- Experiments: A non-exhaustive list of experiments, and some tips on running these experiments.


## Setup

Clone repository locally, install Julia v1.7.x. In the ActionRNN folder.

Start the julia repl with `julia`

```julia
julia> ]add Revise, Plots
julia> ]activate .
julia> ]instantiate
```


## To run the example experiment

```julia
julia> using Revise; includet("experiment/RingWorldERExperiment.jl")
julia> ret = RingWorldERExperiment.working_experiment()
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


## Consistency Tests


### Adding Consistency Tests


Adding a consistency test requires you know which experiment you are targeting, what cell, and the various arguments you want. For example. There is some magic macros in the ActionRNNsTests to make this simpler. Bellow is the easiest way to find the test for an experiment with an example in ringworld_er.  This workflow is still WIP, but this should get you all you need to add new cells to the files in `tests/consistency`.

```julia
push!(LOAD_PATH, "../ActionRNNs.jl/test/")
using ReTest
import ActionRNNsTests: ActionRNNsTests, Consistency.@run_experiment
begin
    ret = @run_experiment RingWorldERExperiment "FacMARNN" Consistency.RINGWORLD_ER_BASE_CONFIG factors=3
    sum(ret["err"])
end
```


### Running Consistency Tests

We use ReTest to run all our tests. This allows for testing certain experiments or cells across all experiments.

```julia
push!(LOAD_PATH, "../ActionRNNs.jl/test/")
using ReTest; import ActionRNNsTests
retest() # Run all tests
retest("AARNN") # Run all AARNN tests
retest("Ringworld") # Run all Ring World experiments
retest("Fac") # run all factored tests (including FacMA* and FacTucMA*)
```

## Documentation

We are using [Documenter.jl](https://juliadocs.github.io/Documenter.jl/stable/). Please see the documentation for documenter to figure out how it works. These docs are very much WIP, but for now we are putting all the docs in library.md raw and the index.md will have a bit more details to walk ppl through the different ideas in the repo.

To add docs all you need to do is add a documentation section above a function or struct and then add this to the appropriate docs section.

To build the documentation go to the `ActionRNNs/docs` directory and run:
```shell
$ julia --project make.jl
```

I'm still figuring out how to best host the docs.
