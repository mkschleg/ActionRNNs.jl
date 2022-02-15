using Documenter

import ActionRNNs: ActionRNNs, ExpUtils
import DirectionalTMazeERExperiment
import MaskedGridWorldERExperiment

makedocs(
    sitename = "ActionRNNs",
    format = Documenter.HTML(),
    modules = [ActionRNNs,
               DirectionalTMazeERExperiment,
               MaskedGridWorldERExperiment],
    pages = [
        "library.md",
        "Experiment" =>
        [
            "experiments/directional_tmaze.md",
            "experiments/masked_gw.md"
        ]
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/mkschleg/ActionRNN.jl.git"
)
