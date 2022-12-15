using Documenter

# push!(LOAD_PATH,"../src/")
# push!(LOAD_PATH,"../experiment/")

import ActionRNNs: ActionRNNs, HelpfulKernelFuncs, ExpUtils
import DirectionalTMazeERExperiment, DirectionalTMazeInterventionExperiment, DirectionalTMazeLearnInterExperiment
import TMazeERExperiment
import RingWorldERExperiment, RingWorldERExperiment_FixRNG
import MaskedGridWorldERExperiment
import ImageDirectionalTMazeERExperiment
import LunarLanderExperiment

makedocs(
    sitename = "ActionRNNs",
    format = Documenter.HTML(),
    modules = [ActionRNNs,
               HelpfulKernelFuncs,
               RingWorldERExperiment,
               DirectionalTMazeERExperiment,
               MaskedGridWorldERExperiment,
               DirectionalTMazeInterventionExperiment,
               TMazeERExperiment,
               ImageDirectionalTMazeERExperiment,
               LunarLanderExperiment],
    pages = [
        "index.md",
        "library.md",
        "Experiment" =>
        [
            "experiments/ringworld.md",
            "experiments/directional_tmaze.md",
            "experiments/tmaze.md",
            "experiments/lunarlander.md",
            "experiments/image_dir_tmaze.md",
            "experiments/masked_gw.md",
        ]
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
# deploydocs(
#     repo = "github.com/mkschleg/ActionRNN.jl.git"
# )
