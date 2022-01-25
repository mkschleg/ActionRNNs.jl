using Documenter
using ActionRNNs

makedocs(
    sitename = "ActionRNNs",
    format = Documenter.HTML(),
    modules = [ActionRNNs]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/mkschleg/ActionRNN.jl.git"
)
