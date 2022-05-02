using PackageCompiler
using Pkg

Pkg.activate(".")

using Flux, StatsPlots, Plots

create_sysimage(["Flux", "StatsPlots", "Plots"], sysimage_path="NotebookSysimage.so")
