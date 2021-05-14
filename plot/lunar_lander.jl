#=
ringworld_online:
- Julia version: 
- Author: Vtkachuk
- Date: 2021-04-25
=#
module RingWorldPlotting

using JLD2
using Plots
using Statistics
using RollingFunctions

function plot_error(data_file, save_file; title="Prediction Error", window=1000)
   f=jldopen(data_file, "r")
   data = read(f, keys(f)[1])
   err = data["err"]
   avg_err = rollmean(err, window)
   plot(1:size(avg_err, 1), abs.(avg_err[:, 1:1]), title=title, ylim=(0, 0.5))
   savefig(save_file)
end

function plot_error_data(data, save_file; title="Prediction Error", window=1000)
   err = data["err"]
   plt = nothing
   for i in 1:size(err, 2)
       avg_err = rollmean(abs.(err[:, i]), window)
       if plt isa Nothing
           plt = plot(1:size(avg_err, 1), abs.(avg_err), title=title, ylim=(0, 0.5))
       else
           plt = plot!(plt, 1:size(avg_err, 1), abs.(avg_err), title=title, ylim=(0, 0.5))
       end
   end
   savefig(save_file)
end


end