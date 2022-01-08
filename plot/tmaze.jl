#=
tmaze_online:
- Julia version: 
- Author: Vtkachuk
- Date: 2021-05-04
=#
module TMazePlotting

using JLD2
using Plots
using Statistics
using RollingFunctions

function plot_successes(data_file, save_file; title="Prediction Error", avg=100)
   f=jldopen(data_file, "r")
   data = read(f, keys(f)[1])
   successes = data[:save_results][:successes]
   avg_successes = rollmean(successes, avg)
   plot(1:size(avg_successes, 1), avg_successes, title=title, ylim=(0.4, 1.0))
   savefig(save_file)

end

function plot_successes_data(data, save_file; title="Successes", window=100)
   successes = data[:save_results][:successes]
   avg_successes = rollmean(successes, window)
   plot(1:size(avg_successes, 1), avg_successes, title=title, ylim=(0.4, 1.0))
   savefig(save_file)
end

function plot_successes_data_(data, save_file; title="Successes", avg=100)
   successes = data[:successes]
   avg_successes = rollmean(successes, avg)
   plot(1:size(avg_successes, 1), avg_successes, title=title, ylim=(0.4, 1.0))
   savefig(save_file)
end

end