
using Plots
using Reproduce
using FileIO
using ProgressMeter
using Logging

function heatmap_ring(loc::AbstractString, y_axis, x_axis;
                      static_params::Dict{String, Any}=Dict{String,Any}(), clean_data=(data)->data, data_type=Float64)

    ic = ItemCollection(loc)

    _, hashes, subset_items = search(ic, static_params)

    subset_ic = ItemCollection(subset_items)
    
    diff_dict = diff(subset_items)
    y_axis_params = diff_dict[y_axis]
    x_axis_params = diff_dict[x_axis]

    data_matrix = Array{data_type}(undef, length(y_axis_params), length(x_axis_params))
    
    for ((y_param_idx, y_param), (x_param_idx, x_param)) ∈ Iterators.product(enumerate(reverse(y_axis_params)), enumerate(x_axis_params))

        _, setting_hashes, setting_items =
            search(subset_ic, Dict(y_axis=>y_param, x_axis=>x_param))

        res_sum = 0.0
        i = 0
        for si ∈ setting_items
            try
                res_sum += clean_data(load(joinpath(dirname(si.folder_str), "results.jld2")))
                i+=1
            catch
                @info "Can't open: $(si.folder_str)"
            end
        end
        data_matrix[y_param_idx, x_param_idx] = res_sum/i
    end

    data_matrix
end

data_ring(loc::AbstractString, args...; kwargs...) =
    data_ring(ItemCollection(loc), args...; kwargs...)


function data_ring(ic::ItemCollection, y_axis, x_axis, sweep_key;
                   static_params::Dict{String, Any}=Dict{String,Any}(), clean_data=(data)->data, data_type=Float64)

    subset_ic = search(ic, static_params)

    # subset_ic = ItemCollection(subset_items)
    
    diff_dict = diff(subset_ic)
    y_axis_params = diff_dict[y_axis]
    x_axis_params = diff_dict[x_axis]
    sweep_params = diff_dict[sweep_key]

    data_matrix = Array{data_type}(undef, length(y_axis_params), length(x_axis_params))
    param_matrix = Array{Tuple{eltype(y_axis_params), eltype(x_axis_params)}}(undef, length(y_axis_params), length(x_axis_params))
    best_matrix = Array{eltype(sweep_params)}(undef, length(y_axis_params), length(x_axis_params))
    
    for ((y_param_idx, y_param), (x_param_idx, x_param)) ∈ Iterators.product(enumerate(y_axis_params), enumerate(x_axis_params))

        param_matrix[y_param_idx, x_param_idx] = (y_param, x_param)
        res_sum = zeros(length(sweep_params))

        for (sp_idx, sp) ∈ enumerate(sweep_params)
            setting_ic =
                search(subset_ic, Dict(y_axis=>y_param, x_axis=>x_param, sweep_key=>sp))
            i = 0
            for si ∈ setting_ic
                # res_sum[sp_idx] += clean_data(load(joinpath(dirname(si.folder_str), "results.jld2")))
                try
                    res_sum[sp_idx] += clean_data(FileIO.load(joinpath(si.folder_str, "results.jld2")))
                    i += 1
                catch
                    @info "Can't open: $(si.folder_str)"
                end
            end
            res_sum[sp_idx] /= i
        end
        res_min = findmin(res_sum)
        data_matrix[y_param_idx, x_param_idx] = res_min[1]
        best_matrix[y_param_idx, x_param_idx] = sweep_params[res_min[2]]
    end

    data_matrix, best_matrix, param_matrix, (y_axis_params, x_axis_params)
    # heatmap(string.(x_axis_params), string.(y_axis_params), data_matrix, ylabel=y_axis, xlabel=x_axis)
end

heatmap_ring(loc::AbstractString, args...; kwargs...) =
    heatmap_ring(ItemCollection(loc), args...; kwargs...)

function heatmap_ring(ic::ItemCollection, y_axis, x_axis, sweep_key;
                      static_params=Dict{String,Any}(),
                      clean_data=(data)->data,
                      data_type=Float64,
                      kwargs...)
    data_matrix, _, _, (y_axis_params, x_axis_params) = data_ring(ic, y_axis, x_axis, sweep_key; static_params=static_params, clean_data=clean_data, data_type=data_type)
    heatmap(string.(x_axis_params),
            string.(y_axis_params),
            data_matrix,
            ylabel=y_axis,
            xlabel=x_axis; kwargs...)
end

