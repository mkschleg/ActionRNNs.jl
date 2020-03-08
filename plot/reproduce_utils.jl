using Plots
using Reproduce
using FileIO
using Statistics
using ProgressMeter


function get_best_setting(ic, sweep_param, clean_func)

    diff_dict = diff(ic)

    μ = zeros(length(diff_dict[sweep_param]))
    σ = zeros(length(diff_dict[sweep_param]))
    
    for (swprm_idx, swprm) ∈ enumerate(diff_dict[sweep_param])
        _, _, itms = search(ic, Dict(sweep_param=>swprm))
        res = zeros(length(itms))
        for (itm_idx, itm) ∈ enumerate(itms)
            try
                res[itm_idx] = clean_func(
                    load(joinpath(dirname(itm.folder_str), "results.jld2")))
            catch
                res[itm_idx] = Inf
            end
        end
        μ[swprm_idx] = mean(res)
        σ[swprm_idx] = std(res)./sqrt(length(itms))
    end
    min_idx = findmin(μ)
    diff_dict[sweep_param][min_idx[2]], (min_idx[1], σ[min_idx[2]])
end


"""
    get_lines_sensitivity(folder_loc, sens_param, line_params, get_best_func)

Get sensitivity curves over sens_parameters using get_best_Func and line_params. 
# Arguments
-`get_best_func`: A function which takes an itemcollection and returns the best setting. Usually this will be a closure.
- `line_params`: The parameters defining each new line.
- `sens_param`: The x-axis parametere
"""
function get_lines_sensitivity(item_col, sens_param, line_params, get_best_func)

    diff_dict = diff(item_col)
    res_dict = Dict()

    @showprogress 0.1 "Line: " for line_prm ∈ Iterators.product((diff_dict[k] for k ∈ line_params)...)
        sd = Dict(line_params[i]=>line_prm[i] for i ∈ 1:length(line_params))
        _, _, _sub_itms = search(item_col, Dict(line_params[i]=>line_prm[i] for i ∈ 1:length(line_params)))
        sub_ic = ItemCollection(_sub_itms)
        sub_diff_dict = diff(sub_ic)
        μ = zeros(length(sub_diff_dict[sens_param]))
        σ = zeros(length(sub_diff_dict[sens_param]))
        for (sprm_idx, sprm) ∈ enumerate(sub_diff_dict[sens_param])
            _, _, sprm_items = search(sub_ic, Dict(sens_param=>sprm))
            _sprm_ic = ItemCollection(sprm_items)
            prm, (μ[sprm_idx], σ[sprm_idx]) = get_best_func(_sprm_ic)
        end
        res_dict[line_prm] = (μ, σ)
    end
    res_dict, diff_dict[sens_param]
end

get_lines_sensitivity(folder_str::AbstractString, sens_param, line_params, get_best_func) =
    get_lines_sensitivity(ItemCollection(folder_str), sens_param, line_params, get_best_func)

function plot_sensitivity(folder_loc_or_ic, sens_param, line_params, get_best_func, settings_dict=nothing; kwargs...)
    
    res_dict, x_axis = get_lines_sensitivity(folder_loc_or_ic, sens_param, line_params, get_best_func)
    kys = collect(keys(res_dict))
    
    if settings_dict isa Nothing
        settings_dict = Dict(k=>[:label=>"$k"] for k ∈ kys)
    end
    
    plt = if settings_dict isa Nothing
        plot(x_axis, res_dict[kys[1]][1], ribbon=res_dict[kys[1]][2]; kwargs...)
    else
        plot(x_axis, res_dict[kys[1]][1], ribbon=res_dict[kys[1]][2]; settings_dict[kys[1]]..., kwargs...)
    end
    for k ∈ kys[2:end]
        if settings_dict isa Nothing
            plot!(plt, x_axis, res_dict[k][1], ribbon=res_dict[k][2])
        else
            plot!(plt, x_axis, res_dict[k][1], ribbon=res_dict[k][2]; settings_dict[k]...)
        end
    end
    plt
end


function get_grid_sensitivity(item_col, sens_param_x, sens_param_y, grid_params, get_best_func)

    diff_dict = diff(item_col)
    res_dict = Dict()
    
    @showprogress 0.1 "Grid: " for line_prm ∈ Iterators.product((diff_dict[k] for k ∈ grid_params)...)
        sd = Dict(grid_params[i]=>line_prm[i] for i ∈ 1:length(grid_params))
        _, _, _sub_itms = search(item_col, sd)
        sub_ic = ItemCollection(_sub_itms)
        
        sub_diff_dict = diff(sub_ic)
        μ = zeros(length(sub_diff_dict[sens_param_y]), length(sub_diff_dict[sens_param_x]))
        σ = zeros(length(sub_diff_dict[sens_param_y]), length(sub_diff_dict[sens_param_x]))
        for (sprmy_idx, sprmy) ∈ enumerate(sub_diff_dict[sens_param_y])
            for (sprmx_idx, sprmx) ∈ enumerate(sub_diff_dict[sens_param_x])
                _, _, sprm_items = search(sub_ic, Dict(sens_param_x=>sprmx, sens_param_y=>sprmy))
                _sprm_ic = ItemCollection(sprm_items)
                prm, (μ[sprmy_idx, sprmx_idx], σ[sprmy_idx, sprmx_idx]) = get_best_func(_sprm_ic)
            end
        end
        res_dict[line_prm] = (μ, σ)
    end
    res_dict, (diff_dict[sens_param_x], diff_dict[sens_param_y])
end

function plot_heatmap(folder_loc, sens_param_x, sens_param_y, line_params, get_best_func, settings_dict=nothing; kwargs...)
    
end
