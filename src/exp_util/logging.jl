using EllipsisNotation

# import ..ActionRNNs: DataLevel

import LoggingExtras: EarlyFilteredLogger, ActiveFilteredLogger, AbstractLogger, TeeLogger
import Logging: Logging, @logmsg, Info, handle_message, LogLevel, current_logger
import MacroTools: @capture

import ChoosyDataLoggers: ChoosyDataLoggers, @data
import ChoosyDataLoggers: construct_logger, NotDataFilter, DataLogger

ChoosyDataLoggers.construct_logger(; steps=nothing, extra_groups_and_names=[]) =
    ChoosyDataLoggers.construct_logger([[:EXP]; extra_groups_and_names]; steps=steps)


function prep_save_results(data, save_extras)
    save_results = copy(data[:EXP])
    for ex in save_extras
        if ex isa AbstractArray
            save_results[Symbol(ex[1]*"_"*ex[2])] = data[Symbol(ex[1])][Symbol(ex[2])]
        else
            for k in keys(data[Symbol(ex)])
                save_results[Symbol(ex * "_" * string(k))] = data[Symbol(ex)][Symbol(k)]
            end
        end
    end
    save_results
end

# function construct_logger(;steps=nothing, extra_groups_and_names=[])
#     res = Dict{Symbol, Dict{Symbol, AbstractArray}}()
#     logger = TeeLogger(
#         ExpUtils.NotDataFilter(current_logger()),
#         ExpUtils.DataLogger(:EXP, res, steps), # always capture exp
#         (ExpUtils.DataLogger(gn, res) for gn in extra_groups_and_names)...
#     )
#     res, logger
# end

# NotDataFilter(logger) = EarlyFilteredLogger(logger) do log_args
#     log_args.level != DataLevel
# end

# DataLogger(args...; kwargs...) = EarlyFilteredLogger(ArrayLogger(args...; kwargs...)) do log_args
#     log_args.level == DataLevel
# end

# DataLogger(group::Symbol, args...; kwargs...) = EarlyFilteredLogger(ArrayLogger(args...; kwargs...)) do log_args
#     log_args.level == DataLevel && log_args.group == group
# end

# DataLogger((group, name)::Tuple{Symbol, Symbol}, args...; kwargs...) = EarlyFilteredLogger(
#     ActiveFilteredLogger(ArrayLogger(args...; kwargs...)) do log
#         name ∈ keys(log.kwargs)
#     end) do log_args
#         log_args.level == DataLevel && log_args.group == group
# end

# DataLogger((group, name, proc)::Tuple{Symbol, Symbol, Symbol}, args...; kwargs...) = EarlyFilteredLogger(
#     ActiveFilteredLogger(ArrayLogger(args...; proc=proc, kwargs...)) do log
#         name ∈ keys(log.kwargs)
#     end) do log_args
#         log_args.level == DataLevel && log_args.group == group
# end

# DataLogger(group::String, args...; kwargs...) = DataLogger(Symbol(group), args...; kwargs...)

# function DataLogger(gnp::Vector{<:AbstractString}, args...; kwargs...)
#     if length(gnp) == 1
#         DataLogger(Symbol(gnp[1]), args...; kwargs...)
#     elseif length(gnp) == 2
#         DataLogger((Symbol(gnp[1]), Symbol(gnp[2])), args...; kwargs...)
#     elseif length(gnp) == 3
#         DataLogger((Symbol(gnp[1]), Symbol(gnp[2]), Symbol(gnp[3])), args...; kwargs...)
#     else
#         @error "Logging extras can only have up-to 3 arguments"
#     end
        
# end



# struct ArrayLogger{V<:Union{Val, Nothing}} <: AbstractLogger
#     data::Dict{Symbol, Dict{Symbol, AbstractArray}}
#     n::Union{Int, Nothing}
#     proc::V
# end

# ArrayLogger(data, n=nothing; proc=nothing) = ArrayLogger(data, n, isnothing(proc) ? nothing : Val(proc))

# function Logging.handle_message(logger::ArrayLogger, level, message, _module, group, id, file, line; kwargs...)
#     group_strg = get!(logger.data, group, Dict{Symbol, AbstractArray}())
#     for (k, v) in filter((kv)->kv.first!=:idx, kwargs)
#         data_strg = get!(group_strg, k) do
#             if :idx ∈ keys(kwargs)
#                 create_new_strg(logger.proc, v, logger.n)
#             else
#                 create_new_strg(logger.proc, v, nothing)
#             end
#         end
        
#         if isnothing(logger.n) || :idx ∉ keys(kwargs)
#             insert_data_strg!(logger.proc, data_strg, v, nothing)
#         else
#             insert_data_strg!(logger.proc, data_strg, v, kwargs[:idx])
#         end
#     end
    
# end

# Logging.min_enabled_level(::ArrayLogger) = LogLevel(-93)
# Logging.shouldlog(::ArrayLogger, ::Base.CoreLogging.LogLevel, args...) = true
# Logging.catch_exceptions(::ArrayLogger) = false

# create_new_strg(data::T, n::Nothing) where T = T[]
# # create_new_strg(data::AbstractVector{<:Number}, n::Int) = zeros(eltype(data), length(data), n) # create matrix to store all the data
# create_new_strg(data::AbstractArray{<:Number}, n::Int) = zeros(eltype(data), size(data)..., n) # create matrix to store all the data
# create_new_strg(data::AbstractArray, n::Int) = begin
#     Array{eltype{data}}(undef, size(data)..., n)
# end

# create_new_strg(data::Number, n::Int) = zeros(typeof(data), n)
# create_new_strg(data, n::Int) = Vector{eltype{data}}(undef, n)

# insert_data_strg!(strg::AbstractVector, data, ::Nothing) = push!(strg, data)

# insert_data_strg!(strg::AbstractVector, data, idx::Int) = if length(strg) < idx
#     push!(strg, data)
# else
#     strg[idx] = data
# end

# insert_data_strg!(strg::AbstractArray, data, idx::Int) = strg[.., idx] .= data

# # processing data while logging

# create_new_strg(t::Union{Val, Nothing}, data, n) = create_new_strg(process_data(t, data), n)
# insert_data_strg!(t::Union{Val, Nothing}, strg, data, idx) = insert_data_strg!(strg, process_data(t, data), idx)

# process_data(t::Nothing, data) = data

