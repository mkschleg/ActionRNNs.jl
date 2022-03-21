

import MacroTools: @capture
import Logging: Logging, @logmsg, LogLevel

const DataLevel = LogLevel(-93)
const SPECIAL_NAMES = [:idx]
const DataGroupsAndNames = Dict{Symbol, Dict{Symbol, Vector{LineNumberNode}}}()
global _datagroupnames_vec = []


# __init__() = begin
#     # @show "init", keys(DataGroupsAndNames)
#     # @show _datagroupnames_vec
# end

function proc_exs(group_sym, source, exs)
    @nospecialize
    if group_sym ∉ keys(DataGroupsAndNames)
        DataGroupsAndNames[group_sym] = Dict{String, LineNumberNode}()
    end
    NamesDict = get!(Dict{String, Vector{LineNumberNode}}, DataGroupsAndNames, group_sym)
    
    for ex in filter((ex)->!any([startswith(string(ex), string(k)) for k in SPECIAL_NAMES]), exs)
        if @capture(ex, name_=value_)
            if name ∉ keys(NamesDict)
                NamesDict[name] = [source]
            elseif source ∉ NamesDict[name]
                push!(NamesDict[name], source)
            end
            push!(_datagroupnames_vec, (group_sym, name, source))
        elseif @capture(ex, name_) && !contains(string(name), "=") 
            if name ∉ keys(NamesDict)
                NamesDict[name] = [source]
            elseif source ∉ NamesDict[name]
                push!(NamesDict[name], source)
            end
            push!(_datagroupnames_vec, (group_sym, name, source))
        else
            throw("Not a valid expressions for @data")
        end
    end

end

macro data(group, exs...)
    group_str = string(group)
    proc_exs(Symbol(group_str), __source__, exs)
    group_exp = :(_group = Symbol($group_str))
    :($Logging.@logmsg($DataLevel, "DATA", $(exs...), $group_exp)) |> esc
end
