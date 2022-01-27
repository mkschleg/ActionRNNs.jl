

import ..UpdateState


struct SimpleLogger{T}
    data::T
    f_map::Dict{Symbol, Function}
    function SimpleLogger(names, dtypes, f_map)
        nt = NamedTuple{names}([Vector{dt}() for dt ∈ dtypes])
        new{typeof(nt)}(nt,
                        f_map)
    end
end

get_data(sl::SimpleLogger) = sl.data
Base.getindex(sl::SimpleLogger, idx::Symbol) = sl.data[idx]

function (sl::SimpleLogger)(args...; kwargs...)
    for k ∈ keys(sl.f_map)
        push!(getindex(sl.data, k), sl.f_map[k](args...; kwargs...))
    end
end



macro CreateSimpleLogger
    
end




mutable struct UpdateStateAnalysis{T}
    data::T
    f_map::Dict{Symbol, Function}
    function UpdateStateAnalysis(initial, f_map)
        nt = initial
        new{typeof(nt)}(
            nt,
            f_map
        )
    end
end

function (usa::UpdateStateAnalysis)(us::UpdateState)
    usa.data = typeof(usa.data)((usa.f_map[n](getindex(usa.data, n), us)) for n ∈ keys(usa.data))
end

Base.getindex(usa::UpdateStateAnalysis, idx::Symbol) = getindex(usa.data, idx)


function sum_grad(f::Function, us::UpdateState)

    
    sum(us.grads) do grad
        # @show typeof(p)
        if grad isa Nothing
            zero(us.loss)
        else
            sum(f, grad)
        end
        # p.second isa Nothing ? zero(eltype(p.first)) : sum(f, p.second)
    end
end

l1_grad(s, us::UpdateState) = sum_grad(abs, us)
l2_grad(s, us::UpdateState) = sum_grad(abs2, us)

function layer_sum_grad(f::Function, us::UpdateState)
    Dict(
        p.second isa Nothing ? p.first=>p.second : p.first=>sum(f, p.second) for p in us.grads
    )
end

