

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

function l1_grad(s, us::UpdateState)
    ns = zero(s)
    for k ∈ us.grads.params
        if !(us.grads[k] isa Nothing)
            ns += sum(abs.(us.grads[k]))
        end
    end
    ns
end

function l2_grad(s, us::UpdateState)
    ns = zero(s)
    for (k, v) ∈ us.grads
        ns += dot(v, v)
    end
    ns
end

