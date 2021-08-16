
using Flux

mutable struct SingleLayer{F, FP, A, V}
    σ::F
    σ′::FP
    W::A
    b::V
end

SingleLayer(in::Integer, out::Integer, σ, σ′; init=(dims...)->zeros(Float32, dims...)) =
    SingleLayer(σ, σ′, init(out, in), init(out))

(layer::SingleLayer)(x) = layer.σ.(layer.W*x .+ layer.b)
deriv(layer::SingleLayer, x) = layer.σ′.(layer.W*x .+ layer.b)

Linear(in::Integer, out::Integer; kwargs...) =
    SingleLayer(in, out, identity, (x)->1.0; kwargs...)

# sigmoid = Flux.sigmoid
sigmoid′(x) = sigmoid(x)*(1.0-sigmoid(x))
# sigmoid′(x) = begin; tmp = sigmoid(x); tmp*(1.0-tmp); end;


struct ParallelStreams{T<:Tuple}
    l::T
end

ParallelStreams(args...) = ParallelStreams((args))

Flux.@functor ParallelStreams
(l::ParallelStreams)(x) = map((mdl)->mdl(x), l.l)

function Base.show(io::IO, l::ParallelStreams)
  print(io, "ParallelStreams(", (string(layer)*", " for layer in l.l)..., ")")
end


struct DualStreams{M1, M2}
    m1::M1
    m2::M2
end

Flux.@functor DualStreams
(l::DualStreams)(x) = (l.m1(x), l.m2(x))

struct ConcatStreams{M1, M2}
    m1::M1
    m2::M2
end

Flux.@functor ConcatStreams
(l::ConcatStreams)(x) = vcat(l.m1(x), l.m2(x))

function Base.show(io::IO, l::ConcatStreams)
  print(io, "ConcatStreams(", string(l.m1), ", ", string(l.m2), ")")
end


struct ActionStateStreams{AM, SM}
    action_model::AM
    state_model::SM
    num_actions::Int
end

Flux.@functor ActionStateStreams
(l::ActionStateStreams)(x::Tuple{Vector, Vector}) = (l.action_model(x[1]), l.state_model(x[2]))
(l::ActionStateStreams)(x::Tuple{Int, Vector}) =
    (l.action_model(make_action_matrix(x[1], l.num_actions)), l.state_model(x[2]))
(l::ActionStateStreams)(x::Tuple{Vector, Matrix}) =
    (l.action_model(make_action_matrix(x[1], l.num_actions)), l.state_model(x[2]))

function make_action_matrix(actions, na)
    action_matrix = zeros(Float32, na, size(actions, 1))
    for i in 1:size(actions, 1)
        action_matrix[actions[i], i] = 1
    end
    action_matrix
end

function Base.show(io::IO, l::ActionStateStreams)
  print(io, "ActionStateStream(", string(l.action_model), ", ", string(l.state_model), ")")
end




