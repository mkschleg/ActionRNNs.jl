
using Flux
using Flux: gate


mutable struct ActionGRUCell{A1,A2, V1, V2, H} <: AbstractActionRNN
    Wi_input::A1
    Wh_input::A1
    b_input::V1
    Wi::A2
    Wh::A2
    b::V2
    h::H
end

function ActionGRUCell(in::Integer, num_actions::Integer, out::Integer;
                        init = Flux.glorot_uniform)
    ActionGRUCell(
        [init(in, out) for a in 1:num_actions],
        [init(out, out) for a in 1:num_actions],
        [Flux.zeros(out) for a in 1:num_actions],
        init(out * 2, in),
        init(out * 2, out),
        Flux.zeros(out * 2),
        Flux.zeros(out))
end


function (m::ActionGRUCell)((h, c), ax::Tuple{I, A}) where {I<:Integer, A<:AbstractArray}
    a, x = ax
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi*x, m.Wh*h
    gx_input, gh_input, b_input = m.Wi_input[a]*x, m.Wh_input[a]*h, m.b_input[a]
    r = σ.(gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
    z = σ.(gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
    h̃ = tanh.(gx_input .+ r .* gh_input .+ b_input)
    h′ = (1 .- z).*h̃ .+ z.*h
    return h′, h′
end

# Flux.hidden(m::ActionGRUCell) = m.h

Flux.@functor ActionGRUCell

Base.show(io::IO, l::ActionGRUCell) =
    print(io, "ActionGRUCell(", size(l.Wi_input[1], 2), ", ", size(l.Wi, 1), ")")

"""
    AGRU(in::Integer, num_actions::Integer, out::Integer)
 Gated Recurrent Unit layer. Behaves like an RNN but generally
exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
AGRU(a...; ka...) = Flux.Recur(ActionGRUCell(a...; ka...))
