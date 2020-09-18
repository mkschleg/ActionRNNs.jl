
using Flux
using Flux: gate

function (m::Flux.LSTMCell)((h, c), x::T) where {T<:Tuple}
    m((h,c), x[2])
end

mutable struct ActionLSTMCell{A1,A2, V1, V2, H}
    Wi_input::A1
    Wh_input::A1
    b_input::V1
    Wi::A2
    Wh::A2
    b::V2
    h::H
    c::H
    num_actions::Int
end

function ActionLSTMCell(in::Integer, num_actions::Integer, out::Integer;
                        init = Flux.glorot_uniform)
    # Only for the input gate
    cell = ActionLSTMCell(
        [init(out, in) for l ∈ 1:num_actions],
        [init(out, out) for l in 1:num_actions],
        [init(out) for l ∈ 1:num_actions],
        init(out*3, in),
        init(out*3, out),
        init(out*3),
        Flux.zeros(out), Flux.zeros(out), num_actions)
    cell.b[:, 1] .= 1
    return cell
end


function _contract(W::AbstractArray{N, 3}, x1::AbstractArray{N, 1}) where {N<:Number}
    [(@view W[:,:,l])*x1 for l in 1:size(W)[3]]
end

_contract(W::AbstractArray{N, 3}, x1::AbstractArray{N, 1}, l::Integer) where {N<:Number} = 
    (@view W[:,:,l])*x1


function get_views(m, o, a)
    idx = [1:(3*o); ((a+2)*o+1):(a+3)*o]
    ((@view m.Wi[idx, :]), (@view m.Wh[idx, :]), (@view m.b[idx]))
end

function (m::ActionLSTMCell)((h, c), ax::Tuple{I, A}) where {I<:Integer, A<:AbstractArray}
    o = size(h, 1)
    x = ax[2]
    a = ax[1]

    Wi, Wh, b = m.Wi, m.Wh, m.b
    Wi_input, Wh_input, b_input = m.Wi_input[a], m.Wh_input[a], m.b_input[a]

    g = Wi*x .+ Wh*h .+ b
    input = σ.(Wi_input*x .+ Wh_input*h .+ b_input)
    forget = gate(g, o, 1)
    cell = gate(g, o, 2)
    output = gate(g, o, 3)

    c = forget .* c .+ input .* cell
    h′ = output .* tanh.(c)
    return (h′, c), h′
end

Flux.hidden(m::ActionLSTMCell) = (m.h, m.c)

Flux.@functor ActionLSTMCell

Base.show(io::IO, l::ActionLSTMCell) =
    print(io, "ActionLSTMCell(", size(l.Wi, 2), ", ", size(l.Wi, 1), ")")


"""
    ALSTM(in::Integer, out::Integer)
Long Short Term Memory recurrent layer. Behaves like an RNN but generally
exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
ALSTM(a...; ka...) = Flux.Recur(ActionLSTMCell(a...; ka...))

