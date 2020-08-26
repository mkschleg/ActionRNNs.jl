
using Flux

function (m::Flux.LSTMCell)((h, c), x::T) where {T<:Tuple}
    m((h,c), x[2])
end

mutable struct ActionLSTMCell{A,V}
  Wi::A
  Wh::A
  b::V
  h::V
  c::V
end

function ActionLSTMCell(in::Integer, out::Integer;
                  init = glorot_uniform)
    cell = LSTMCell(init(out * 4, in), init(out * 4, out), init(out * 4),
                    zeros(out), zeros(out))
    cell.b[gate(out, 2)] .= 1
    return cell
end

function (m::ActionLSTMCell)((h, c), (a, x))
    b, o = m.b, size(h, 1)
    g = m.Wi*x .+ m.Wh*h .+ b
    input = σ.(gate(g, o, 1))
    forget = σ.(gate(g, o, 2))
    cell = tanh.(gate(g, o, 3))
    output = σ.(gate(g, o, 4))
    c = forget .* c .+ input .* cell
    h′ = output .* tanh.(c)
    return (h′, c), h′
end

hidden(m::ActionLSTMCell) = (m.h, m.c)

@treelike ActionLSTMCell

Base.show(io::IO, l::ActionLSTMCell) =
    print(io, "ActionLSTMCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷4, ")")


"""
    ALSTM(in::Integer, out::Integer)
Long Short Term Memory recurrent layer. Behaves like an RNN but generally
exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
ALSTM(a...; ka...) = Recur(ActionLSTMCell(a...; ka...))

