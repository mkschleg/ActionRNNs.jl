using Flux
using Flux: gate

struct AALSTMCell{A,V,S} <: AbstractActionRNN
    Wi::A
    Wa::A
    Wh::A
    b::V
    state0::S
end

function AALSTMCell(in::Integer, na::Integer, out::Integer;
                  init = Flux.glorot_uniform,
                  initb = Flux.zeros,
                  init_state = Flux.zeros)
    cell = AALSTMCell(init(out * 4, in),
                      init(out * 4, na),
                      init(out * 4, out),
                      initb(out * 4),
                      (init_state(out,1), init_state(out,1)))
  cell.b[gate(out, 2)] .= 1
  return cell
end

function (m::AALSTMCell)((h, c), x::Tuple{A, O}) where {A, O}
    b, o = m.b, size(h, 1)

    a = x[1]
    obs = x[2]

    g = m.Wi*obs .+ m.Wh*h .+ get_waa(m.Wa, a) .+ b

    input = σ.(gate(g, o, 1))
    forget = σ.(gate(g, o, 2))
    cell = tanh.(gate(g, o, 3))
    output = σ.(gate(g, o, 4))
    c = forget .* c .+ input .* cell
    h′ = output .* tanh.(c)
    sz = size(obs)
    return (h′, c), reshape(h′, :, sz[2:end]...) # h′
end

Flux.@functor AALSTMCell

Base.show(io::IO, l::AALSTMCell) =
  print(io, "AALSTMCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷4, ")")

"""
    AALSTM(in::Integer, na::Integer, out::Integer)
[Additive Action Long Short Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
recurrent layer. Behaves like an RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
AALSTM(a...; ka...) = Flux.Recur(AALSTMCell(a...; ka...))
Flux.Recur(m::AALSTMCell) = Flux.Recur(m, m.state0)



struct MALSTMCell{A,V,S} <: AbstractActionRNN
    Wi::A
    Wh::A
    b::V
    state0::S
end

function MALSTMCell(in::Integer, na::Integer, out::Integer;
                  init = glorot_uniform,
                  initb = Flux.zeros,
                  init_state = Flux.zeros)
    cell = MALSTMCell(init(na, out * 4, in; ignore_dims=1),
                      init(na, out * 4, out; ignore_dims=1),
                      initb(out * 4, na),
                      (init_state(out,1), init_state(out,1)))
  cell.b[gate(out, 2)] .= 1
  return cell
end

function (m::MALSTMCell)((h, c), x::Tuple{A, O}) where {A, O}
    o = size(h, 1)

    a = x[1]
    obs = x[2]
    
    g = contract_WA(m.Wi, a, obs) .+ contract_WA(m.Wh, a, h) .+ get_waa(m.b, a)

    input = σ.(gate(g, o, 1))
    forget = σ.(gate(g, o, 2))
    cell = tanh.(gate(g, o, 3))
    output = σ.(gate(g, o, 4))
    c = forget .* c .+ input .* cell
    h′ = output .* tanh.(c)
    sz = size(obs)
    return (h′, c), reshape(h′, :, sz[2:end]...) # h′
end

Flux.@functor MALSTMCell

Base.show(io::IO, l::MALSTMCell) =
  print(io, "MALSTMCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷4, ")")

"""
    MALSTM(in::Integer, na::Integer, out::Integer)
[Muliplicative Action Long Short Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
recurrent layer. Behaves like an RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
MALSTM(a...; ka...) = Flux.Recur(MALSTMCell(a...; ka...))
Flux.Recur(m::MALSTMCell) = Flux.Recur(m, m.state0)
