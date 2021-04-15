struct AALSTMCell{A,V,S} <: AbstractActionRNN
  Wi::A
  Wh::A
  b::V
  state0::S
end

function AALSTMCell(in::Integer, out::Integer;
                  init = glorot_uniform,
                  initb = zeros,
                  init_state = zeros)
    cell = AALSTMCell(init(out * 4, in),
                      init(out * 4, out),
                      initb(out * 4),
                      (init_state(out,1), init_state(out,1)))
  cell.b[gate(out, 2)] .= 1
  return cell
end

function (m::AALSTMCell{A,V,<:NTuple{2,AbstractMatrix{T}}})((h, c), x::Union{AbstractVecOrMat{T},OneHotArray}) where {A,V,T}
  b, o = m.b, size(h, 1)
  g = m.Wi*x .+ m.Wh*h .+ b
  input = σ.(gate(g, o, 1))
  forget = σ.(gate(g, o, 2))
  cell = tanh.(gate(g, o, 3))
  output = σ.(gate(g, o, 4))
  c = forget .* c .+ input .* cell
  h′ = output .* tanh.(c)
  sz = size(x)
  return (h′, c), reshape(h′, :, sz[2:end]...)
end

@functor AALSTMCell

Base.show(io::IO, l::AALSTMCell) =
  print(io, "LSTMCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷4, ")")

"""
    LSTM(in::Integer, out::Integer)
[Long Short Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)
recurrent layer. Behaves like an RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
AALSTM(a...; ka...) = Flux.Recur(AALSTMCell(a...; ka...))
Flux.Recur(m::AALSTMCell) = Flux.Recur(m, m.state0)
