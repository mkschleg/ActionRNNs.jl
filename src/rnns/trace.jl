struct TraceRNN{R, S}
    rnn::R
    state0::S
    lambda::Float32
end

function (m::TraceRNNCell)(h, x)

    trace = h[1]
    hs = h[2]

    
    # o = x[2]
    # a = x[1]
    
    # new_h = σ.(Wi*o .+ get_waa(Wa, a) .+ Wh*h .+ b)
    # sz = size(o)
    # return new_h, new_h#reshape(h, :, sz[2:end]...)
end

Flux.@functor AARNNCell

function Base.show(io::IO, l::AARNNCell)
  print(io, "AARNNCell(", size(l.Wi, 2), ", ", size(l.Wa), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    AARNN(in::Integer, out::Integer, σ = tanh)
The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.
"""
AARNN(a...; ka...) = Flux.Recur(AARNNCell(a...; ka...))
Flux.Recur(m::AARNNCell) = Flux.Recur(m, m.state0)
