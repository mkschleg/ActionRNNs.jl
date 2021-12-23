
# Sepcifying a action-conditional RNN Cell
using Flux
using Tullio
import TensorToolbox: cp_als


struct CaddRNNCell{F,A,V,T,S} <: AbstractActionRNN
    σ::F
    Wi::A
    Wa::A
    Wha::A
    ba::V
    Wx::T
    Whm::T
    bm::A
    state0::S
end

CaddRNNCell(in::Integer,
          na::Integer,
          out::Integer,
          σ=tanh;
          init=Flux.glorot_uniform,
          initb=Flux.zeros,
          init_state=Flux.zeros) = 
              CaddRNNCell(σ,
                        init(out, in),
                        init(out, na),
                        init(out, out),
                        initb(out),
                        init(na, out, in; ignore_dims=1),
                        init(na, out, out; ignore_dims=1),
                        initb(out, na),
                        init_state(out, 1))

function (m::CaddRNNCell)(h, x::Tuple{A, X}) where {A, X}
    σ, Wi, Wa, Wha, ba = m.σ, m.Wi, m.Wa, m.Wha, m.ba
    Wx, Whm, bm = m.Wx, m.Whm, m.bm

    o = x[2]
    a = x[1]

    # additive
    new_ha = σ.(Wi*o .+ get_waa(Wa, a) .+ Wha*h .+ ba)

    # multiplicative
    wx = contract_WA(m.Wx, a, o)
    wh = contract_WA(m.Whm, a, h)
    ba = get_waa(m.bm, a)

    new_hm = σ.(wx .+ wh .+ ba)
    
    if new_hm isa AbstractVector
        new_hm = reshape(new_hm, :, 1)
    end

    # adding together state
    new_h = new_ha + new_hm

    sz = size(o)
    return new_h, reshape(new_h, :, sz[2:end]...)
end


Flux.@functor CaddRNNCell

function Base.show(io::IO, l::AARNNCell)
  print(io, "CaddRNNCell(", size(l.Wi, 2), ", ", size(l.Wa), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    CaddRNN(in::Integer, out::Integer, σ = tanh)
The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.
"""

CaddRNN(a...; ka...) = Flux.Recur(CaddRNNCell(a...; ka...))
Flux.Recur(m::CaddRNNCell) = Flux.Recur(m, m.state0)
