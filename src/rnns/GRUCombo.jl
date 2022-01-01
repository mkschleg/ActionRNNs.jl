
# Sepcifying a action-conditional RNN Cell
using Flux
using Tullio
import TensorToolbox: cp_als
using Flux: gate


struct CaddGRUCell{A,V,T,S}  <: AbstractActionRNN
    w::V
    Wia::A
    Wa::A
    Wha::A
    ba::V
    Wim::T
    Whm::T
    bm::A
    state0::S
end

CaddGRUCell(in, 
            na, 
            out; 
            init = Flux.glorot_uniform, 
            initb = Flux.zeros, 
            init_state = Flux.zeros) =
    CaddGRUCell(Flux.ones(2),
              init(out * 3, in),
              init(out * 3, na),
              init(out * 3, out),
              initb(out * 3),
              init(na, out * 3, in; ignore_dims=1),
              init(na, out * 3, out; ignore_dims=1),
              initb(out * 3, na),
              init_state(out,1))

function (m::CaddGRUCell)(h, x::Tuple{A, O}) where {A, O}
    o = size(h, 1)

    a = x[1]
    obs = x[2]

    # additive
    gxa, gha = m.Wia*obs, m.Wha*h
    ba = m.ba
    ga = get_waa(m.Wa, a)
    
    ra = σ.(gate(gxa, o, 1) .+ gate(ga, o, 1) .+ gate(gha, o, 1) .+ gate(ba, o, 1))
    za = σ.(gate(gxa, o, 2) .+ gate(ga, o, 2) .+ gate(gha, o, 2) .+ gate(ba, o, 2))
    h̃a = tanh.(gate(gxa, o, 3) .+ gate(ga, o, 3) .+ ra .* gate(gha, o, 3) .+ gate(ba, o, 3))
    h′a = (1 .- za) .* h̃a .+ za .* h

    # multiplicative 
    gxm, ghm = contract_WA(m.Wim, a, obs), contract_WA(m.Whm, a, h)
    bm = get_waa(m.bm, a)
    
    rm = σ.(gate(gxm, o, 1)  .+ gate(ghm, o, 1) .+ gate(bm, o, 1))
    zm = σ.(gate(gxm, o, 2)  .+ gate(ghm, o, 2) .+ gate(bm, o, 2))
    h̃m = tanh.(gate(gxm, o, 3) .+ rm .* gate(ghm, o, 3) .+ gate(bm, o, 3))
    h′m = (1 .- zm) .* h̃m .+ zm .* h

    # adding together state
    h′ = m.w[1]*h′a + m.w[2]*h′m

    sz = size(obs)
    return h′, reshape(h′, :, sz[2:end]...)

end

Flux.@functor CaddGRUCell

Base.show(io::IO, l::CaddGRUCell) =
  print(io, "CaddGRUCell(", size(l.Wia, 2), ", ", size(l.Wa), ", ", size(l.Wia, 1)÷3, ")")

"""
    CaddGRU(in::Integer, out::Integer)
[Additive Action Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""

CaddGRU(a...; ka...) = Flux.Recur(CaddGRUCell(a...; ka...))
Flux.Recur(m::CaddGRUCell) = Flux.Recur(m, m.state0)


struct CcatGRUCell{A,V,T,S}  <: AbstractActionRNN
    w::V
    Wia::A
    Wa::A
    Wha::A
    ba::V
    Wim::T
    Whm::T
    bm::A
    state0::S
end

CcatGRUCell(in, 
            na, 
            out; 
            init = Flux.glorot_uniform, 
            initb = Flux.zeros,
            init_state = Flux.zeros) =
    CcatGRUCell(Flux.ones(2),
              init(out * 3, in),
              init(out * 3, na),
              init(out * 3, out * 2),
              initb(out * 3),
              init(na, out * 3, in; ignore_dims=1),
              init(na, out * 3, out * 2; ignore_dims=1),
              initb(out * 3, na),
              init_state(out * 2,1))

function (m::CcatGRUCell)(h, x::Tuple{A, O}) where {A, O}
    o = size(h, 1) ÷ 2

    a = x[1]
    obs = x[2]

    # additive
    gxa, gha = m.Wia*obs, m.Wha*h
    ba = m.ba
    ga = get_waa(m.Wa, a)
    
    ra = σ.(gate(gxa, o, 1) .+ gate(ga, o, 1) .+ gate(gha, o, 1) .+ gate(ba, o, 1))
    za = σ.(gate(gxa, o, 2) .+ gate(ga, o, 2) .+ gate(gha, o, 2) .+ gate(ba, o, 2))
    h̃a = tanh.(gate(gxa, o, 3) .+ gate(ga, o, 3) .+ ra .* gate(gha, o, 3) .+ gate(ba, o, 3))
    h′a = (1 .- za) .* h̃a .+ za .* h[1:o, :]

    # multiplicative 
    gxm, ghm = contract_WA(m.Wim, a, obs), contract_WA(m.Whm, a, h)
    bm = get_waa(m.bm, a)
    
    rm = σ.(gate(gxm, o, 1)  .+ gate(ghm, o, 1) .+ gate(bm, o, 1))
    zm = σ.(gate(gxm, o, 2)  .+ gate(ghm, o, 2) .+ gate(bm, o, 2))
    h̃m = tanh.(gate(gxm, o, 3) .+ rm .* gate(ghm, o, 3) .+ gate(bm, o, 3))
    h′m = (1 .- zm) .* h̃m .+ zm .* h[o+1:end, :]

    # concatenating together state
    h′ = vcat(m.w[1]*h′a, m.w[2]*h′m)

    sz = size(obs)
    return h′, reshape(h′, :, sz[2:end]...)

end

Flux.@functor CcatGRUCell

Base.show(io::IO, l::CcatGRUCell) =
  print(io, "CcatGRUCell(", size(l.Wia, 2), ", ", size(l.Wa), ", ", size(l.Wia, 1)÷3, ")")

"""
    CcatGRU(in::Integer, out::Integer)
[Additive Action Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""

CcatGRU(a...; ka...) = Flux.Recur(CcatGRUCell(a...; ka...))
Flux.Recur(m::CcatGRUCell) = Flux.Recur(m, m.state0)