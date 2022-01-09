
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
    h′ = (m.w[1]*h′a + m.w[2]*h′m) ./ sum(m.w)

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


struct CaddAAGRUCell{A,V,S}  <: AbstractActionRNN
    w::V
    Wi1::A
    Wa1::A
    Wh1::A
    b1::V
    Wi2::A
    Wa2::A
    Wh2::A
    b2::V
    state0::S
end

CaddAAGRUCell(in, 
            na, 
            out; 
            init = Flux.glorot_uniform, 
            initb = Flux.zeros, 
            init_state = Flux.zeros) =
    CaddAAGRUCell(Flux.ones(2),
              init(out * 3, in),
              init(out * 3, na),
              init(out * 3, out),
              initb(out * 3),
              init(out * 3, in),
              init(out * 3, na),
              init(out * 3, out),
              initb(out * 3),
              init_state(out,1))

function (m::CaddAAGRUCell)(h, x::Tuple{A, O}) where {A, O}
    o = size(h, 1)

    a = x[1]
    obs = x[2]

    # additive 1
    gx1, gh1 = m.Wi1*obs, m.Wh1*h
    b1 = m.b1
    g1 = get_waa(m.Wa1, a)
    
    r1 = σ.(gate(gx1, o, 1) .+ gate(g1, o, 1) .+ gate(gh1, o, 1) .+ gate(b1, o, 1))
    z1 = σ.(gate(gx1, o, 2) .+ gate(g1, o, 2) .+ gate(gh1, o, 2) .+ gate(b1, o, 2))
    h̃1 = tanh.(gate(gx1, o, 3) .+ gate(g1, o, 3) .+ r1 .* gate(gh1, o, 3) .+ gate(b1, o, 3))
    h′1 = (1 .- z1) .* h̃1 .+ z1 .* h

    # additive 2
    gx2, gh2 = m.Wi2*obs, m.Wh2*h
    b2 = m.b2
    g2 = get_waa(m.Wa2, a)
    
    r2 = σ.(gate(gx2, o, 1) .+ gate(g2, o, 1) .+ gate(gh2, o, 1) .+ gate(b2, o, 1))
    z2 = σ.(gate(gx2, o, 2) .+ gate(g2, o, 2) .+ gate(gh2, o, 2) .+ gate(b2, o, 2))
    h̃2 = tanh.(gate(gx2, o, 3) .+ gate(g2, o, 3) .+ r2 .* gate(gh2, o, 3) .+ gate(b2, o, 3))
    h′2 = (1 .- z2) .* h̃2 .+ z2 .* h

    # adding together state
    h′ = (m.w[1]*h′1 + m.w[2]*h′2) ./ sum(m.w)

    sz = size(obs)
    return h′, reshape(h′, :, sz[2:end]...)

end

Flux.@functor CaddAAGRUCell

Base.show(io::IO, l::CaddAAGRUCell) =
  print(io, "CaddAAGRUCell(", size(l.Wi1, 2), ", ", size(l.Wa1), ", ", size(l.Wi1, 1)÷3, ")")

"""
    CaddAAGRU(in::Integer, out::Integer)
[Additive Action Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""

CaddAAGRU(a...; ka...) = Flux.Recur(CaddAAGRUCell(a...; ka...))
Flux.Recur(m::CaddAAGRUCell) = Flux.Recur(m, m.state0)


struct CaddMAGRUCell{A,V,T,S}  <: AbstractActionRNN
    w::V
    Wi1::T
    Wh1::T
    b1::A
    Wi2::T
    Wh2::T
    b2::A
    state0::S
end

CaddMAGRUCell(in, 
            na, 
            out; 
            init = Flux.glorot_uniform, 
            initb = Flux.zeros, 
            init_state = Flux.zeros) =
    CaddMAGRUCell(Flux.ones(2),
              init(na, out * 3, in; ignore_dims=1),
              init(na, out * 3, out; ignore_dims=1),
              initb(out * 3, na),
              init(na, out * 3, in; ignore_dims=1),
              init(na, out * 3, out; ignore_dims=1),
              initb(out * 3, na),
              init_state(out,1))

function (m::CaddMAGRUCell)(h, x::Tuple{A, O}) where {A, O}
    o = size(h, 1)

    a = x[1]
    obs = x[2]

    # multiplicative 1
    gx1, gh1 = contract_WA(m.Wi1, a, obs), contract_WA(m.Wh1, a, h)
    b1 = get_waa(m.b1, a)
    
    r1 = σ.(gate(gx1, o, 1)  .+ gate(gh1, o, 1) .+ gate(b1, o, 1))
    z1 = σ.(gate(gx1, o, 2)  .+ gate(gh1, o, 2) .+ gate(b1, o, 2))
    h̃1 = tanh.(gate(gx1, o, 3) .+ r1 .* gate(gh1, o, 3) .+ gate(b1, o, 3))
    h′1 = (1 .- z1) .* h̃1 .+ z1 .* h

    # multiplicative 2
    gx2, gh2 = contract_WA(m.Wi2, a, obs), contract_WA(m.Wh2, a, h)
    b2 = get_waa(m.b2, a)
    
    r2 = σ.(gate(gx2, o, 1)  .+ gate(gh2, o, 1) .+ gate(b2, o, 1))
    z2 = σ.(gate(gx2, o, 2)  .+ gate(gh2, o, 2) .+ gate(b2, o, 2))
    h̃2 = tanh.(gate(gx2, o, 3) .+ r1 .* gate(gh2, o, 3) .+ gate(b2, o, 3))
    h′2 = (1 .- z2) .* h̃2 .+ z2 .* h

    # adding together state
    h′ = (m.w[1]*h′1 + m.w[2]*h′2) ./ sum(m.w)

    sz = size(obs)
    return h′, reshape(h′, :, sz[2:end]...)

end

Flux.@functor CaddMAGRUCell

Base.show(io::IO, l::CaddMAGRUCell) =
  print(io, "CaddMAGRUCell(", size(l.Wia, 2), ", ", size(l.Wa), ", ", size(l.Wia, 1)÷3, ")")

"""
    CaddMAGRU(in::Integer, out::Integer)
[Additive Action Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""

CaddMAGRU(a...; ka...) = Flux.Recur(CaddMAGRUCell(a...; ka...))
Flux.Recur(m::CaddMAGRUCell) = Flux.Recur(m, m.state0)


struct CcatGRUCell{A,V,T,S}  <: AbstractActionRNN
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
    CcatGRUCell(init(out * 3, in),
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
    h′ = vcat(h′a, h′m)

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