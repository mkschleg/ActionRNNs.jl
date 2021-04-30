using Flux
using Flux: gate

struct AAGRUCell{A,V,S}  <: AbstractActionRNN
    Wi::A
    Wa::A
    Wh::A
    b::V
    state0::S
end

AAGRUCell(in, na, out; init = Flux.glorot_uniform, initb = Flux.zeros, init_state = Flux.zeros) =
    AAGRUCell(init(out * 3, in),
              init(out * 3, na),
              init(out * 3, out),
              initb(out * 3),
              init_state(out,1))

function (m::AAGRUCell)(h, x::Tuple{A, O}) where {A, O}
    b, o = m.b, size(h, 1)

    a = x[1]
    obs = x[2]
    
    gx, gh = m.Wi*obs, m.Wh*h
    ga = get_waa(m.Wa, a)
    
    r = σ.(gate(gx, o, 1) .+ gate(ga, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
    z = σ.(gate(gx, o, 2) .+ gate(ga, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
    h̃ = tanh.(gate(gx, o, 3) .+ gate(ga, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
    h′ = (1 .- z) .* h̃ .+ z .* h
    sz = size(obs)
  return h′, reshape(h′, :, sz[2:end]...)
end

Flux.@functor AAGRUCell

Base.show(io::IO, l::AAGRUCell) =
  print(io, "AAGRUCell(", size(l.Wi, 2), ", ", size(l.Wa), ", ", size(l.Wi, 1)÷3, ")")

"""
    AAGRU(in::Integer, out::Integer)
[Additive Action Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
AAGRU(a...; ka...) = Flux.Recur(AAGRUCell(a...; ka...))
Flux.Recur(m::AAGRUCell) = Flux.Recur(m, m.state0)


struct MAGRUCell{A,V,S}  <: AbstractActionRNN
    Wi::A
    Wh::A
    b::V
    state0::S
end

MAGRUCell(in, na, out; init = glorot_uniform, initb = Flux.zeros, init_state = Flux.zeros) =
    MAGRUCell(init(na, out * 3, in; ignore_dims=1),
              init(na, out * 3, out; ignore_dims=1),
              initb(out * 3, na),
              init_state(out,1))


function (m::MAGRUCell)(h, x::Tuple{A, O}) where {A, O}
    o = size(h, 1)

    a = x[1]
    obs = x[2]
    
    gx, gh = contract_WA(m.Wi, a, obs), contract_WA(m.Wh, a, h)
    b = get_waa(m.b, a)
    
    r = σ.(gate(gx, o, 1)  .+ gate(gh, o, 1) .+ gate(b, o, 1))
    z = σ.(gate(gx, o, 2)  .+ gate(gh, o, 2) .+ gate(b, o, 2))
    h̃ = tanh.(gate(gx, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
    h′ = (1 .- z) .* h̃ .+ z .* h
    sz = size(obs)
  return h′, reshape(h′, :, sz[2:end]...)
end

Flux.@functor MAGRUCell

Base.show(io::IO, l::MAGRUCell) =
  print(io, "MAGRUCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ")")

"""
    MAGRU(in::Integer, out::Integer)
[Multiplicative Action Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
MAGRU(a...; ka...) = Flux.Recur(MAGRUCell(a...; ka...))
Flux.Recur(m::MAGRUCell) = Flux.Recur(m, m.state0)


struct FacMAGRUCell{A,V,S}  <: AbstractActionRNN
    W::A
    Wi::A
    Wh::A
    Wa::A
    b::V
    state0::S
end

FacMAGRUCell(in, na, out, factors; init = glorot_uniform, initb = Flux.zeros, init_state = Flux.zeros) =
    FacMAGRUCell(init(out * 3, factors),
                 init(factors, in),
                 init(factors, out),
                 init(factors, na),
                 initb(out * 3, na),
                 init_state(out,1))


function (m::FacMAGRUCell)(h, x::Tuple{A, O}) where {A, O}
    o = size(h, 1)

    a = x[1]
    obs = x[2]

    g = m.W * ((m.Wi*obs .+ m.Wh*h) .* get_Wabya(m.Wa, a)) .+ get_waa(m.b, a)
    
    r = σ.(gate(g, o, 1))
    z = σ.(gate(g, o, 2))
    h̃ = tanh.(gate(g, o, 3))
    h′ = (1 .- z) .* h̃ .+ z .* h
    sz = size(obs)
  return h′, reshape(h′, :, sz[2:end]...)
end

Flux.@functor FacMAGRUCell

Base.show(io::IO, l::FacMAGRUCell) =
  print(io, "FacMAGRUCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ")")

"""
    FacMAGRU(in::Integer, out::Integer)
[Multiplicative Action Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
FacMAGRU(a...; ka...) = Flux.Recur(FacMAGRUCell(a...; ka...))
Flux.Recur(m::FacMAGRUCell) = Flux.Recur(m, m.state0)
