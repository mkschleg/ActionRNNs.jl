using Flux
using Flux: gate

struct ActionGatedCell{A,V,S}  <: AbstractActionRNN
    Wi::A
    Wa::A
    Wh::A
    b::V
    W::A
    state0::S
end

ActionGatedCell(in, na, internal, out; init = Flux.glorot_uniform, initb = Flux.zeros, init_state = Flux.zeros) =
    ActionGatedCell(init(internal * 2, in),
                    init(internal * 2, na),
                    init(internal * 2, out),
                    initb(internal * 2),
                    init(out, internal),
                    init_state(out,1))

function (m::ActionGatedCell)(h, x::Tuple{A, O}) where {A, O}

    a, obs = x
    
    b, o = m.b, size(m.W, 2)
    gx, gh = m.Wi*obs, m.Wh*h
    ga = get_waa(m.Wa, a)
    
    c = gate(gx, o, 1)  .+ gate(gh, o, 1) .+ gate(ga, o, 1) .+ gate(b, o, 1)
    r = σ.(gate(gx, o, 2)  .+ gate(gh, o, 2) .+ gate(ga, o, 2) .+ gate(b, o, 2))
    h′ = tanh.(m.W*(r .* c))

    sz = size(obs)
    return h′, reshape(h′, :, sz[2:end]...)
end

Flux.@functor ActionGatedCell

Base.show(io::IO, l::ActionGatedCell) =
  print(io, "ActionGatedCell(", size(l.Wi, 2), ", ", size(l.Wa), ", ", size(l.Wi, 1)÷3, ")")


"""
    ActionGatedRNN(in::Integer, na, internal, out::Integer, σ = tanh)
The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.
"""
ActionGatedRNN(a...; ka...) = Flux.Recur(ActionGatedCell(a...; ka...))
Flux.Recur(m::ActionGatedCell) = Flux.Recur(m, m.state0)


struct GAIARNNCell{A,V,S}  <: AbstractActionRNN
    Wi::A
    Wa::A
    Wh::A
    bi::V
    ba::V
    W::A
    state0::S
end

GAIARNNCell(in, na, internal, out; init = Flux.glorot_uniform, initb = Flux.zeros, init_state = Flux.zeros) =
    GAIARNNCell(init(internal, in),
                init(internal, na),
                init(internal, out),
                initb(internal),
                initb(internal),
                init(out, internal),
                init_state(out,1))

function (m::GAIARNNCell)(h, x::Tuple{A, O}) where {A, O}
    a, obs = x

    gx, gh = m.Wi*obs, m.Wh*h
    ga = get_waa(m.Wa, a)

    c = gh .+ gx .+ m.bi
    r = σ.(ga .+ m.ba)
    h′ = tanh.(m.W*(r .* c))

    sz = size(obs)
    return h′, reshape(h′, :, sz[2:end]...)
end

Flux.@functor GAIARNNCell

Base.show(io::IO, l::GAIARNNCell) =
  print(io, "GAIARNNCell(", size(l.Wi, 2), ", ", size(l.Wa), ", ", size(l.Wi, 1)÷3, ")")


"""
    GAIARNN(in::Integer, na, internal, out::Integer, σ = tanh)
The most basic recurrent layer; essentially acts as a `Dense` layer, but with the
output fed back into the input each time step.
"""
GAIARNN(a...; ka...) = Flux.Recur(GAIARNNCell(a...; ka...))
Flux.Recur(m::GAIARNNCell) = Flux.Recur(m, m.state0)


struct GAIGRUCell{A,V,S}  <: AbstractActionRNN
    Wi_::A
    Wa_::A
    Wh_::A
    b_::V
    W::A
    Wi::A
    Wa::A
    Wh::A
    b::V
    state0::S
end

GAIGRUCell(in, na, internal, out; init = Flux.glorot_uniform, initb = Flux.zeros, init_state = Flux.zeros) =
    GAIGRUCell(init(internal * 2, in),
              init(internal * 2, na),
              init(internal * 2, out),
              initb(internal * 2),
              init(out, internal),
              init(out * 3, in),
              init(out * 3, na),
              init(out * 3, out),
              initb(out * 3),
              init_state(out,1))

function (m::GAIGRUCell)(h, x::Tuple{A, O}) where {A, O}
    b, b_, o, o_ = m.b, m.b_, size(h, 1), size(m.W, 2)

    a = x[1]
    obs = x[2]

    gx_, gh_ = m.Wi_*obs, m.Wh_*h
    ga_ = get_waa(m.Wa_, a)

    c = gate(gx_, o_, 1)  .+ gate(gh_, o_, 1) .+ gate(ga_, o_, 1) .+ gate(b_, o_, 1)
    r_ = σ.(gate(gx_, o_, 2)  .+ gate(gh_, o_, 2) .+ gate(ga_, o_, 2) .+ gate(b_, o_, 2))
    h_ = tanh.(m.W*(r_ .* c))

    gx, gh = m.Wi*obs, m.Wh*h_
    ga = get_waa(m.Wa, a)

    r = σ.(gate(gx, o, 1) .+ gate(ga, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
    z = σ.(gate(gx, o, 2) .+ gate(ga, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
    h̃ = tanh.(gate(gx, o, 3) .+ gate(ga, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
    h′ = (1 .- z) .* h̃ .+ z .* h_
    sz = size(obs)
  return h′, reshape(h′, :, sz[2:end]...)
end

Flux.@functor GAIGRUCell

Base.show(io::IO, l::GAIGRUCell) =
  print(io, "GAIGRUCell(", size(l.Wi, 2), ", ", size(l.Wa), ", ", size(l.Wi, 1)÷3, ")")

"""
    GAIGRU(in::Integer, na::Integer, internal::Integer, out::Integer)
[Gated Action Input Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
GAIGRU(a...; ka...) = Flux.Recur(GAIGRUCell(a...; ka...))
Flux.Recur(m::GAIGRUCell) = Flux.Recur(m, m.state0)


struct GAIAGRUCell{A,V,S}  <: AbstractActionRNN
    Wi_::A
    Wa_::A
    Wh_::A
    bi::V
    ba::V
    W::A
    Wi::A
    Wa::A
    Wh::A
    b::V
    state0::S
end

GAIAGRUCell(in, na, internal, out; init = Flux.glorot_uniform, initb = Flux.zeros, init_state =Flux.zeros) =
    GAIAGRUCell(init(internal, in),
                init(internal, na),
                init(internal, out),
                initb(internal),
                initb(internal),
                init(out, internal),
                init(out * 3, in),
                init(out * 3, na),
                init(out * 3, out),
                initb(out * 3),
                init_state(out,1))

function (m::GAIAGRUCell)(h, x::Tuple{A, O}) where {A, O}
    b, o, = m.b, size(h, 1)

    a = x[1]
    obs = x[2]

    gx_, gh_ = m.Wi_*obs, m.Wh_*h
    ga_ = get_waa(m.Wa_, a)

    c = gh_ .+ gx_ .+ m.bi
    r_ = σ.(ga_ .+ m.ba)
    h_ = tanh.(m.W*(r_ .* c))

    gx, gh = m.Wi*obs, m.Wh*h_
    ga = get_waa(m.Wa, a)

    r = σ.(gate(gx, o, 1) .+ gate(ga, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
    z = σ.(gate(gx, o, 2) .+ gate(ga, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
    h̃ = tanh.(gate(gx, o, 3) .+ gate(ga, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
    h′ = (1 .- z) .* h̃ .+ z .* h_
    sz = size(obs)
  return h′, reshape(h′, :, sz[2:end]...)
end

Flux.@functor GAIAGRUCell

Base.show(io::IO, l::GAIAGRUCell) =
  print(io, "GAIAGRUCell(", size(l.Wi, 2), ", ", size(l.Wa), ", ", size(l.Wi, 1)÷3, ")")

"""
    GAIAGRU(in::Integer, na::Integer, internal::Integer, out::Integer)
[Gated Action Input Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
GAIAGRU(a...; ka...) = Flux.Recur(GAIAGRUCell(a...; ka...))
Flux.Recur(m::GAIAGRUCell) = Flux.Recur(m, m.state0)


struct GAUGRUCell{A,V,S}  <: AbstractActionRNN
    Wi::A
    Wa::A
    Wh::A
    b::V
    state0::S
end

GAUGRUCell(in, na, internal, out; init = Flux.glorot_uniform, initb = Flux.zeros, init_state = Flux.zeros) =
    GAUGRUCell(init(out * 2, in),
               init(out * 3, na),
               init(out * 2, out),
               initb(out * 3),
               init_state(out,1))

function (m::GAUGRUCell)(h, x::Tuple{A, O}) where {A, O}
    b, o = m.b, size(h, 1)

    a = x[1]
    obs = x[2]

    gx, gh = m.Wi*obs, m.Wh*h
    ga = get_waa(m.Wa, a)

    r = σ.(gate(gx, o, 1) .+ gate(ga, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
    z = σ.(gate(ga, o, 2) .+ gate(b, o, 2))
    h̃ = tanh.(gate(gx, o, 2) .+ gate(ga, o, 3) .+ r .* gate(gh, o, 2) .+ gate(b, o, 3))
    h′ = (1 .- z) .* h̃ .+ z .* h
    sz = size(obs)
  return h′, reshape(h′, :, sz[2:end]...)
end

Flux.@functor GAUGRUCell

Base.show(io::IO, l::GAIGRUCell) =
  print(io, "GAUGRUCell(", size(l.Wi, 2), ", ", size(l.Wa), ", ", size(l.Wi, 1)÷3, ")")

"""
    GAUGRU(in::Integer, na::Integer, internal::Integer, out::Integer)
[Gated Action Input Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
GAUGRU(a...; ka...) = Flux.Recur(GAUGRUCell(a...; ka...))
Flux.Recur(m::GAUGRUCell) = Flux.Recur(m, m.state0)
