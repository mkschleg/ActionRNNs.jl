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
