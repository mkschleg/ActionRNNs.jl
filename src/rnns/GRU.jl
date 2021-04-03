using Flux
using Flux: gate

# gate(h, n) = (1:h) .+ h*(n-1)
# gate(x::AbstractVector, h, n) = @view x[gate(h,n)]
# gate(x::AbstractMatrix, h, n) = x[gate(h,n),:]


struct AAGRUCell{A,V,S}  <: AbstractActionRNN
    Wi::A
    Wa::A
    Wh::A
    b::V
    state0::S
end

AAGRUCell(in, na, out; init = glorot_uniform, initb = zeros, init_state = zeros) =
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
    GRU(in::Integer, out::Integer)
[Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
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

MAGRUCell(in, na, out; init = glorot_uniform, initb = zeros, init_state = zeros) =
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
    GRU(in::Integer, out::Integer)
[Gated Recurrent Unit](https://arxiv.org/abs/1406.1078) layer. Behaves like an
RNN but generally exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
MAGRU(a...; ka...) = Flux.Recur(MAGRUCell(a...; ka...))
Flux.Recur(m::MAGRUCell) = Flux.Recur(m, m.state0)

# mutable struct ActionGRUCell{A1,A2, V1, V2, H} <: AbstractActionRNN
#     Wi_input::A1
#     Wh_input::A1
#     b_input::V1
#     Wi::A2
#     Wh::A2
#     b::V2
#     state0::H
# end

# function ActionGRUCell(in::Integer, num_actions::Integer, out::Integer;
#                         init = Flux.glorot_uniform)
#     ActionGRUCell(
#         [init(in, out) for a in 1:num_actions],
#         [init(out, out) for a in 1:num_actions],
#         [Flux.zeros(out) for a in 1:num_actions],
#         init(out * 2, in),
#         init(out * 2, out),
#         Flux.zeros(out * 2),
#         Flux.zeros(out))
# end


# function (m::ActionGRUCell)((h, c), ax::Tuple{I, A}) where {I<:Integer, A<:AbstractArray}
#     a, x = ax
#     b, o = m.b, size(h, 1)
#     gx, gh = m.Wi*x, m.Wh*h
#     gx_input, gh_input, b_input = m.Wi_input[a]*x, m.Wh_input[a]*h, m.b_input[a]
#     r = σ.(gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
#     z = σ.(gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
#     h̃ = tanh.(gx_input .+ r .* gh_input .+ b_input)
#     h′ = (1 .- z).*h̃ .+ z.*h
#     return h′, h′
# end

# # Flux.hidden(m::ActionGRUCell) = m.h

# Flux.@functor ActionGRUCell

# Base.show(io::IO, l::ActionGRUCell) =
#     print(io, "ActionGRUCell(", size(l.Wi_input[1], 2), ", ", size(l.Wi, 1), ")")

# """
#     AGRU(in::Integer, num_actions::Integer, out::Integer)
#  Gated Recurrent Unit layer. Behaves like an RNN but generally
# exhibits a longer memory span over sequences.
# See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
# for a good overview of the internals.
# """
# AGRU(a...; ka...) = Flux.Recur(ActionGRUCell(a...; ka...))
