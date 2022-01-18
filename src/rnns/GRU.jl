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
    AAGRU(in, actions, out)
Additive Action Gated Recurrent Unit layer. Behaves like an
[`AARNN`](@ref) but uses a GRU internal structure
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
    MAGRU(in, actions, out)
Multiplicative Action Gated Recurrent Unit layer. Behaves like an
[`MARNN`](@ref) but uses a GRU internal structure.
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

function FacMAGRUCell(args...; init_style="standard", kwargs...)
    init_cell_name = "FacMAGRUCell_$(init_style)"
    rnn_init = getproperty(ActionRNNs, Symbol(init_cell_name))
    ret = rnn_init(args...; kwargs...)
    println(typeof(ret))
    ret
end


FacMAGRUCell_standard(in, na, out, factors; init = glorot_uniform, initb = Flux.zeros, init_state = Flux.zeros) =
    FacMAGRUCell(init(out * 3, factors),
                 init(factors, in),
                 init(factors, out),
                 init(factors, na),
                 initb(out * 3, na),
                 init_state(out,1))

function FacMAGRUCell_tensor(in, na, out, factors; init = glorot_uniform, initb = Flux.zeros, init_state = Flux.zeros)

    W_t = init(na, out*3, in+out; ignore_dims=1)
    W_d = cp_als(W_t, factors)
    
    W_a, W_o, W_hi = W_d.fmat
    W_o .*= W_d.lambda'

    FacMAGRUCell(Float32.(W_o),
                 Float32.(transpose(W_hi[1:in, :])),
                 Float32.(transpose(W_hi[(in+1):end, :])),
                 Float32.(transpose(W_a)),
                 initb(out*3, na),
                 init_state(out, 1))
    
end

function (m::FacMAGRUCell)(h, x::Tuple{A, O}) where {A, O}
    o = size(h, 1)

    a = x[1]
    obs = x[2]

    wa = get_Wabya(m.Wa, a)
    gx, gh = m.W * (m.Wi*obs .* wa), m.W * (m.Wh*h .* wa)
    b = get_waa(m.b, a)

    r = σ.(gate(gx, o, 1)  .+ gate(gh, o, 1) .+ gate(b, o, 1))
    z = σ.(gate(gx, o, 2)  .+ gate(gh, o, 2) .+ gate(b, o, 2))
    h̃ = tanh.(gate(gx, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
    h′ = (1 .- z) .* h̃ .+ z .* h
    sz = size(obs)
  return h′, reshape(h′, :, sz[2:end]...)
end

Flux.@functor FacMAGRUCell

Base.show(io::IO, l::FacMAGRUCell) =
  print(io, "FacMAGRUCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ")")

"""
    FacMAGRU(in, actions, out, factors)
Factored Multiplicative Action Gated Recurrent Unit layer. Behaves like an
[`FacMARNN`](@ref) but uses a GRU internal structure.

Three init_styles:
- standard: using init and initb w/o any keywords
- ignore: `W = init(out, factors, ignore_dims=2)`
- tensor: Decompose `W_t = init(actions, out, in+out; ignore_dims=1)` to get `W_o, W_a, W_hi` using `TensorToolbox.cp_als`.

"""
FacMAGRU(a...; ka...) = Flux.Recur(FacMAGRUCell(a...; ka...))
Flux.Recur(m::FacMAGRUCell) = Flux.Recur(m, m.state0)


struct FacTucMAGRUCell{T,A,V,S}  <: AbstractActionRNN
    Wg::T
    Wa::A
    Wh::A
    Wxx::A
    Wxh::A
    b::V
    state0::S
end

function FacTucMAGRUCell(args...; init_style="standard", kwargs...)
    init_cell_name = "FacTucMAGRUCell_$(init_style)"
    rnn_init = getproperty(ActionRNNs, Symbol(init_cell_name))
    rnn_init(args...; kwargs...)
end

FacTucMAGRUCell_standard(in, na, out, action_factors, out_factors, in_factors;
             init = glorot_uniform, initb = Flux.zeros, init_state = Flux.zeros) =
    FacTucMAGRUCell(init(action_factors, out_factors, in_factors),
                 init(action_factors, na),
                 init(out * 3, out_factors),
                 init(in_factors, in),
                 init(in_factors, out),
                 initb(out * 3, na),
                 init_state(out,1))

FacTucMAGRUCell_ignore(in, na, out, action_factors, out_factors, in_factors;
             init = glorot_uniform, initb = Flux.zeros, init_state = Flux.zeros) =
    FacTucMAGRUCell(init(action_factors, out_factors, in_factors),
                 init(action_factors, na; ignore_dims=2),
                 init(out * 3, out_factors),
                 init(in_factors, in),
                 init(in_factors, out),
                 initb(out * 3, na; ignore_dims=2),
                 init_state(out,1))

function (m::FacTucMAGRUCell)(h, x::Tuple{A, O}) where {A, O}
    o = size(h, 1)

    a = x[1]
    obs = x[2]

    waa = get_Wabya(m.Wa, a)
    gx, gh = if a isa Int
        m.Wh * (contract_Wga(m.Wg, waa) * (m.Wxx*obs)), m.Wh * (contract_Wga(m.Wg, waa) * (m.Wxh*h[:]))
    else
        m.Wh * contract_Wgax(m.Wg, waa, m.Wxx*obs), m.Wh * contract_Wgax(m.Wg, waa, m.Wxh*h)
    end
    b = get_waa(m.b, a)

    r = σ.(gate(gx, o, 1)  .+ gate(gh, o, 1) .+ gate(b, o, 1))
    z = σ.(gate(gx, o, 2)  .+ gate(gh, o, 2) .+ gate(b, o, 2))
    h̃ = tanh.(gate(gx, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
    h′ = (1 .- z) .* h̃ .+ z .* h
    sz = size(obs)
  return h′, reshape(h′, :, sz[2:end]...)
end

Flux.@functor FacTucMAGRUCell

Base.show(io::IO, l::FacTucMAGRUCell) =
  print(io, "FacTucMAGRUCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ")")

"""
    FacTucMAGRU(in, actions, out, factors)
Factored Multiplicative Action Gated Recurrent Unit layer. Behaves like an
[`FacTucMARNN`](@ref) but uses a GRU internal structure.
"""
FacTucMAGRU(a...; ka...) = Flux.Recur(FacTucMAGRUCell(a...; ka...))
Flux.Recur(m::FacTucMAGRUCell) = Flux.Recur(m, m.state0)
