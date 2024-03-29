
# Sepcifying a action-conditional RNN Cell
using Flux
using Tullio
import TensorToolbox: cp_als
using Flux: gate


struct CaddRNNCell{F,A,V,T,S} <: AbstractActionRNN
    σ::F
    w::V
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
                            Flux.ones(2),
                            init(out, in),
                            init(out, na),
                            init(out, out),
                            initb(out),
                            init(na, out, in; ignore_dims=1),
                            init(na, out, out; ignore_dims=1),
                            initb(out, na),
                            init_state(out, 1))

function (m::CaddRNNCell)(h, x::Tuple{A, X}) where {A, X}
    σ, w, Wi, Wa, Wha, ba = m.σ, m.w, m.Wi, m.Wa, m.Wha, m.ba
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
    new_h = (w[1]*new_ha + w[2]*new_hm) ./ sum(w)

    sz = size(o)
    return new_h, reshape(new_h, :, sz[2:end]...)
end


Flux.@functor CaddRNNCell

function Base.show(io::IO, l::CaddRNNCell)
  print(io, "CaddRNNCell(", size(l.Wi, 2), ", ", size(l.Wa), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    CaddRNN(in, actions, out, σ = tanh)

Mixing between [`AARNN`](@ref) and [`MARNN`](@ref) through a weighting

```julia
h′ = (w[1]*new_hAA + w[2]*new_hMA) ./ sum(w)
```

"""
CaddRNN(a...; ka...) = Flux.Recur(CaddRNNCell(a...; ka...))
Flux.Recur(m::CaddRNNCell) = Flux.Recur(m, m.state0)


struct CcatRNNCell{F,A,V,T,S} <: AbstractActionRNN
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

CcatRNNCell(in::Integer,
          na::Integer,
          out::Integer,
          σ=tanh;
          init=Flux.glorot_uniform,
          initb=Flux.zeros,
          init_state=Flux.zeros) = 
              CcatRNNCell(σ,
                        init(out, in),
                        init(out, na),
                        init(out, out*2),
                        initb(out),
                        init(na, out, in; ignore_dims=1),
                        init(na, out, out*2; ignore_dims=1),
                        initb(out, na),
                        init_state(out*2, 1))

function (m::CcatRNNCell)(h, x::Tuple{A, X}) where {A, X}
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

    # concatenating together state
    new_h = vcat(new_ha, new_hm)

    sz = size(o)
    return new_h, reshape(new_h, :, sz[2:end]...)
end


Flux.@functor CcatRNNCell

function Base.show(io::IO, l::CcatRNNCell)
  print(io, "CcatRNNCell(", size(l.Wi, 2), ", ", size(l.Wa), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end


"""
    CcatRNN(in, actions, out, σ = tanh)

Mixing between [`AARNN`](@ref) and [`MARNN`](@ref) through

```julia
h′ = cat(AA_h′, MA_h′)
```

"""
CcatRNN(a...; ka...) = Flux.Recur(CcatRNNCell(a...; ka...))
Flux.Recur(m::CcatRNNCell) = Flux.Recur(m, m.state0)





#=
Element wise add.
=#


struct CaddElRNNCell{F,A,V,T,S} <: AbstractActionRNN
    σ::F
    θa::V
    θm::V
    Wi::A
    Wa::A
    Wha::A
    ba::V
    Wx::T
    Whm::T
    bm::A
    state0::S
end

CaddElRNNCell(in::Integer,
          na::Integer,
          out::Integer,
          σ=tanh;
          init=Flux.glorot_uniform,
          initb=Flux.zeros,
          init_state=Flux.zeros) = 
              CaddElRNNCell(σ,
                            # Flux.ones(2),
                            Flux.ones(out),
                            Flux.ones(out),
                            init(out, in),
                            init(out, na),
                            init(out, out),
                            initb(out),
                            init(na, out, in; ignore_dims=1),
                            init(na, out, out; ignore_dims=1),
                            initb(out, na),
                            init_state(out, 1))

function (m::CaddElRNNCell)(h, x::Tuple{A, X}) where {A, X}
    σ, θa, θm, Wi, Wa, Wha, ba = m.σ, m.θa, m.θm, m.Wi, m.Wa, m.Wha, m.ba
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
    # new_h = (w[1]*new_ha + w[2]*new_hm) ./ sum(w)
    new_h = (θa .* new_ha .+ θm .* new_hm) ./ (θa .+ θm)

    sz = size(o)
    return new_h, reshape(new_h, :, sz[2:end]...)
end


Flux.@functor CaddElRNNCell

function Base.show(io::IO, l::CaddElRNNCell)
  print(io, "CaddElRNNCell(", size(l.Wi, 2), ", ", size(l.Wa), ", ", size(l.Wi, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

"""
    CaddElRNN(in, actions, out, σ = tanh)

Mixing between [`AARNN`](@ref) and [`MARNN`](@ref) through a weighting

```julia
h′ = (AA_θ .* AA_h′ .+ MA_θ .* MA_h′) ./ (AA_θ .+ MA_θ)
```
"""
CaddElRNN(a...; ka...) = Flux.Recur(CaddElRNNCell(a...; ka...))
Flux.Recur(m::CaddElRNNCell) = Flux.Recur(m, m.state0)
