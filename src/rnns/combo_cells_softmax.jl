



struct CsoftmaxElRNNCell{F,A,V,T,S} <: AbstractActionRNN
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

CsoftmaxElRNNCell(in::Integer,
          na::Integer,
          out::Integer,
          σ=tanh;
          init=Flux.glorot_uniform,
          initb=Flux.zeros,
          init_state=Flux.zeros) = 
              CsoftmaxElRNNCell(σ,
                            # Flux.ones(2),
                            Flux.zeros(out),
                            Flux.zeros(out),
                            init(out, in),
                                init(out, na),
                            init(out, out),
                            initb(out),
                            init(na, out, in; ignore_dims=1),
                            init(na, out, out; ignore_dims=1),
                            initb(out, na),
                            init_state(out, 1))

function (m::CsoftmaxElRNNCell)(h, x::Tuple{A, X}) where {A, X}
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
    eθa = exp.(θa)
    eθm = exp.(θm)
    new_h = (eθa .* new_ha .+ eθm .* new_hm) ./ (eθa .+ eθm)

    sz = size(o)
    return new_h, reshape(new_h, :, sz[2:end]...)
end


Flux.@functor CsoftmaxElRNNCell

function Base.show(io::IO, l::CsoftmaxElRNNCell)
  print(io, "CsoftmaxElRNNCell(", size(l.Wi, 2), ", ", size(l.Wa), ", ", size(l.Wi, 1))
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
CsoftmaxElRNN(a...; ka...) = Flux.Recur(CsoftmaxElRNNCell(a...; ka...))
Flux.Recur(m::CsoftmaxElRNNCell) = Flux.Recur(m, m.state0)



struct CsoftmaxElGRUCell{A,V,T,S}  <: AbstractActionRNN
    θa::V
    θm::V
    Wia::A
    Wa::A
    Wha::A
    ba::V
    Wim::T
    Whm::T
    bm::A
    state0::S
end

CsoftmaxElGRUCell(in, 
            na, 
            out; 
            init = Flux.glorot_uniform, 
            initb = Flux.zeros, 
            init_state = Flux.zeros) =
                CsoftmaxElGRUCell(
                    Flux.zeros(out),
                    Flux.zeros(out),
                    init(out * 3, in),
                    init(out * 3, na),
                    init(out * 3, out),
                    initb(out * 3),
                    init(na, out * 3, in; ignore_dims=1),
                    init(na, out * 3, out; ignore_dims=1),
                    initb(out * 3, na),
                    init_state(out,1))

function (m::CsoftmaxElGRUCell)(h, x::Tuple{A, O}) where {A, O}
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

    eθa = exp.(m.θa)
    eθm = exp.(m.θm)
    h′ = (eθa .* h′a .+ eθm .* h′m) ./ (eθa .+ eθm)
    
    sz = size(obs)
    return h′, reshape(h′, :, sz[2:end]...)

end

Flux.@functor CsoftmaxElGRUCell

Base.show(io::IO, l::CsoftmaxElGRUCell) =
  print(io, "CsoftmaxElGRUCell(", size(l.Wia, 2), ", ", size(l.Wa), ", ", size(l.Wia, 1)÷3, ")")


"""
    CaddElGRU(in, actions, out)

Mixing between [`AAGRU`](@ref) and [`MAGRU`](@ref) through a weighting

```julia
h′ = (AA_θ .* AA_h′ .+ MA_θ .* MA_h′) ./ (AA_θ .+ MA_θ)
```

"""
CsoftmaxElGRU(a...; ka...) = Flux.Recur(CsoftmaxElGRUCell(a...; ka...))
Flux.Recur(m::CsoftmaxElGRUCell) = Flux.Recur(m, m.state0)



