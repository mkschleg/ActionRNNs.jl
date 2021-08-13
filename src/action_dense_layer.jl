

struct ActionDense{F, M<:AbstractMatrix, B}
    weight::M
    weight_a::M
    bias::B
    σ::F
    function ActionDense(W::M, Wa::M, bias = true, σ::F = identity) where {M<:AbstractMatrix, F}
        b = Flux.create_bias(W, bias, size(W,1))
        new{F,M,typeof(b)}(W, Wa, b, σ)
    end
end

_needs_action_input(m::ActionDense) = true

function ActionDense(in::Integer, na::Integer, out::Integer, σ = identity;
               initW = nothing, initb = nothing,
               init = glorot_uniform, bias=true)

  W, Wa = if initW !== nothing
      Base.depwarn("keyword initW is deprecated, please use init (which similarly accepts a funtion like randn)", :Dense)
      initW(out, in), initW(out, na)
  else
      init(out, in), init(out, na)
  end

  b = if bias === true && initb !== nothing
    Base.depwarn("keyword initb is deprecated, please simply supply the bias vector, bias=initb(out)", :Dense)
    initb(out)
  else
    bias
  end

  return ActionDense(W, Wa, b, σ)
end

Flux.@functor ActionDense

function (a::ActionDense)(x::Tuple{A, X}) where {A, X}
    W, Wa, b, σ = a.weight, a.weight_a, a.bias, a.σ
    
    return σ.(W*x .+ get_waa(Wa, a) .+ b)
end

# (a::ActionDense)(x::AbstractArray) = 
#   reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)

function Base.show(io::IO, l::ActionDense)
  print(io, "Dense(", size(l.weight, 2), ", ", size(l.weight, 1))
  l.σ == identity || print(io, ", ", l.σ)
  l.bias == Zeros() && print(io, "; bias=false")
  print(io, ")")
end