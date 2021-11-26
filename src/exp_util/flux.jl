module FluxUtils


using Flux

function get_optimizer(parsed::Dict; opt_key="opt")
    opt_string = parsed[opt_key]
    get_optimizer(opt_string, parsed)
end

function get_optimizer(opt_string, parsed)
    opt_type = getproperty(FluxUtils, Symbol(opt_string))
    _init_optimizer(opt_type, parsed)
end

struct MissingParamInit end
struct OneParamInit end
struct TwoParamInit end
struct AdamParamInit end

param_init_style(opt_type) = MissingParamInit()
param_init_style(::Union{Type{Descent}, Type{ADAGrad}, Type{ADADelta}}) = OneParamInit()
param_init_style(::Union{Type{RMSProp}, Type{Momentum}, Type{Nesterov}}) = TwoParamInit()
param_init_style(::Union{Type{ADAM}, Type{RADAM}, Type{NADAM}, Type{AdaMax}, Type{OADAM}, Type{AMSGrad}, Type{AdaBelief}}) = AdamParamInit()

function _init_optimizer(opt_type, parsed::Dict)
    _init_optimizer(param_init_style(opt_type), opt_type, parsed::Dict)
end

function _init_optimizer(::MissingParamInit, args...)
    throw("Optimizer initialization not found")
end

function _init_optimizer(::OneParamInit, opt_type, parsed::Dict)
    try
        η = parsed["eta"]
        opt_type(η)
    catch
        throw("$(opt_type) needs: eta (float)")
    end
end

function _init_optimizer(::TwoParamInit, opt_type, parsed::Dict)
    try
        η = parsed["eta"]
        ρ = parsed["rho"]
        opt_type(η, ρ)
    catch
        throw("$(opt_type) needs: eta (float), and rho (float)")
    end
end

function _init_optimizer(::AdamParamInit, opt_type, parsed::Dict)
    try
        η = parsed["eta"]
        if "beta" ∈ keys(parsed)
            β = parsed["beta"]
            opt_type(η, β)
        elseif "beta_m" ∈ keys(parsed)
            β = (parsed["beta_m"], parsed["beta_v"])
            opt_type(η, β)
        end
        opt_type(η)
    catch
        throw("$(opt_type) needs: eta (float), and beta ((float, float)), or (beta_m, beta_v))")
    end
end

function get_activation(act::AbstractString)
    if act == "sigmoid"
        return Flux.σ
    elseif act == "tanh"
        return tanh
    elseif act == "linear"
        return Flux.identity
    elseif act == "relu"
        return Flux.relu
    elseif act == "softplus"
        return Flux.softplus
    else
        throw("$(act) not known...")
    end
end


struct RMSPropTFParamInit end


"""
    RMSPropTF(η, ρ)
Implements the RMSProp algortihm as implemented in tensorflow. 
  - Learning Rate (η): Defaults to `0.001`.
  - Rho (ρ): Defaults to `0.9`.
  - Gamma (γ): Defaults to `0.0`.
  - Epsilon (ϵ): Defaults to `1e-6`
## Examples
## References
[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
[Tensorflow RMSProp](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop)
"""
mutable struct RMSPropTF
    eta::Float64
    rho::Float64
    gamma::Float64
    epsilon::Float64
    acc::IdDict
    mom::IdDict
end

RMSPropTF(η = 0.001, ρ = 0.9, γ = 0.0, ϵ = 1e-6) = RMSPropTF(η, ρ, γ, ϵ, IdDict(), IdDict())


function Flux.Optimise.apply!(o::RMSPropTF, x, Δ)
    η, ρ, γ, ϵ = o.eta, o.rho, o.gamma, o.epsilon
    acc = get!(o.acc, x, zero(x))::typeof(x)
    mom = get!(o.mom, x, zero(x))::typeof(x)
    @. acc = ρ * acc + (1 - ρ) * Δ^2
    @. mom = γ * mom + η * Δ/(sqrt(acc + ϵ))
    mom
end



"""
    RMSPropTFCentered(η, ρ)
Implements the Centered version of RMSProp algortihm as implemented in tensorflow. 
  - Learning Rate (η): Defaults to `0.001`.
  - Rho (ρ): Defaults to `0.9`.
  - Gamma (γ): Defaults to `0.0`.
  - Epsilon (ϵ): Defaults to `1e-6`
## Examples
## References
[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
[Tensorflow RMSProp](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop)
"""
mutable struct RMSPropTFCentered
    eta::Float64
    rho::Float64
    gamma::Float64
    epsilon::Float64
    grad::IdDict
    acc::IdDict
    mom::IdDict
end

RMSPropTFCentered(η = 0.001, ρ = 0.9, γ = 0.0, ϵ = 1e-6) =
    RMSPropTFCentered(η, ρ, γ, ϵ, IdDict(), IdDict(), IdDict())

function Flux.Optimise.apply!(o::RMSPropTFCentered, x, Δ)
    η, ρ, γ, ϵ = o.eta, o.rho, o.gamma, o.epsilon
    grad = get!(o.grad, x, zero(x))::typeof(x)
    acc = get!(o.acc, x, zero(x))::typeof(x)
    mom = get!(o.mom, x, zero(x))::typeof(x)
    @. grad = ρ * grad + (1 - ρ) * Δ
    @. acc = ρ * acc + (1 - ρ) * Δ^2
    @. mom = γ * mom + η * Δ/(sqrt(acc - grad^2 + ϵ))
    mom
end


param_init_style(::Union{Type{RMSPropTF}, Type{RMSPropTFCentered}}) = RMSPropTFParamInit()

function _init_optimizer(::RMSPropTFParamInit, opt_type, parsed::Dict)
    try
        η = parsed["eta"]
        ρ = parsed["rho"]
        mom = parsed["mom"]
        
        opt_type(η, ρ, mom)
    catch
        throw("$(opt_type) needs: eta (float), and rho (float)")
    end
end

end # module FluxUtils
