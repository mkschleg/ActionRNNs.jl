module FluxUtils

using ..Flux

function get_optimizer(parsed::Dict; opt_key="opt")
    opt_string = parsed[opt_key]
    get_optimizer(opt_string, parsed)
end

function get_optimizer(opt_string, parsed)
    opt_type = getproperty(Flux, Symbol(opt_string))
    _init_optimizer(opt_type, parsed)
end

function _init_optimizer(opt, parsed::Dict)
    throw("Optimizer initialization not found")
end

function _init_optimizer(opt_type::Union{Type{Descent}, Type{ADAGrad}, Type{ADADelta}}, parsed::Dict)
    try
        η = parsed["eta"]
        opt_type(η)
    catch
        throw("$(opt_type) needs: eta (float)")
    end
end

function _init_optimizer(opt_type::Union{Type{RMSProp}, Type{Momentum}, Type{Nesterov}}, parsed::Dict)
    try
        η = parsed["eta"]
        ρ = parsed["rho"]
        opt_type(η, ρ)
    catch
        throw("$(opt_type) needs: eta (float), and rho (float)")
    end
end

function _init_optimizer(opt_type::Union{Type{ADAM}, Type{RADAM}, Type{NADAM}, Type{AdaMax}, Type{OADAM}, Type{AMSGrad}, Type{AdaBelief}}, parsed::Dict)
    try
        η = parsed["eta"]
        β = if "beta" ∈ keys(parsed)
            parsed["beta"]
        else
            (parsed["beta_m"], parsed["beta_v"])
        end
        opt_type(η, β)
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


end
