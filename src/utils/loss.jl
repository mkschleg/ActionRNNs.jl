using LinearAlgebra

using Flux
using Flux.Zygote: dropgrad

import Statistics: mean



"""
    smooth_l1_loss(y, fx; δ)
l1 loss w/ clipping. Also known as the huber loss. Only usable w/ numbers (see `huber_loss` for vector form)
"""
function smooth_l1_loss(y::Number, fx::Number, δ=1)
    α = abs(y - fx)
    α <= δ && return 0.5f0 * α ^ 2
    δ * α - (0.5f0 * δ ^ 2)
end

"""
    huber_loss(y, fx; δ)
Huber loss. Convenience for smoot_l1_loss
"""
function huber_loss(y, fx; δ=1)
    return smooth_l1_loss.(y, fx, δ)
end

"""
    mean_huber_loss(y, fx; δ)
mean huber loss.
"""
function mean_huber_loss(y, fx; δ=1)
    return mean(huber_loss(y, fx; δ=δ))
end

