
module RingWorldUtils

import ..RingWorld
import ..RingWorldConst

import MinimalRLCore: MinimalRLCore, AbstractFeatureConstructor

import GVFHordes: GVFHordes, GVF, Horde, GVFParamFuncs
import .GVFParamFuncs: FeatureCumulant, PredictionCumulant, ConstantDiscount, StateTerminationDiscount, PersistentPolicy

const RWC = RingWorldConst


function onestep()
    gvfs = [GVF(FeatureCumulant(1), ConstantDiscount(0.0), PersistentPolicy(RWC.FORWARD))
            GVF(FeatureCumulant(1), ConstantDiscount(0.0), PersistentPolicy(RWC.BACKWARD))]
    return Horde(gvfs)
end

function half_chain(chain_length::Integer, pred_offset::Integer=0)
    gvfs = [[GVF(FeatureCumulant(1), ConstantDiscount(0.0), PersistentPolicy(RWC.FORWARD))];
            [GVF(PredictionCumulant(i-1 + pred_offset), ConstantDiscount(0.0), PersistentPolicy(RWC.FORWARD)) for i in 2:chain_length];
            ]
    return Horde(gvfs)
end

function gamma_half_chain(chain_length::Integer, γ::AbstractFloat, pred_offset::Integer=0)
    gvfs = [[GVF(FeatureCumulant(1), ConstantDiscount(0.0), PersistentPolicy(RWC.FORWARD))];
            [GVF(PredictionCumulant(i-1 + pred_offset), ConstantDiscount(0.0), PersistentPolicy(RWC.FORWARD)) for i in 2:chain_length];
            [GVF(FeatureCumulant(1), StateTerminationDiscount(γ, ((env_state)->env_state[1] == 1)), PersistentPolicy(RWC.FORWARD))];
            ]
    return Horde(gvfs)
end

function chain(chain_length::Integer, pred_offset::Integer=0)
    gvfs = [[GVF(FeatureCumulant(1), ConstantDiscount(0.0), PersistentPolicy(RWC.FORWARD))];
            [GVF(PredictionCumulant(i-1 + pred_offset), ConstantDiscount(0.0), PersistentPolicy(RWC.FORWARD)) for i in 2:chain_length];
            [GVF(FeatureCumulant(1), ConstantDiscount(0.0), PersistentPolicy(RWC.BACKWARD))];
            [GVF(PredictionCumulant(chain_length + i-1 + pred_offset), ConstantDiscount(0.0), PersistentPolicy(RWC.BACKWARD)) for i in 2:chain_length]
            ]
    return Horde(gvfs)
end

function gamma_chain(chain_length::Integer, γ::AbstractFloat, pred_offset::Integer=0)
    gvfs = [[GVF(FeatureCumulant(1), ConstantDiscount(0.0), PersistentPolicy(RWC.FORWARD))];
            [GVF(PredictionCumulant(i-1 + pred_offset), ConstantDiscount(0.0), PersistentPolicy(RWC.FORWARD)) for i in 2:chain_length];
            [GVF(FeatureCumulant(1), StateTerminationDiscount(γ, ((env_state)->env_state[1] == 1)), PersistentPolicy(RWC.FORWARD))];
            [GVF(FeatureCumulant(1), ConstantDiscount(0.0), PersistentPolicy(RWC.BACKWARD))];
            [GVF(PredictionCumulant(chain_length + 1 + i-1 + pred_offset), ConstantDiscount(0.0), PersistentPolicy(RWC.BACKWARD)) for i in 2:chain_length];
            [GVF(FeatureCumulant(1), StateTerminationDiscount(γ, ((env_state)->env_state[1] == 1)), PersistentPolicy(RWC.BACKWARD))];
            ]
    return Horde(gvfs)
end

function gamma_chain_scaled(chain_length::Integer, γ::AbstractFloat, pred_offset::Integer=0)
    gvfs = [[GVF(FeatureCumulant(1), ConstantDiscount(0.0), PersistentPolicy(RWC.FORWARD))];
            [GVF(PredictionCumulant(i-1 + pred_offset), ConstantDiscount(0.0), PersistentPolicy(RWC.FORWARD)) for i in 2:chain_length];
            [GVF(ScaledCumulant(1-γ, FeatureCumulant(1)), ConstantDiscount(γ), PersistentPolicy(RWC.FORWARD))];
            [GVF(FeatureCumulant(1), ConstantDiscount(0.0), PersistentPolicy(RWC.BACKWARD))];
            [GVF(PredictionCumulant(chain_length + 1 + i-1), ConstantDiscount(0.0), PersistentPolicy(RWC.BACKWARD)) for i in 2:chain_length]
            [GVF(ScaledCumulant(1-γ, FeatureCumulant(1)), ConstantDiscount(γ), PersistentPolicy(RWC.BACKWARD))]]
    return Horde(gvfs)
end

function gammas(gms::Array{Float64, 1})
    gvfs = [[GVF(FeatureCumulant(1), ConstantDiscount(γ), PersistentPolicy(RWC.FORWARD)) for γ in gms];
            [GVF(FeatureCumulant(1), ConstantDiscount(γ), PersistentPolicy(RWC.BACKWARD)) for γ in gms]]
    return Horde(gvfs)
end

function gammas_scaled(gms::AbstractArray)
    gvfs = [[GVF(ScaledCumulant(1-γ, FeatureCumulant(1)), ConstantDiscount(γ), PersistentPolicy(RWC.FORWARD)) for γ in gms];
            [GVF(ScaledCumulant(1-γ, FeatureCumulant(1)), ConstantDiscount(γ), PersistentPolicy(RWC.BACKWARD)) for γ in gms]]
    return Horde(gvfs)
end

function gammas_term(gms::AbstractArray)
    gvfs = [[GVF(FeatureCumulant(1), StateTerminationDiscount(γ, ((env_state)->env_state[1] == 1)), PersistentPolicy(RWC.FORWARD)) for γ in gms];
            [GVF(FeatureCumulant(1), StateTerminationDiscount(γ, ((env_state)->env_state[1] == 1)), PersistentPolicy(RWC.BACKWARD)) for γ in gms]]
    return Horde(gvfs)
end

function gammas_aj()
    gms = 1.0 .- 2.0 .^ collect(-7:-1)
    return gammas(gms)
end

function gammas_aj_term()
    gms = 1.0 .- 2.0 .^ collect(-7:-1)
    return gammas_term(gms)
end

function gammas_aj_scaled()
    gms = 1.0 .- 2.0 .^ collect(-7:-1)
    return gammas_scaled(gms)
end

function get_horde(horde_str::AbstractString, chain_length::Integer, gamma::AbstractFloat, pred_offset::Integer=0)
    horde = chain(chain_length, pred_offset)
    if horde_str == "gamma_chain"
        horde = gamma_chain(chain_length, gamma, pred_offset)
    elseif horde_str == "half_chain"
        horde = half_chain(chain_length, pred_offset)
    elseif horde_str == "gamma_half_chain"
        horde = gamma_half_chain(chain_length, gamma, pred_offset)
    elseif horde_str == "onestep"
        horde = onestep()
    elseif horde_str == "gammas"
        horde = gammas(collect(0.0:0.1:0.9))
    elseif horde_str == "gammas_term"
        horde = gammas_term(collect(0.0:0.1:0.9))
    elseif horde_str == "gammas_term_one"
        horde = gammas_term([0.9])
    elseif horde_str == "gammas_scaled"
        horde = gammas_scaled(collect(0.0:0.1:0.9))
    elseif horde_str == "gammas_aj"
        horde = gammas_aj()
    elseif horde_str == "gammas_aj_term"
        horde = gammas_aj_term()
    elseif horde_str == "gammas_aj_scaled"
        horde = gammas_aj_scaled()
    end
    return horde
end

get_horde(parsed::Dict, prefix="", pred_offset::Integer=0) =
    get_horde(parsed["$(prefix)horde"], parsed["size"], parsed["$(prefix)gamma"], pred_offset)

function oracle(env::RingWorld, horde_str, γ=0.9)
    chain_length = env.ring_size
    state = env.agent_state
    ret = Array{Float64,1}()
    if horde_str == "chain"
        ret = zeros(chain_length*2)
        ret[1 + chain_length - state] = 1
        ret[chain_length + state] = 1
    elseif horde_str == "gamma_chain"
        ret = zeros(chain_length*2 + 2)
        ret[1 + chain_length - state] = 1
        ret[chain_length + 1 + state] = 1
        ret[chain_length + 1] = γ^(chain_length - state)
        ret[end] = γ^(state - 1)
    elseif horde_str == "onestep"
        #TODO: Hack fix.
        ret = zeros(2)
        ret[1] = state == chain_length ? 1 : 0
        ret[2] = state == 2 ? 1 : 0
    elseif horde_str == "gammas_term"
        ret = zeros(20)
        if state == 1
            ret[1:10] .= collect(0.0:0.1:0.9).^(chain_length)
            ret[11:end] .= collect(0.0:0.1:0.9).^(chain_length)
        else
            ret[1:10] .= collect(0.0:0.1:0.9).^(chain_length - state)
            ret[11:end] .= collect(0.0:0.1:0.9).^(state - 2)
        end
    elseif horde_str == "gammas_term_one"
        ret = zeros(2)
        if state == 1
            ret[1] = 0.9^(chain_length)
            ret[2] = 0.9^(chain_length)
        else
            ret[1] = 0.9^(chain_length - state)
            ret[2] = 0.9^(state - 2)
        end
    else
        throw("Bug Found")
    end

    return ret
end



mutable struct StandardFeatureCreator{T} <: AbstractFeatureConstructor end


(fc::StandardFeatureCreator)(s, a) = MinimalRLCore.create_features(fc, s, a)

MinimalRLCore.create_features(fc::StandardFeatureCreator{true}, s, a) =
    Float32[1.0, s[1], 1-s[1], a==1, a==2, 1.0 - a==1, 1.0 - a==2]
MinimalRLCore.feature_size(fc::StandardFeatureCreator{true}) = 7

# MinimalRLCore.create_features(fc::StandardFeatureCreator{false}, s, a) =
#     Float32[s[1], 1-s[1]]
# MinimalRLCore.feature_size(fc::StandardFeatureCreator{false}) = 2

MinimalRLCore.create_features(fc::StandardFeatureCreator{false}, s, a) =
    Float32[s[1]]
MinimalRLCore.feature_size(fc::StandardFeatureCreator{false}) = 1

Base.size(fc::StandardFeatureCreator) = MinimalRLCore.feature_size(fc)

mutable struct SansBiasFeatureCreator <: AbstractFeatureConstructor end

(fc::SansBiasFeatureCreator)(s, a=nothing) = MinimalRLCore.create_features(fc, s, a)
MinimalRLCore.create_features(fc::SansBiasFeatureCreator, s, a) =
    Float32[s[1], 1-s[1], a==1, a==2, 1.0 - a==1, 1.0 - a==2]
MinimalRLCore.create_features(fc::SansBiasFeatureCreator, s, a::Nothing) =
    Float32[s[1], 1-s[1], 0, 0, 0, 0]
MinimalRLCore.feature_size(fc::SansBiasFeatureCreator) = 6

mutable struct OneHotFeatureCreator <: AbstractFeatureConstructor end

(fc::OneHotFeatureCreator)(s, a=nothing) = MinimalRLCore.create_features(fc, s, a)
MinimalRLCore.create_features(fc::OneHotFeatureCreator, s, a) =
    Float32[s[1], 1-s[1], a==1, a==2]
MinimalRLCore.create_features(fc::OneHotFeatureCreator, s, a::Nothing) =
    Float32[s[1], 1-s[1], 0, 0]
MinimalRLCore.feature_size(fc::OneHotFeatureCreator) = 4


mutable struct SansActionFeatureCreator <: AbstractFeatureConstructor end

(fc::SansActionFeatureCreator)(s, a=nothing) = MinimalRLCore.create_features(fc, s, a)
MinimalRLCore.create_features(fc::SansActionFeatureCreator, s, a) =
    Float32[s[1], 1-s[1]]
MinimalRLCore.create_features(fc::SansActionFeatureCreator, s, a::Nothing) =
    Float32[s[1], 1-s[1]]
MinimalRLCore.feature_size(fc::SansActionFeatureCreator) = 2

# build_features_ringworld_sans_bias(s, a) = Float32[s[1], 1-s[1], a==1, a==2, 1.0 - a==1, 1.0 - a==2]

end
