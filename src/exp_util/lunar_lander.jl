module LunarLanderUtils

using ..MinimalRLCore

struct IdentityFeatureCreator <: AbstractFeatureConstructor end

(fc::IdentityFeatureCreator)(s, a) = MinimalRLCore.create_features(fc, s, a)
MinimalRLCore.create_features(fc::IdentityFeatureCreator, s, a) = Float32.(s)
MinimalRLCore.feature_size(fc::IdentityFeatureCreator) = 8

end
