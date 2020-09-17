module TMazeUtils

using ..ActionRNNs
using ..MinimalRLCore

mutable struct OneHotFeatureCreator{WA} <: AbstractFeatureConstructor end

OneHotFeatureCreator() = OneHotFeatureCreator{true}()

(fc::OneHotFeatureCreator)(s, a) = MinimalRLCore.create_features(fc, s, a)
MinimalRLCore.create_features(fc::OneHotFeatureCreator, s, a) =
    Float32[s; [a==1, a==2, a==3, a==4]]
MinimalRLCore.create_features(fc::OneHotFeatureCreator, s, a::Nothing) =
    Float32[s; [0, 0, 0, 0]]
MinimalRLCore.feature_size(fc::OneHotFeatureCreator) = 7


end
