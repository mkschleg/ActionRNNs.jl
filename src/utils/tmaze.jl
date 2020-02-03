module TMazeUtils

using ..ActionRNN, Reproduce
using ..RLCore

mutable struct OneHotFeatureCreator <: AbstractFeatureConstructor end

(fc::OneHotFeatureCreator)(s, a) = RLCore.create_features(fc, s, a)
RLCore.create_features(fc::OneHotFeatureCreator, s, a) =
    Float32[s; [a==1, a==2, a==3, a==4]]
RLCore.create_features(fc::OneHotFeatureCreator, s, a::Nothing) =
    Float32[s; [0, 0, 0, 0]]
RLCore.feature_size(fc::OneHotFeatureCreator) = 7


end
