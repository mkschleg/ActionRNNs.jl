module TMazeUtils

using ..ActionRNN, Reproduce
using ..RLCore

mutable struct OneHotFeatureCreator{WA} <: AbstractFeatureConstructor end

OneHotFeatureCreator() = OneHotFeatureCreator{true}()

(fc::OneHotFeatureCreator)(s, a) = RLCore.create_features(fc, s, a)
RLCore.create_features(fc::OneHotFeatureCreator{true}, s, a) =
    Float32[s; [a==1, a==2, a==3, a==4]]
RLCore.feature_size(fc::OneHotFeatureCreator{true}) = 7
RLCore.create_features(fc::OneHotFeatureCreator{true}, s, a::Nothing) = 
   Float32[s; [0, 0, 0, 0]]
RLCore.feature_size(fc::OneHotFeatureCreator{false}) = 3
RLCore.create_features(fc::OneHotFeatureCreator{false}, s, a) =
    Float32.(s)

end
