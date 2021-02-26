module TMazeUtils

using ..ActionRNNs
using ..MinimalRLCore

mutable struct StandardFeatureCreator{WA} <: AbstractFeatureConstructor end

StandardFeatureCreator() = StandardFeatureCreator{true}()

(fc::StandardFeatureCreator)(s, a) = MinimalRLCore.create_features(fc, s, a)

MinimalRLCore.create_features(fc::StandardFeatureCreator, s, a) =
    Float32[s; [a==1, a==2, a==3, a==4]]
MinimalRLCore.feature_size(fc::StandardFeatureCreator{true}) = 7

MinimalRLCore.create_features(fc::StandardFeatureCreator{false}, s, a::Nothing=nothing) =
    Float32.(s)
MinimalRLCore.feature_size(fc::StandardFeatureCreator{false}) = 3

end
