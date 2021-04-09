module TMazeUtils

using ..ActionRNNs
using ..MinimalRLCore

mutable struct StandardFeatureCreator{WA} <: AbstractFeatureConstructor end

StandardFeatureCreator() = StandardFeatureCreator{true}()

(fc::StandardFeatureCreator)(s, a) = MinimalRLCore.create_features(fc, s, a)

MinimalRLCore.create_features(fc::StandardFeatureCreator{true}, s, a) =
    Float32[s; [a==1, a==2, a==3, a==4]]

MinimalRLCore.create_features(fc::StandardFeatureCreator{false}, s, a) = Float32.(s)


MinimalRLCore.feature_size(fc::StandardFeatureCreator{true}) = 7


MinimalRLCore.feature_size(fc::StandardFeatureCreator{false}) = 3

end
