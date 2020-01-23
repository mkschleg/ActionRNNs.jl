module FluxWidgets

abstract type AbstractFluxWidget end

mutable struct EEG{T<:AbstractFloat} <: AbstractFluxWidget
    activations::Array{Array{Array{T}, 1}, 1}
end

EEG(type::Type{T}) where {T<:AbstractFloat}  = EEG(Array{Array{Array{T}, 1}, 1}())

function update!(eeg::EEG, model, input)
    push!(eeg.activations, DeepRL.FluxUtils.get_activations(mapleaves(Flux.data, model), input))
end


mutable struct WidgetChain
    widgets::Array{AbstractFluxWidget}
end



end
