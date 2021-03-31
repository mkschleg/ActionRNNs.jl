

struct VizBackbone{VB, VB2, O}
    encoder::VB
    decoder::VB2
    outnetwork::O
end

Flux.@functor VizBackbone

(m::VizBackbone)(x) = x |> m.encoder |> dropgrad |> m.outnetwork
reconstruct(m::VizBackbone, x) = x |> m.encoder |> m.decoder
Base.length(m::VizBackbone) = length(m.encoder) + length(m.decoder) + length(m.outnetwork)

function Base.getindex(m::VizBackbone, idx::Int)
    if idx <= length(m.encoder)
        m.decoder[idx]
    elseif idx - length(m.encoder) <= length(m.decoder)
        m.decoder[idx - length(m.encoder)]
    else 
        m.outnetwork[idx - length(m.encoder) - length(m.decoder)]
    end
end

Base.iterate(m::VizBackbone, state=1) = state > length(m) ? nothing : (m[state], state+1)

needs_action_input(m::VizBackbone) = needs_action_input(m.outnetwork)

is_gpu(model::VizBackbone) = begin
    ig = false
    foreach(x -> ig = ig || is_gpu(x),
            Flux.functor(model)[1])
    ig
end
