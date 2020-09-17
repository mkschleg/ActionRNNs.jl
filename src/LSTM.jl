
using Flux
using Flux: gate

function (m::Flux.LSTMCell)((h, c), x::T) where {T<:Tuple}
    m((h,c), x[2])
end

mutable struct ActionLSTMCell{A1,A2,V, V2}
    Wi_input::A1
    Wh_input::A1
    Wi::A2
    Wh::A2
    b::V
    h::V2
    c::V2
    num_actions::Int
end

function ActionLSTMCell(in::Integer, num_actions::Integer, out::Integer;
                        init = Flux.glorot_uniform)
    # Only for the input gate
    cell = ActionLSTMCell(init(out, in, 3+num_actions), init(out, out, 3+num_actions), init(out, (3 + num_actions)),
                    Flux.zeros(out), Flux.zeros(out), num_actions)
    # cell = ActionLSTMCell(init(out, in, 3+num_actions), init(out, out, 3+num_actions), init(out, (3 + num_actions)),
    #                       Flux.zeros(out), Flux.zeros(out), num_actions)
    cell.b[:, num_actions+1] .= 1
    return cell
end

# function (m::ARNNCell)(h, x::Tuple{I, A}) where {I<:Integer, A<:AbstractArray}
#     @inbounds new_h =
#         m.σ.((@view m.Wx[:, :, x[1]])*x[2] + (@view m.Wh[:, :, x[1]])*h + (@view m.b[:, x[1]]))

#     return new_h, new_h
# end


# function _contract(W::AbstractArray{<:Number, 3}, x1::AbstractArray{<:Number, 2}, x2::AbstractArray{<:Number, 2})
#     sze_W = size(W)
#     Wx2 = reshape(reshape(W, :, sze_W[end])*x2, sze_W[1:2]..., :)
#     @ein ret[i, l] := Wx2[i, j, l]*x1[j, l]
# end

function _contract(W::AbstractArray{N, 3}, x1::AbstractArray{N, 1}) where {N<:Number}
    # ret = zeros(eltype(W), size(W)[1], size(W)[3])
    # for l in 1:(size(W)[3])
    #     ret[:, l] .= (@view W[:, :, l])*x1
    # end
    # ret = Array{Array{N, 1}, 1}[]
    # for l in 1:(size(W)[3])
    #     push!(ret, (@view W[:,:,l])*x1)
    # end
    # ret
    [(@view W[:,:,l])*x1 for l in 1:size(W)[3]]
end

_contract(W::AbstractArray{N, 3}, x1::AbstractArray{N, 1}, l::Integer) where {N<:Number} = 
    (@view W[:,:,l])*x1


function get_views(m, o, a)
    idx = [1:(3*o); ((a+2)*o+1):(a+3)*o]
    ((@view m.Wi[idx, :]), (@view m.Wh[idx, :]), (@view m.b[idx]))
end

function (m::ActionLSTMCell)((h, c), ax::Tuple{I, A}) where {I<:Integer, A<:AbstractArray}
    o = size(h, 1)
    x = ax[2]
    a = ax[1]
    # Wi, Wh, b = get_views(m, o, a)
    idx = [1:3; [3+a]]
    Wi, Wh, b = (@view m.Wi[:, :, idx]), (@view m.Wh[:, :, idx]), (@view m.b[:, idx])
    # g = _contract(Wi,x) + _contract(Wh,h) + b
    # input = σ.(g[:, 4])
    # forget = σ.(g[:, 1])
    # cell = tanh.(g[:, 2])
    # output = σ.(g[:, 3])
    # g_1 = _contract(Wi,x) .+ _contract(Wh,h)
    input = σ.(_contract(Wi, x, 4) .+ _contract(Wh, h, 4) .+ b[:, 4])
    forget = σ.(_contract(Wi, x, 1) .+ _contract(Wh, h, 1) .+ b[:, 1])
    cell = tanh.(_contract(Wi, x, 2) .+ _contract(Wh, h, 2) .+ b[:, 2])
    output = σ.(_contract(Wi, x, 3) .+ _contract(Wh, h, 3) .+ b[:, 3])
    # # input = σ.(gate(g, o, a))
    # # forget = σ.(gate(g, o, 1+m.num_actions))
    # # cell = tanh.(gate(g, o, 2+m.num_actions))
    # # output = σ.(gate(g, o, 3+m.num_actions))
    c = forget .* c .+ input .* cell
    h′ = output .* tanh.(c)
    return (h′, c), h′
end

Flux.hidden(m::ActionLSTMCell) = (m.h, m.c)

Flux.@functor ActionLSTMCell

Base.show(io::IO, l::ActionLSTMCell) =
    print(io, "ActionLSTMCell(", size(l.Wi, 2), ", ", size(l.Wi, 1), ")")


"""
    ALSTM(in::Integer, out::Integer)
Long Short Term Memory recurrent layer. Behaves like an RNN but generally
exhibits a longer memory span over sequences.
See [this article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
for a good overview of the internals.
"""
ALSTM(a...; ka...) = Flux.Recur(ActionLSTMCell(a...; ka...))

