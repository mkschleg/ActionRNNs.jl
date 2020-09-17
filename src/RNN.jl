
# Sepcifying a action-conditional RNN Cell
using Flux
using OMEinsum

# Utilities for using RNNs and Action RNNs online and in chains.
dont_learn_initial_state!(rnn) = prefor(x -> x isa Flux.Recur && _dont_learn_initial_state_!(x), rnn)
dont_learn_initial_state_!(rnn) = 
    rnn.init = Flux.data(rnn.init)
function _dont_learn_initial_state!(rnn::Flux.Recur{Flux.LSTMCell})
    rnn.init = Flux.data.(rnn.init)
end

reset!(m, h_init) = 
    foreach(x -> x isa Flux.Recur && _reset!(x, h_init), Flux.functor(m)[1])

reset!(m, h_init::IdDict) = 
    foreach(x -> x isa Flux.Recur && _reset!(x, h_init[x]), Flux.functor(m)[1])

function _reset!(m::Flux.Recur, h_init)
    Flux.reset!(m)
    m.state .= h_init
end

function _reset!(m::Flux.Recur{T}, h_init) where {T<:Flux.LSTMCell}
    Flux.reset!(m)
    m.state[1] .= h_init[1]
    m.state[2] .= h_init[2]
end

function contains_rnntype(m, rnn_type::Type)
    is_rnn_type = Bool[]
    foreach(x -> push!(is_rnn_type, x isa Flux.Recur && x.cell isa rnn_type), Flux.functor(m)[1])
    return any(is_rnn_type)
end

function needs_action_input(m)
    needs_action = Bool[]
    foreach(x -> push!(needs_action, _needs_action_input(x)), Flux.functor(m)[1])
    return any(needs_action)
end

_needs_action_input(m) = false
_needs_action_input(m::Flux.Recur{T}) where {T} = _needs_action_input(m.cell)

function get_next_hidden_state(rnn::Flux.Recur{T}, h_init, input) where {T}
    return copy(rnn.cell(h_init, input)[1])
end

function get_next_hidden_state(rnn::Flux.Recur{T}, h_init, input) where {T<:Flux.LSTMCell}
    return deepcopy(rnn.cell(h_init, input)[1])
end

function get_next_hidden_state(c, h_init, input)
    reset!(c, h_init)
    c(input)
    get_hidden_state(c)
end

### TODO there may be extra copies here than what is actually needed. Test in the future if there is slowdown from allocations here.
function get_hidden_state(c)
    # h_state = IdDict{Any, Array{Float32, 1}}()
    h_state = IdDict()
    foreach(x -> x isa Flux.Recur && get!(h_state, x, get_hidden_state(x)), Flux.functor(c)[1])
    h_state
end

get_hidden_state(rnn::Flux.Recur{T}) where {T} = copy(rnn.state)
get_hidden_state(rnn::Flux.Recur{T}) where {T<:Flux.LSTMCell} = deepcopy(rnn.state)

function get_initial_hidden_state(c)
    h_state = IdDict()
    foreach(x -> x isa Flux.Recur && get!(h_state, x, get_initial_hidden_state(x)), Flux.functor(c)[1])
    h_state
end

get_initial_hidden_state(rnn::Flux.Recur{T}) where {T} = copy(rnn.init)
get_initial_hidden_state(rnn::Flux.Recur{T}) where {T<:Flux.LSTMCell} = deepcopy(rnn.init)


abstract type AbstractActionRNN end

_needs_action_input(m::M) where {M<:AbstractActionRNN} = true


"""
    ARNNCell

    An RNN cell which explicitily transforms the hidden state of the recurrent neural network according to action.

"""
mutable struct ARNNCell{F, A, V, H} <: AbstractActionRNN
    σ::F
    Wx::A
    Wh::A
    b::V
    h::H
end

# Figure for A-RNN with 3 actions.
#     O - Concatenate
#     X - Split by action
#           -----------------------------------
#          |     |--> W_1*[o_{t+1};h_t]-|      |
#          |     |                      |      |
#   h_t    |     |                      |      | h_{t+1}
# -------->|-O---X--> W_2*[o_{t+1};h_t]-X--------------->
#          | |   |                      |      |
#          | |   |                      |      |
#          | |   |--> W_3*[o_{t+1};h_t]-|      |
#           -|---------------------------------
#            | (o_{t+1}, a_t)
#            |

ARNNCell(num_ext_features, num_actions, num_hidden; init=Flux.glorot_uniform, σ_int=tanh) =
    ARNNCell(σ_int,
             init(num_hidden, num_ext_features, num_actions),
             init(num_hidden, num_hidden, num_actions),
             zeros(Float32, num_hidden, num_actions),
             Flux.zeros(num_hidden))

Flux.hidden(m::ARNNCell) = m.h
Flux.@functor ARNNCell
ARNN(args...; kwargs...) = Flux.Recur(ARNNCell(args...; kwargs...))


function _contract(W::AbstractArray{<:Number, 3}, x1::AbstractArray{<:Number, 2}, x2::AbstractArray{<:Number, 2})
    sze_W = size(W)
    Wx2 = reshape(reshape(W, :, sze_W[end])*x2, sze_W[1:2]..., :)
    @ein ret[i, l] := Wx2[i, j, l]*x1[j, l]
end

function _contract(W::AbstractArray{<:Number, 3}, x1::AbstractArray{<:Number, 1}, x2::AbstractArray{<:Number, 2})
    pdW = permutedims(W, (1, 3, 2))
    sze_W = size(pdW)
    Wx1 = reshape(reshape(pdW, :, sze_W[end])*x1, sze_W[1:2]...)
    Wx1*x2
end

_contract(W::Array{<:Number, 3}, x1::AbstractArray{<:Number, 1}, x2::AbstractArray{<:Number, 1}) =
    @ein ret[i] := W[i,j,k]*x1[j]*x2[k]


function (m::ARNNCell)(h, x::Tuple{I, A}) where {I<:Integer, A<:AbstractArray}
    # @info "Here"
    # @inbounds new_h =
    #     m.σ.((@view m.Wx[:, :, x[1]])*x[2] + (@view m.Wh[:, :, x[1]])*h + (@view m.b[:, x[1]]))
    @inbounds new_h =
        m.σ.((@view m.Wx[:, :, x[1]])*x[2] + (@view m.Wh[:, :, x[1]])*h + (@view m.b[:, x[1]]))

    return new_h, new_h
end

function (m::ARNNCell)(h, x::Tuple{TA, A}) where {TA<:AbstractArray{<:AbstractFloat, 1}, A}
    # new_h = m.σ.(m.Wx[x[1], :, :]*x[2] .+ m.Wh[x[1], :, :]*h .+ m.b[x[1], :])
    # new_h =
    #     m.σ.(mapreduce((i)->reduce_func(m, x, h, i), +, 1:(size(m.Wx)[3]))[:,1] + m.b*x[1])
    a = x[1]
    o = x[2]
    # out_x = sum(@inbounds (@view m.Wx[:,:,i])*o*a[i] +
    #             (@view m.Wh[:,:,i])*h*a[i] for i ∈ 1:length(a))
    out_x = _contract_batch(W)
    
    new_h = m.σ.(out_x + m.b*a)
    new_h, new_h
end




function (m::ARNNCell)(h, x::Tuple{TA, A}) where {TA<:AbstractArray{<:AbstractFloat, 2}, A}
    # new_h = m.σ.(m.Wx[x[1], :, :]*x[2] .+ m.Wh[x[1], :, :]*h .+ m.b[x[1], :])
    # new_h =
    #     m.σ.(mapreduce((i)->reduce_func(m, x, h, i), +, 1:(size(m.Wx)[3]))[:,1] + m.b*x[1])
    a = x[1]
    o = x[2]

    @show size(h)
    @show h isa AbstractArray{<:Number, 2}
    # out_x = _contract(m.Wx, o, a)
    # @show size(out_x)
    
    out_h = _contract(m.Wh, h, a)
    # @show size(out_h)
    # out_h = if length(size(h)) == 1
    #     hcat((_contract_batch_tensor_1(m.Wh, a, h, l) for l in 1:size(a)[2])...)
    # else
    #     hcat((_contract_batch_tensor(m.Wh, a, h, l) for l in 1:size(a)[2])...)
    # end

    new_h = m.σ.(out_h + m.b*a)
    new_h, new_h
end

function (m::ARNNCell)(h, x::Tuple{Array{<:Integer, 1}, A}) where {A}
    if length(size(h)) == 1
        @inbounds new_h = m.σ.(
            hcat((((@view m.Wx[:, :, x[1][i]])*x[2][:, i]) +
                  ((@view m.Wh[:, :, x[1][i]])*h) +
                  (@view m.b[:, x[1][i]]) for i in 1:length(x[1]))...))
        return new_h, new_h
    else
        @inbounds new_h = m.σ.(
            hcat((((@view m.Wx[:, :, x[1][i]])*x[2][:, i]) +
                  ((@view m.Wh[:, :, x[1][i]])*h[:, i]) +
                  (@view m.b[:, x[1][i]]) for i in 1:length(x[1]))...))

        return new_h, new_h
    end
end


function reduce_func(m::ARNNCell, x, h, i)
    @inbounds x[1][i]*view(m.Wx, :, :, i)*x[2] +
        x[1][i]*view(m.Wh, :, :, i)*h
end





function Base.show(io::IO, l::ARNNCell)
  print(io, "ARNNCell(", size(l.Wx, 2), ", ", size(l.Wx, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

(l::Flux.Dense)(x::T) where {T<:Tuple} = (x[2], l(x[1]))


@doc raw"""
    FactorizedARNNCell

    An RNN cell which explicitily transforms the hidden state of the recurrent neural network according to action.
    
    ```math
        W (x_{t+1}Wx \odot a_{t}Wa)^T
    ```

"""
mutable struct FacARNNCell{F, A, V, H} <: AbstractActionRNN
    σ::F
    W::A
    Wx::A
    Wh::A
    Wa::A
    b::V
    h::H
end

# FacARNNCell(num_ext_features, num_actions, num_hidden; init=Flux.glorot_uniform, σ_int=tanh) =
FacARNNCell(in, actions, out, factors, activation=tanh; init=Flux.glorot_uniform) = 
    FacARNNCell(
        activation,
        param(init(out, factors)),
        param(init(factors, in)),
        param(init(factors, out)),
        param(init(factors, actions)),
        param(Flux.zeros(out, actions)),
        param(Flux.zeros(out)))

FacARNN(args...; kwargs...) = Flux.Recur(FacARNNCell(args...; kwargs...))

function (m::FacARNNCell)(h, x::Tuple{A, O}) where {A, O}
    W = m.W; Wx = m.Wx; Wh = m.Wh; Wa = m.Wa; a = x[1]; o = x[2]; b = m.b
    new_h = m.σ.(m.W*((Wx*o .+ Wh*h) .* Wa*a) .+ b*a)
    return new_h, new_h
end

function (m::FacARNNCell)(h, x::Tuple{A, O}) where {A<:Integer, O}
    W = m.W; Wx = m.Wx; Wh = m.Wh; Wa = m.Wa; a = x[1]; o = x[2]; b = m.b
    new_h = m.σ.(W*((Wx*o + Wh*h) .* Wa[:, a]) + b[:, a])
    return new_h, new_h
end

Flux.hidden(m::FacARNNCell) = m.h


# @doc raw"""
#     FactorizedARNNCell

#     An RNN cell which explicitily transforms the hidden state of the recurrent neural network according to action.
    
#     ```math
#         W (x_{t+1}Wx \odot a_{t}Wa)^T
#     ```

# """
# mutable struct FiltARNNCell{F, A, V, H} <: AbstractActionRNN
#     σ::F
#     W::A
#     b::V
#     h::H
# end

# # FiltARNNCell(num_ext_features, num_actions, num_hidden; init=Flux.glorot_uniform, σ_int=tanh) =
# FiltARNNCell(in, actions, out, factors, out_act=tanh, filter_act=tanh; init=Flux.glorot_uniform) = 
#     FiltARNNCell(
#         activation,
#         param(init(out, )),
#         param(Flux.zeros(out)),
#         param(Flux.zeros(out)))

# FiltARNN(args...; kwargs...) = Flux.Recur(FiltARNNCell(args...; kwargs...))

# function (m::FiltARNNCell)(h, x::Tuple{A, O}) where {A, O}
#     W = m.W; Wx = m.Wx; Wh = m.Wh; Wa = m.Wa; a = x[1]; o = x[2]; b = m.b
#     new_h = m.σ.(m.W*((Wx*o .+ Wh*h) .* Wa*a) .+ b*a)
#     return new_h, new_h
# end

# function (m::FiltARNNCell)(h, x::Tuple{A, O}) where {A<:Integer, O}
#     W = m.W; Wx = m.Wx; Wh = m.Wh; Wa = m.Wa; a = x[1]; o = x[2]; b = m.b
#     new_h = m.σ.(W*((Wx*o .+ Wh*h) .* Wa[:, a]) .+ b[a])
#     return new_h, new_h
# end

# Flux.hidden(m::FiltARNNCell) = m.h

