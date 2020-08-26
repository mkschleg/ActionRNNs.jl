
# Sepcifying a action-conditional RNN Cell
using Flux

# Utilities for using RNNs and Action RNNs online and in chains.

dont_learn_initial_state!(rnn) = prefor(x -> x isa Flux.Recur && _dont_learn_initial_state_!(x), rnn)
dont_learn_initial_state_!(rnn) = 
    rnn.init = Flux.data(rnn.init)
function _dont_learn_initial_state!(rnn::Flux.Recur{Flux.LSTMCell})
    rnn.init = Flux.data.(rnn.init)
end

reset!(m, h_init) = 
    Flux.prefor(x -> x isa Flux.Recur && _reset!(x, h_init), m)

reset!(m, h_init::IdDict) = 
    Flux.prefor(x -> x isa Flux.Recur && _reset!(x, h_init[x]), m)

function _reset!(m, h_init)
    Flux.reset!(m)
    m.state.data .= Flux.data(h_init)
end

function _reset!(m::Flux.Recur{T}, h_init) where {T<:Flux.LSTMCell}
    Flux.reset!(m)
    m.state[1].data .= Flux.data(h_init[1])
    m.state[2].data .= Flux.data(h_init[2])
end

function contains_rnntype(m, rnn_type::Type)
    is_rnn_type = Bool[]
    Flux.prefor(x -> push!(is_rnn_type, x isa Flux.Recur && x.cell isa rnn_type), m)
    return any(is_rnn_type)
end

function needs_action_input(m)
    needs_action = Bool[]
    Flux.prefor(x -> push!(needs_action, _needs_action_input(x)), m)
    return any(needs_action)
end

_needs_action_input(m) = false
_needs_action_input(m::Flux.Recur{T}) where {T} = _needs_action_input(m.cell)

function get_next_hidden_state(rnn::Flux.Recur{T}, h_init, input) where {T}
    return Flux.data(rnn.cell(h_init, input)[1])
end

function get_next_hidden_state(rnn::Flux.Recur{T}, h_init, input) where {T<:Flux.LSTMCell}
    return Flux.data.(rnn.cell(h_init, input)[1])
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
    Flux.prefor(x -> x isa Flux.Recur && get!(h_state, x, get_hidden_state(x)), c)
    h_state
end

get_hidden_state(rnn::Flux.Recur{T}) where {T} = copy(Flux.data(rnn.state))
get_hidden_state(rnn::Flux.Recur{T}) where {T<:Flux.LSTMCell} = copy.(Flux.data.(rnn.state))

function get_initial_hidden_state(c)
    h_state = IdDict()
    Flux.prefor(x -> x isa Flux.Recur && get!(h_state, x, get_initial_hidden_state(x)), c)
    h_state
end

get_initial_hidden_state(rnn::Flux.Recur{T}) where {T} = copy(Flux.data(rnn.init))
get_initial_hidden_state(rnn::Flux.Recur{T}) where {T<:Flux.LSTMCell} = deepcopy(Flux.data.(rnn.init))


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
    ARNNCell(
        σ_int,
        param(init(num_actions, num_hidden, num_ext_features)),
        param(init(num_actions, num_hidden, num_hidden)),
        param(zeros(Float32, num_actions, num_hidden)),
        # cumulants,
        # discounts,
        param(Flux.zeros(num_hidden)))


function (m::ARNNCell)(h, x::Tuple{I, A}) where {I<:Integer, A<:AbstractArray}
    @inbounds new_h =
        m.σ.(m.Wx[x[1], :, :]*x[2] + m.Wh[x[1], :, :]*h + m.b[x[1], :])
    return new_h, new_h
end

function (m::ARNNCell)(h, x::Tuple{TA, A}) where {TA<:AbstractArray{<:AbstractFloat, 1}, A}
    # new_h = m.σ.(m.Wx[x[1], :, :]*x[2] .+ m.Wh[x[1], :, :]*h .+ m.b[x[1], :])
    new_h =
        m.σ.(mapreduce((i)->reduce_func(m, x, h, i), +, 1:(size(m.Wx)[1]))[:,1] + m.b'x[1])

    new_h, new_h
end

function (m::ARNNCell)(h, x::Tuple{Array{<:Integer, 1}, A}) where {A}
    if length(size(h)) == 1
        new_h = m.σ.(
            cat(collect((m.Wx[x[1][i], :, :]*x[2][:, i]) for i in 1:length(x[1]))...; dims=2) .+
            cat(collect((m.Wh[x[1][i], :, :]*h) for i in 1:length(x[1]))...; dims=2) .+
            m.b[x[1], :]')
        return new_h, new_h
    else
        new_h = m.σ.(
            cat(collect((m.Wx[x[1][i], :, :]*x[2][:, i]) for i in 1:length(x[1]))...; dims=2) .+
            cat(collect((m.Wh[x[1][i], :, :]*h[:, i]) for i in 1:length(x[1]))...; dims=2) .+
            m.b[x[1], :]')
        return new_h, new_h
    end
end


function reduce_func(m::ARNNCell, x, h, i)
    @inbounds x[1][i]*view(m.Wx, i, :, :)*x[2] +
        x[1][i]*view(m.Wh, i, :, :)*h
        # x[1][i]*view(m.b, i, :)
end



Flux.hidden(m::ARNNCell) = m.h
Flux.@treelike ARNNCell
ARNN(args...; kwargs...) = Flux.Recur(ARNNCell(args...; kwargs...))


function Base.show(io::IO, l::ARNNCell)
  print(io, "ARNNCell(", size(l.Wx, 2), ", ", size(l.Wx, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

(l::Flux.Dense)(x::T) where {T<:Tuple} = (x[2], l(x[1]))


#    `\Lambda \mathbf{W} \left(x_{t+1}\mathbf{Wx} \odot a_{t}\mathbf{Wa}\right)^\top`

@doc raw"""
    FactorizedARNNCell

    An RNN cell which explicitily transforms the hidden state of the recurrent neural network according to action.

    
    ``W (x_{t+1}Wx \odot a_{t}Wa)^T``

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

# Figure for A-RNN with 3 actions.
#     O - Concatenate
#     X - Split by action
#           -----------------------------------
#          |                                   |
#          |                                   |
#   h_t    |                                   | h_{t+1}
# -------->|-O--- 
#          | |                                 |
#          | |                                 |
#          | |                                 |
#           -|---------------------------------
#            | (o_{t+1}, a_t)
#            |


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
    new_h = m.σ.(m.W*((Wx*o .+ Wh*h) .* Wa[:, a]) .+ b[a])
    return new_h, new_h
end

Flux.hidden(m::FacARNNCell) = m.h
