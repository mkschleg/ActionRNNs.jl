
using Flux

# Utilities for using RNNs and Action RNNs online and in chains.

# _dont_learn_initial_state!(rnn) = 
#     rnn.init = Flux.data(rnn.init)

# function _dont_learn_initial_state!(rnn::Flux.Recur{Flux.LSTMCell})
#     rnn.init = Flux.data.(rnn.init)
# end

# dont_learn_initial_state!(rnn) = foreach(x -> x isa Flux.Recur && _dont_learn_initial_state!(x), rnn)

reset!(m, h_init) = 
    foreach(x -> x isa Flux.Recur && _reset!(x, h_init), Flux.functor(m)[1])

reset!(m, h_init::IdDict) = 
    foreach(x -> x isa Flux.Recur && _reset!(x, h_init[x]), Flux.functor(m)[1])

# Can't do inplace, because it will overwrite init if you ever call reset!(m)
function _reset!(m::Flux.Recur, h_init)
    Flux.reset!(m)
    m.state = h_init
end

function _reset!(m::Flux.Recur{T}, h_init) where {T<:Flux.LSTMCell}
    Flux.reset!(m)
    m.state = h_init
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

get_initial_hidden_state(rnn::Flux.Recur{T}) where {T} = rnn.init
get_initial_hidden_state(rnn::Flux.Recur{T}) where {T<:Flux.LSTMCell} = rnn.init


