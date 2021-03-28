
using Flux

# Utilities for using RNNs and Action RNNs online and in chains.

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

contains_rnntype(m, rnn_type) = contains_layer_type(m, rnn_type)

function contains_layer_type(m::Flux.Chain, type::Type)
    is_type = false
    foreach(x -> is_type = is_type || (x isa Flux.Recur && x.cell isa type) || (x isa type),
            Flux.functor(m)[1])
    return is_type
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

function find_layers_with_eq(m, eq)
    layer_indecies = Union{Int, Tuple}[]
    for (idx, l) in enumerate(m)
        if l isa Flux.Chain
            layer_idx = find_layers_with_eq(l, eq)
            for l_idx in layer_idx
                push!(layer_indecies, (idx, l_idx))
            end
        elseif eq(l)
            push!(layer_indecies, idx)
        end
    end
    return layer_indecies
end

tuple_hidden_state(rnn::Flux.Recur) = Flux.hidden(rnn.cell) isa Tuple


hs_symbol_layer(l, idx) = if tuple_hidden_state(l)
    throw("You Shouldn't Be here")
else
    Symbol("hs_$(idx)")
end

function get_hs_details_for_er(model)

    # Find all the RNN layers in the model
    rnn_idx = find_layers_with_eq(model, (l)->l isa Flux.Recur)

    # get hidden state archetype from model
    hidden_state_init = get_initial_hidden_state(model)
    
    hs_type = DataType[]
    hs_length = Int[]
    hs_symbol = Symbol[]
    for idx ∈ rnn_idx
        if hidden_state_init[model[idx]] isa Tuple
            # push!(hs_type, hidden_state_init[model[idx]])
            throw("LSTMs not supported yet")
        else
            hs_archtype = hidden_state_init[model[idx]]
            push!(hs_type, eltype(hs_archtype))
            push!(hs_length, length(hs_archtype))
            push!(hs_symbol, hs_symbol_layer(model[idx], idx))
        end
    end

    hs_type, hs_length, hs_symbol
end


function get_hs_from_experience(model, exp::NamedTuple, device=CPU())
    hs = IdDict()

    rnn_idx = find_layers_with_eq(model, (l)->l isa Flux.Recur)
    for idx in rnn_idx
        if tuple_hidden_state(model[idx])
            throw("How did you get here?")
        else
            hs_symbol = hs_symbol_layer(model[idx], idx)
            if :beg ∈ keys(exp)
                hs[model[idx]] = device(hcat([(exp.beg[b][1] ? model[idx].init : exp[hs_symbol][b][1]) for b in 1:length(exp.beg)]...))
            else
            end
        end
    end
    hs

end

function get_hs_from_experience(model, exp)
    hs = IdDict()
    rnn_idx = find_layers_with_eq(model, (l)->l isa Flux.Recur)
    for idx in rnn_idx
        if tuple_hidden_state(model[idx])
            throw("How did you get here?")
        else
            hs_symbol = hs_symbol_layer(model[idx], idx)
            if :beg ∈ keys(exp[1][1])
                hs[model[idx]] = hcat([(seq[1].beg[1] ? model[idx].init : getindex(seq[1], hs_symbol)) for seq in exp]...)
            else
                if exp[1] isa NamedTuple
                    hs[model[idx]] = getindex(exp[1], hs_symbol)
                else
                    hs[model[idx]] = hcat([(getindex(seq[1], hs_symbol)) for seq in exp]...)
                end
            end
        end
    end
    hs
end


function modify_hs_in_er!(replay::ImageReplay, model, exp, exp_idx, hs)
    
    rnn_idx = find_layers_with_eq(model, (l)->l isa Flux.Recur)
    
    for ridx in rnn_idx
        if tuple_hidden_state(model[ridx])
            throw("How did you get here?")
        else
            hs_symbol = ActionRNNs.hs_symbol_layer(model[ridx], ridx)
            init_grad = zero(model[ridx].init)
            init_grad_n = 0
            for (i, idx) ∈ enumerate(exp_idx)
                if exp.beg[i][1] == true
                    init_grad  .+= hs[model[ridx]][:, i]
                    init_grad_n += 1
                else
                    if getindex(replay.replay.buffer._stg_tuple, hs_symbol) isa Vector
                        getindex(replay.replay.buffer._stg_tuple, hs_symbol)[idx] = hs[model[ridx]][i]
                    else
                        getindex(replay.replay.buffer._stg_tuple, hs_symbol)[:, idx] .= hs[model[ridx]][:, i]
                    end
                end
            end
            if init_grad_n != 0
                model[ridx].init .= init_grad ./ init_grad_n
            end
        end
    end
end


function modify_hs_in_er!(replay, model, exp, exp_idx, hs)
    
    rnn_idx = find_layers_with_eq(model, (l)->l isa Flux.Recur)
    
    for ridx in rnn_idx
        if tuple_hidden_state(model[ridx])
            throw("How did you get here?")
        else
            hs_symbol = ActionRNNs.hs_symbol_layer(model[ridx], ridx)
            init_grad = zero(model[ridx].init)
            init_grad_n = 0
            for (i, idx) ∈ enumerate(exp_idx)
                if exp[i][1].beg == true
                    init_grad  .+= hs[model[ridx]][:, i]
                    init_grad_n += 1
                else
                    if getindex(replay.buffer._stg_tuple, hs_symbol) isa Vector
                        getindex(replay.buffer._stg_tuple, hs_symbol)[idx] = hs[model[ridx]][i]
                    else
                        getindex(replay.buffer._stg_tuple, hs_symbol)[:, idx] .= hs[model[ridx]][:, i]
                    end
                end
            end
            if init_grad_n != 0
                model[ridx].init .= init_grad ./ init_grad_n
            end
        end
    end
end



