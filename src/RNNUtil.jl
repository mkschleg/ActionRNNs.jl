
using Flux

######## Information Functions ############

trainable(a::Flux.Recur) = (a.cell, a.state)

tuple_hidden_state(rnn::Flux.Recur) = rnn.state isa Tuple

"""
    find_layers_with_eq(eq::Function, model)

A function which takes a model and a function and returns the locations where the function returns true.
This only supports composing chaings twice.
"""
function find_layers_with_eq(eq::Function, model)
    layer_indecies = Union{Int, Tuple}[]
    for (idx, l) in enumerate(model)
        if l isa Flux.Chain
            layer_idx = find_layers_with_eq(eq, l)
            for l_idx in layer_idx
                push!(layer_indecies, (idx, l_idx))
            end
        elseif eq(l)
            push!(layer_indecies, idx)
        end
    end
    return layer_indecies
end

"""
    find_layers_with_recur(model)

Finds layers with recur. Uses [`find_layers_with_eq`](@ref).
"""
find_layers_with_recur(model) = find_layers_with_eq(model) do (l)
    l isa Flux.Recur
end

"""
    contains_comp(comp::Function, model)

Check if a layer of a model returns true with comp.
"""
function contains_comp(comp::Function, model)
    is_true = false
    foreach(x -> is_true = is_true || comp(x),
            Flux.functor(model)[1])
    return is_true
end

"""
    contains_layer_type(model, type)

Check if the model has a specific layer type.
"""
contains_layer_type(model, type) = contains_comp(model) do (l)
     (l isa type) || (l isa Flux.Recur && l.cell isa type)
end

"""
    contains_rnn_type(m, rnn_type)

Checks if the model has a specific rnn type.
"""
contains_rnn_type(m, rnn_type) = begin
    if rnn_type <: Flux.Recur
        contains_layer_type(m, rnn_type.parameters[1])
    else
        contains_layer_type(m, rnn_type)
    end
end

"""
    needs_action_input(m)

Checks if the model needs action input as a tuple.
"""
needs_action_input(m) = contains_comp(m) do (l)
    _needs_action_input(l)
end


###### Symbols #######

"""
    hs_symbol_layer(l, idx)

Get symbol of current layer's hidden state layer.
"""
hs_symbol_layer(l, idx) = if tuple_hidden_state(l)
    Symbol("hs_h_$(idx)"), Symbol("hs_c_$(idx)")
else
    Symbol("hs_$(idx)")
end

"""
    get_hs_symbol_list(model)

Get list of hidden state symbols for all rnn layers.
"""
function get_hs_symbol_list(model)
    hs_symbol = Symbol[]
    rnn_idx = find_layers_with_recur(model) # find_layers_with_eq((l)->l isa Flux.Recur, model)
    for idx ∈ rnn_idx
        if tuple_hidden_state(model[idx])
            hs = hs_symbol_layer(model[idx], idx)
            push!(hs_symbol, hs[1])
            push!(hs_symbol, hs[2])
        else
            push!(hs_symbol, hs_symbol_layer(model[idx], idx))
        end
    end
    hs_symbol
end



########## Reset Functions ############


"""
    reset!(m, h_init::Dict)
    reset!(m::Flux.Recur, h_init)

Reset the hidden state according to the dict h_init with keys from [`get_hs_symbol_list`](@ref). If model
is a recur just replace the hidden state. 
"""
function reset!(m, h_init::Dict)
    rnn_idx = find_layers_with_recur(m) # find_layers_with_eq((l)->l isa Flux.Recur, m)
    for idx ∈ rnn_idx
        if tuple_hidden_state(m[idx])
            h_sym = hs_symbol_layer(m[idx], idx)
            reset!(m[idx], (h_init[h_sym[1]], h_init[h_sym[2]]))
        else
            h_sym = hs_symbol_layer(m[idx], idx)
            reset!(m[idx], h_init[h_sym])
        end
    end
end

# Can't do inplace, because it will overwrite init if you ever call reset!(m)
function reset!(m::Flux.Recur, h_init)
    (m.state = h_init)
end

############ Hidden State Manipulation ##########

function get_next_hidden_state!(rnn::Flux.Recur{T}, h_init, input) where {T}
    if tuple_hidden_state(rnn)
        deepcopy(rnn.cell(h_init, input)[1])
    else
        copy(rnn.cell(h_init, input)[1])
    end
end

function get_next_hidden_state!(model, h_init, input)
    reset!(model, h_init)
    model(input)
    get_hidden_state(model)
end


function _get_hidden_state(ghs_func::Function, model)
    
    h_state = Dict{Symbol, AbstractMatrix{Float32}}()
    rnn_idx = find_layers_with_recur(model) # find_layers_with_eq((l)->l isa Flux.Recur, c)
    for idx ∈ rnn_idx
        if tuple_hidden_state(model[idx])
            hs_sym = hs_symbol_layer(model[idx], idx)
            h_state[hs_sym[1]], h_state[hs_sym[2]] = ghs_func(model[idx])
        else
            h_sym = hs_symbol_layer(model[idx], idx)
            h_state[h_sym] = ghs_func(model[idx])
        end
    end
    h_state
end

_get_hidden_state(ghs_func::Function, rnn::Flux.Recur) = ghs_func(rnn)

get_hidden_state(model) = _get_hidden_state(model) do rnn
    if tuple_hidden_state(rnn)
        copy(rnn.state[1]), copy(rnn.state[2])
    else
        copy(rnn.state)
    end
end

get_hidden_state_inplace(model) = _get_hidden_state(model) do rnn
    rnn.state
end

get_initial_hidden_state(model) = _get_hidden_state(model) do rnn
    rnn.cell.state0
end

#################
#
# Experience Replay Functions
#
#################

"""
    get_hs_details_for_er(model)

Return the types, sizes, and symbols of the hidden state for the ER buffer.
"""
function get_hs_details_for_er(model)

    # Find all the RNN layers in the model
    rnn_idx = find_layers_with_recur(model) # find_layers_with_eq((l)->l isa Flux.Recur, model)
    hidden_state_init = get_initial_hidden_state(model)
    
    
    hs_type = DataType[]
    hs_length = Int[]
    hs_symbol = Symbol[]
    
    for idx ∈ rnn_idx
        hs_sym = hs_symbol_layer(model[idx], idx)
        if tuple_hidden_state(model[idx])
            hs_archtype_1, hs_archtype_2 = hidden_state_init[hs_sym[1]], hidden_state_init[hs_sym[2]]

            push!(hs_type, eltype(hs_archtype_1))
            push!(hs_length, length(hs_archtype_1))
            push!(hs_symbol, hs_symbol_layer(model[idx], idx)[1])

            push!(hs_type, eltype(hs_archtype_2))
            push!(hs_length, length(hs_archtype_2))
            push!(hs_symbol, hs_symbol_layer(model[idx], idx)[2])
        else
            hs_archtype = hidden_state_init[hs_sym]
            push!(hs_type, eltype(hs_archtype))
            push!(hs_length, length(hs_archtype))
            push!(hs_symbol, hs_symbol_layer(model[idx], idx))
        end
    end

    hs_type, hs_length, hs_symbol
end

"""
    get_hs_from_experience!(model, exp::NamedTuple, hs_dict::Dict, device)
    get_hs_from_experience!(model, exp::Vector, hs_dict::Dict, device)

Get hs in the appropriate formate from the experience (either a Named Tuple or a vector of tuples Named Tuples).
"""
function get_hs_from_experience!(model, exp::NamedTuple, hs_dict::Dict, device)

    hs = hs_dict

    rnn_idx = find_layers_with_recur(model) # find_layers_with_eq((l)->l isa Flux.Recur, model)
    for idx in rnn_idx

        init_hs = CPU()(get_initial_hidden_state(model[idx]))
        if tuple_hidden_state(model[idx])

            hs_symbol = hs_symbol_layer(model[idx], idx)
            h_1 = if :beg ∈ keys(exp)
                device(hcat([(exp.beg[b][1] ? init_hs[1] : exp[hs_symbol[1]][b][1]) for b in 1:length(exp.beg)]...), hs_symbol[1])
            else
                device(hcat([(exp[hs_symbol[1]][b][1]) for b in 1:length(exp.beg)]...), hs_symbol[1])
            end

            h_2 = if :beg ∈ keys(exp)
                device(hcat([(exp.beg[b][1] ? init_hs[1] : exp[hs_symbol[2]][b][1]) for b in 1:length(exp.beg)]...), hs_symbol[2])
            else
                device(hcat([(exp[hs_symbol[2]][b][1]) for b in 1:length(exp.beg)]...), hs_symbol[2])
            end

            int_h_1 = get!(()->h_1, hs, hs_symbol[1])
            int_h_2 = get!(()->h_2, hs, hs_symbol[2])

            copyto!(int_h_1, h_1)
            copyto!(int_h_2, h_2)
        else
            hs_symbol = hs_symbol_layer(model[idx], idx)
            h = if :beg ∈ keys(exp)
                # @show typeof(exp.beg), typeof(exp[hs_symbol]), typeof(init_hs)
                hcat([(exp.beg[b][1] ? init_hs : exp[hs_symbol][b][1]) for b in 1:length(exp.beg)]...)
            else
                hcat([(exp[hs_symbol][b][1]) for b in 1:length(exp.beg)]...)
            end
            int_h = get!(()->device(h, hs_symbol), hs, hs_symbol)
            copyto!(int_h, h)
        end
    end
    hs

end

function get_hs_from_experience!(model, exp::Vector, hs_dict::Dict, device)

    hs = hs_dict

    rnn_idx = find_layers_with_recur(model) # find_layers_with_eq((l)->l isa Flux.Recur, model)
    for idx in rnn_idx
        init_hs = CPU()(get_initial_hidden_state(model[idx]))
        if tuple_hidden_state(model[idx])
            # throw("How did you get here?")
            
            hs_symbol = hs_symbol_layer(model[idx], idx)
            # init_hs = get_initial_hidden_state(model[idx])
            h_1 = if :beg ∈ keys(exp[1][1])
                hcat([(seq[1].beg[1] ? init_hs[1] : getindex(seq[1], hs_symbol[1])) for seq in exp]...)
            else
                if exp[1] isa NamedTuple
                    getindex(exp[1], hs_symbol[1])
                else
                    hcat([(getindex(seq[1], hs_symbol[1])) for seq in exp]...)
                end
            end
            int_h_1 = get!(()->h_1, hs, hs_symbol[1])
            copyto!(int_h_1, h_1)

                        
            h_2 = if :beg ∈ keys(exp[1][1])
                hcat([(seq[1].beg[1] ? init_hs[2] : getindex(seq[1], hs_symbol[2])) for seq in exp]...)
            else
                if exp[1] isa NamedTuple
                    getindex(exp[1], hs_symbol[2])
                else
                    hcat([(getindex(seq[1], hs_symbol[2])) for seq in exp]...)
                end
            end
            int_h_2 = get!(()->h_2, hs, hs_symbol[2])
            copyto!(int_h_2, h_2)
        else
            hs_symbol = hs_symbol_layer(model[idx], idx)
            # init_hs = get_initial_hidden_state(model[idx])
            h = if :beg ∈ keys(exp[1][1])
                hcat([(seq[1].beg[1] ? init_hs : getindex(seq[1], hs_symbol)) for seq in exp]...)
            else
                if exp[1] isa NamedTuple
                    getindex(exp[1], hs_symbol)
                else
                    hcat([(getindex(seq[1], hs_symbol)) for seq in exp]...)
                end
            end
            int_h = get!(()->h, hs, hs_symbol)
            copyto!(int_h, h)
        end
    end
    hs
end

function get_hs_from_experience(model, exp)
    hs = Dict{Symbol, Array{Float32}}()
    rnn_idx = find_layers_with_recur(model) # find_layers_with_eq((l)->l isa Flux.Recur, model)
    for idx in rnn_idx
        if tuple_hidden_state(model[idx])
            throw("How did you get here?")
        else
            hs_symbol = hs_symbol_layer(model[idx], idx)
            init_hs = get_initial_hidden_state(model[idx])
            if :beg ∈ keys(exp[1][1])
                hs[hs_symbol] = hcat([(seq[1].beg[1] ? init_hs : getindex(seq[1], hs_symbol)) for seq in exp]...)
            else
                if exp[1] isa NamedTuple
                    hs[hs_symbol] = getindex(exp[1], hs_symbol)
                else
                    hs[hs_symbol] = hcat([(getindex(seq[1], hs_symbol)) for seq in exp]...)
                end
            end
        end
    end
    hs
end

"""
    modify_hs_in_er!

Updating hidden state in the experience replay buffer.
"""
function modify_hs_in_er!(replay::ImageReplay, model, act_exp, exp_idx, hs, grads=nothing, opt=nothing, device=CPU())
    
    rnn_idx = find_layers_with_recur(model) # find_layers_with_eq((l)->l isa Flux.Recur, model)
    exp = act_exp[2]
    
    for ridx in rnn_idx
        if tuple_hidden_state(model[ridx])
            throw("How did you get here?")
        else
            hs_symbol = ActionRNNs.hs_symbol_layer(model[ridx], ridx)
            
            init_grad = zero(get_initial_hidden_state(model[ridx])) |> cpu
            init_grad_n = 0
            
            h = hs[hs_symbol] |> cpu
            δh = grads[hs[hs_symbol]] |> cpu

            for (i, idx) ∈ enumerate(exp_idx)

                if exp.beg[i][1] == true
                    if grads isa Nothing
                        init_grad  .+= h[:, i]
                    else
                        init_grad .+= δh[:, i]
                    end
                    init_grad_n += 1
                else
                    if getindex(replay.replay.buffer._stg_tuple, hs_symbol) isa Vector
                        getindex(replay.replay.buffer._stg_tuple, hs_symbol)[idx] = h[i]
                    else
                        getindex(replay.replay.buffer._stg_tuple, hs_symbol)[:, idx] .= h[:, i]
                    end
                end
            end
            if init_grad_n != 0
                g = init_grad ./ init_grad_n
                if grads isa Nothing
                    model[ridx].cell.state0 .= device(g)
                else
                    Flux.update!(opt, model[ridx].cell.state0, device(g))
                end
            end
        end
    end
end


function modify_hs_in_er!(replay, model, exp, exp_idx, hs, grads = nothing, opt = nothing, device=CPU())

    if device isa GPU
        error("GPU Not yet supported for $(typeof(replay)) in modify_hs_in_er!")
    end
    
    rnn_idx = find_layers_with_recur(model) # find_layers_with_eq((l)->l isa Flux.Recur, model)
    
    for ridx in rnn_idx
        if tuple_hidden_state(model[ridx])
            hs_symbols = ActionRNNs.hs_symbol_layer(model[ridx], ridx)
            for (idx, hs_symbol) ∈ enumerate(hs_symbols)
                init_grad = zero(get_initial_hidden_state(model[ridx])[idx])
                init_grad_n = 0
                h = hs[hs_symbol]

                for (i, idx) ∈ enumerate(exp_idx)
                    if exp[i][1].beg[] == true
                        if grads isa Nothing
                            init_grad  .+= h[:, i]
                        else
                            init_grad .+= grads[h][:, i]
                        end
                        init_grad_n += 1
                    else
                        if getindex(replay.buffer._stg_tuple, hs_symbol) isa Vector
                            getindex(replay.buffer._stg_tuple, hs_symbol)[idx] = h[i]
                        else
                            getindex(replay.buffer._stg_tuple, hs_symbol)[:, idx] .= h[:, i]
                        end
                    end
                end
                if init_grad_n != 0
                    g = init_grad ./ init_grad_n
                    if grads isa Nothing
                        model[ridx].cell.state0[idx] .= g
                    else
                        Flux.update!(opt, model[ridx].cell.state0[idx], g)
                    end
                end
            end
        else
            hs_symbol = ActionRNNs.hs_symbol_layer(model[ridx], ridx)
            init_grad = zero(get_initial_hidden_state(model[ridx]))
            init_grad_n = 0
            h = if hs isa IdDict
                hs[model[ridx]]
            else
                hs[hs_symbol]
            end
            for (i, idx) ∈ enumerate(exp_idx)
                if exp[i][1].beg[] == true
                    if grads isa Nothing
                        init_grad  .+= h[:, i]
                    else
                        init_grad .+= grads[h][:, i]
                    end
                    init_grad_n += 1
                else
                    if getindex(replay.buffer._stg_tuple, hs_symbol) isa Vector
                        getindex(replay.buffer._stg_tuple, hs_symbol)[idx] = h[i]
                    else
                        getindex(replay.buffer._stg_tuple, hs_symbol)[:, idx] .= h[:, i]
                    end
                end
            end
            if init_grad_n != 0
                g = init_grad ./ init_grad_n
                if grads isa Nothing
                    model[ridx].cell.state0 .= g
                else
                    Flux.update!(opt, model[ridx].cell.state0, g)
                end
            end
        end
    end
end

function modify_hs_in_er!(::SequenceReplay, args...)
    throw("Please don't be here.")
end

include("RNNUtil_deprec.jl")

