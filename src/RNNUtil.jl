
using Flux

######## Information Functions ############

trainable(a::Flux.Recur) = (a.cell, a.state)

tuple_hidden_state(rnn::Flux.Recur) = rnn.state isa Tuple

"""
    find_layers_with_eq(m, eq)

A function which takes a model and a function and returns the locations where the function returns true.
"""
function find_layers_with_eq(eq::Function, model)
    m = model
    layer_indecies = Union{Int, Tuple}[]
    for (idx, l) in enumerate(m)
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

find_layers_with_recur(model) = find_layers_with_recur(model) do (l)
    l isa Flux.Recur
end

function contains_comp(comp::Function, model)
    is_true = false
    foreach(x -> is_true = is_true || comp(x),
            Flux.functor(model)[1])
    return is_true
end

contains_layer_type(model, type) = contains_comp(model) do (l)
    (l isa Flux.Recur && l.cell isa type) || (l isa type)
end

contains_rnntype(m, rnn_type) = contains_layer_type(m, rnn_type)

needs_action_input(m) = contains_comp(m) do (l)
    _needs_action_input(l)
end


###### Symbols #######

hs_symbol_layer(l, idx) = if tuple_hidden_state(l)
    throw("You Shouldn't Be here")
else
    Symbol("hs_$(idx)")
end

function get_hs_symbol_list(model)
    hs_symbol = Symbol[]
    rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, model)
    for idx ∈ rnn_idx
        if tuple_hidden_state(model[idx])
            throw("LSTMs not supported yet")
        else
            push!(hs_symbol, hs_symbol_layer(model[idx], idx))
        end
    end
    hs_symbol
end



########## Reset Functions ############

# reset!(m, h_init) = 
#     foreach(x -> x isa Flux.Recur && reset!(x, h_init), Flux.functor(m)[1])

# function Base.getproperty(m::Flux.Recur, sym::Symbol)
#     getfield(m, sym)
#   # if sym === :init
#   #   Zygote.ignore() do
#   #     @warn "Recur field :init has been deprecated. To access initial state weights, use m::Recur.cell.state0 instead."
#   #   end
#   #   return getfield(m.cell, :state0)
#   # else
#   #   return getfield(m, sym)
#   # end
# end

# Base.setproperty!(m::Flux.Recur, sym::Symbol, x) = begin
#     @show "Here"
    
# end

reset!(m, h_init::IdDict) = 
    foreach(x -> x isa Flux.Recur && reset!(x, h_init[x]), Flux.functor(m)[1])

function reset!(m, h_init::Dict)
    rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, m)
    for idx ∈ rnn_idx
        if tuple_hidden_state(m[idx])
            throw("Not implemented yet.")
        else
            h_sym = hs_symbol_layer(m[idx], idx)
            reset!(m[idx], h_init[h_sym])
        end
    end
end

# Can't do inplace, because it will overwrite init if you ever call reset!(m)
function reset!(m::Flux.Recur, h_init)
    # Flux.reset!(m)
    (m.state = h_init)
    # @show m.state === h_init
end

function reset!(m::Flux.Recur{T}, h_init) where {T<:Flux.LSTMCell}
    # Flux.reset!(m)
    (m.state = h_init)
end


############ Hidden State Manipulation ##########

function get_next_hidden_state(rnn::Flux.Recur{T}, h_init, input) where {T}
    return copy(rnn.cell(h_init, input)[1])
end

function get_next_hidden_state(rnn::Flux.Recur{T}, h_init, input) where {T<:Flux.LSTMCell}
    return deepcopy(rnn.cell(h_init, input)[1])
end

function get_next_hidden_state(c, h_init, input, d=nothing)
    reset!(c, h_init)
    c(input)
    get_hidden_state(c, d)
end

### TODO there may be extra copies here than what is actually needed. Test in the future if there is slowdown from allocations here.
function get_hidden_state(c, d=nothing)
    # h_state = IdDict{Any, Array{Float32, 1}}()
    if d isa Nothing
        h_state = IdDict()
        foreach(x -> x isa Flux.Recur && get!(h_state, x, get_hidden_state(x)), Flux.functor(c)[1])
        h_state
    else
        h_state = Dict{Symbol, AbstractMatrix{Float32}}()
        rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, c)
        for idx ∈ rnn_idx
            if tuple_hidden_state(c[idx])
                throw("Not implemented yet....")
            else
                h_sym = hs_symbol_layer(c[idx], idx)
                h_state[h_sym] = get_hidden_state(c[idx])
            end
        end
        h_state
    end
end

get_hidden_state(rnn::Flux.Recur{T}) where {T} = copy(rnn.state)
get_hidden_state(rnn::Flux.Recur{T}) where {T<:Flux.LSTMCell} = deepcopy(rnn.state)


function get_hidden_state_inplace(c, d=nothing)
    # h_state = IdDict{Any, Array{Float32, 1}}()
    if d isa Nothing
        h_state = IdDict()
        foreach(x -> x isa Flux.Recur && get!(h_state, x, get_hidden_state_inplace(x)), Flux.functor(c)[1])
        h_state
    else
        h_state = Dict{Symbol, AbstractMatrix{Float32}}()
        rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, c)
        for idx ∈ rnn_idx
            if tuple_hidden_state(c[idx])
                throw("Not implemented yet....")
            else
                h_sym = hs_symbol_layer(c[idx], idx)
                h_state[h_sym] = get_hidden_state_inplace(c[idx])
            end
        end
        h_state
    end
end


get_hidden_state_inplace(rnn::Flux.Recur{T}) where {T} = rnn.state
get_hidden_state_inplace(rnn::Flux.Recur{T}) where {T<:Flux.LSTMCell} = rnn.state


function get_initial_hidden_state(c, d=nothing)
    if d isa Nothing
        h_state = IdDict()
        foreach(x -> x isa Flux.Recur && get!(h_state, x, get_initial_hidden_state(x)), Flux.functor(c)[1])
        h_state
    else
        
        h_state = Dict{Symbol, AbstractMatrix{Float32}}()
        rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, c)
        for idx ∈ rnn_idx
            if tuple_hidden_state(c[idx])
                throw("Not implemented yet....")
            else
                h_sym = hs_symbol_layer(c[idx], idx)
                h_state[h_sym] = get_initial_hidden_state(c[idx])
            end
        end
        h_state
    end
end

get_initial_hidden_state(rnn::Flux.Recur{T}) where {T} = rnn.cell.state0
get_initial_hidden_state(rnn::Flux.Recur{T}) where {T<:Flux.LSTMCell} = rnn.cell.state0



###### Experience Replay Functions ########

function get_hs_details_for_er(model)

    # Find all the RNN layers in the model
    rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, model)

    # get hidden state archetype from model
    hidden_state_init = get_initial_hidden_state(model, 1)
    
    hs_type = DataType[]
    hs_length = Int[]
    hs_symbol = Symbol[]
    for idx ∈ rnn_idx
        hs_sym = hs_symbol_layer(model[idx], idx)
        if hidden_state_init[hs_sym] isa Tuple
            # push!(hs_type, hidden_state_init[model[idx]])
            throw("LSTMs not supported yet")
        else
            hs_archtype = hidden_state_init[hs_sym]
            push!(hs_type, eltype(hs_archtype))
            push!(hs_length, length(hs_archtype))
            push!(hs_symbol, hs_symbol_layer(model[idx], idx))
        end
    end

    hs_type, hs_length, hs_symbol
end


function get_hs_from_experience!(model, exp::NamedTuple, d::IdDict, device)

    hs = d

    rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, model)
    for idx in rnn_idx
        init_hs = get_initial_hidden_state(model[idx])
        if tuple_hidden_state(model[idx])
            throw("How did you get here?")
        else
            hs_symbol = hs_symbol_layer(model[idx], idx)
            if :beg ∈ keys(exp)
                hs[model[idx]] .= device(hcat([(exp.beg[b][1] ? init_hs : exp[hs_symbol][b][1]) for b in 1:length(exp.beg)]...))
            else
                hs[model[idx]] .= device(hcat([(exp[hs_symbol][b][1]) for b in 1:length(exp.beg)]...))
            end
        end
    end
    hs

end

function get_hs_from_experience!(model, exp::NamedTuple, d::Dict, device)

    hs = d

    rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, model)
    for idx in rnn_idx
        init_hs = CPU()(get_initial_hidden_state(model[idx]))
        if tuple_hidden_state(model[idx])
            throw("How did you get here?")
        else
            hs_symbol = hs_symbol_layer(model[idx], idx)
            h = if :beg ∈ keys(exp)
                # @show typeof(exp.beg), typeof(exp[hs_symbol]), typeof(init_hs)
                device(hcat([(exp.beg[b][1] ? init_hs : exp[hs_symbol][b][1]) for b in 1:length(exp.beg)]...), hs_symbol)
            else
                device(hcat([(exp[hs_symbol][b][1]) for b in 1:length(exp.beg)]...), hs_symbol)
            end
            int_h = get!(()->h, hs, hs_symbol)
            copyto!(int_h, h)
        end
    end
    hs

end

function get_hs_from_experience!(model, exp::Vector, d::Dict, device)

    hs = d

    rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, model)
    for idx in rnn_idx
        init_hs = CPU()(get_initial_hidden_state(model[idx]))
        if tuple_hidden_state(model[idx])
            throw("How did you get here?")
        else
            hs_symbol = hs_symbol_layer(model[idx], idx)
            init_hs = get_initial_hidden_state(model[idx])
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

function get_hs_from_experience(model, exp, d=nothing)
    if d isa Nothing
        hs = IdDict()
        rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, model)
        for idx in rnn_idx
            if tuple_hidden_state(model[idx])
                throw("How did you get here?")
            else
                hs_symbol = hs_symbol_layer(model[idx], idx)
                init_hs = get_initial_hidden_state(model[idx])
                if :beg ∈ keys(exp[1][1])
                    hs[model[idx]] = hcat([(seq[1].beg[1] ? init_hs : getindex(seq[1], hs_symbol)) for seq in exp]...)
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
    else
        hs = Dict{Symbol, Array{Float32}}()
        rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, model)
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
end



function modify_hs_in_er!(replay::ImageReplay, model, act_exp, exp_idx, hs, grads=nothing, opt=nothing, device=CPU())
    
    rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, model)
    exp = act_exp[2]
    
    for ridx in rnn_idx
        if tuple_hidden_state(model[ridx])
            throw("How did you get here?")
        else
            hs_symbol = ActionRNNs.hs_symbol_layer(model[ridx], ridx)
            init_grad = zero(get_initial_hidden_state(model[ridx])) |> cpu
            init_grad_n = 0
            h = if hs isa IdDict
                hs[model[ridx]] |> cpu
            else
                hs[hs_symbol] |> cpu
            end
            for (i, idx) ∈ enumerate(exp_idx)

                if exp.beg[i][1] == true
                    if grads isa Nothing
                        init_grad  .+= h[:, i]
                    else
                        init_grad .+= grads[h][:, i]
                    end
                    # init_grad .+= h[:, i]
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


function modify_hs_in_er!(replay, model, exp, exp_idx, hs, grads = nothing, opt = nothing)
    
    rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, model)
    
    for ridx in rnn_idx
        if tuple_hidden_state(model[ridx])
            throw("How did you get here?")
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


function modify_hs_in_er!(replay::SequenceReplay, model, exp, exp_idx, hs, grads = nothing, opt = nothing)
    
    # rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, model)
    
    # for ridx in rnn_idx
    #     if tuple_hidden_state(model[ridx])
    #         throw("How did you get here?")
    #     else
    #         hs_symbol = ActionRNNs.hs_symbol_layer(model[ridx], ridx)
    #         init_grad = zero(get_initial_hidden_state(model[ridx]))
    #         init_grad_n = 0
    #         h = if hs isa IdDict
    #             hs[model[ridx]]
    #         else
    #             hs[hs_symbol]
    #         end
    #         for (i, idx) ∈ enumerate(exp_idx)
    #             if exp[i][1].beg[] == true
    #                 if grads isa Nothing
    #                     init_grad  .+= h[:, i]
    #                 else
    #                     init_grad .+= grads[h][:, i]
    #                 end
    #                 init_grad_n += 1
    #             else
    #                 if getindex(replay.buffer._stg_tuple, hs_symbol) isa Vector
    #                     getindex(replay.buffer._stg_tuple, hs_symbol)[idx] = h[i]
    #                 else
    #                     getindex(replay.buffer._stg_tuple, hs_symbol)[:, idx] .= h[:, i]
    #                 end
    #             end
    #         end
    #         if init_grad_n != 0
    #             g = init_grad ./ init_grad_n
    #             if grads isa Nothing
    #                 model[ridx].cell.state0 .= g
    #             else
    #                 Flux.update!(opt, model[ridx].cell.state0, g)
    #             end
    #         end
    #     end
    # end
end


