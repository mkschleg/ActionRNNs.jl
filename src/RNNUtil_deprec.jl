

########## Reset Functions ############

reset!(m, h_init::IdDict) = 
    foreach(x -> x isa Flux.Recur && reset!(x, h_init[x]), Flux.functor(m)[1])



function get_next_hidden_state_iddict(c, h_init, input)
    reset!(c, h_init)
    c(input)
    get_hidden_state_iddict(c)
end


function get_hidden_state_iddict(c)
    h_state = IdDict()
    foreach(x -> x isa Flux.Recur && get!(h_state, x, get_hidden_state(x)), Flux.functor(c)[1])
    h_state
end

function get_hidden_state_inplace_iddict(c)
    h_state = IdDict()
    foreach(x -> x isa Flux.Recur && get!(h_state, x, get_hidden_state_inplace(x)), Flux.functor(c)[1])
    h_state
end


function get_initial_hidden_state_iddict(c)
    h_state = IdDict()
    foreach(x -> x isa Flux.Recur && get!(h_state, x, get_initial_hidden_state(x)), Flux.functor(c)[1])
    h_state
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


function get_hs_from_experience_iddict(model, exp)
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
end


function modify_hs_in_er!(replay::ImageReplay, model, act_exp, exp_idx, hs::IdDict, grads=nothing, opt=nothing, device=CPU())
    
    rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, model)
    exp = act_exp[2]
    
    for ridx in rnn_idx
        if tuple_hidden_state(model[ridx])
            throw("How did you get here?")
        else
            hs_symbol = ActionRNNs.hs_symbol_layer(model[ridx], ridx)
            init_grad = zero(get_initial_hidden_state_iddict(model[ridx])) |> cpu
            init_grad_n = 0
            h = hs[model[ridx]] |> cpu

            δh = grads[hs[model[ridx]]] |> cpu
            for (i, idx) ∈ enumerate(exp_idx)
                if exp.beg[i][1] == true
                    if grads isa Nothing
                        init_grad  .+= h[:, i]
                    else
                        init_grad .+= δh[:, i]
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


function modify_hs_in_er!(replay, model, exp, exp_idx, hs::IdDict, grads = nothing, opt = nothing, device=CPU())

    if device isa GPU
        error("GPU Not yet supported for $(typeof(replay)) in modify_hs_in_er!")
    end
    rnn_idx = find_layers_with_eq((l)->l isa Flux.Recur, model)
    
    for ridx in rnn_idx
        if tuple_hidden_state(model[ridx])
            # throw("How did you get here?")
            hs_symbols = ActionRNNs.hs_symbol_layer(model[ridx], ridx)
            for (idx, hs_symbol) ∈ enumerate(hs_symbols)
                init_grad = zero(get_initial_hidden_state(model[ridx])[idx])
                init_grad_n = 0
                h = hs[hs_symbol] |> cpu

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
                    model[ridx].cell.state0 .= g
                else
                    Flux.update!(opt, model[ridx].cell.state0, g)
                end
            end
        end
    end
end
