


"""
    HSStale
"""
struct HSStale end

"""
    HSMinimize
"""
struct HSMinimize end

"""
    HSRefil
"""
struct HSRefil end




modify_hs_in_er!(hs_strategy::Bool, args...; kwargs...) = if hs_strategy
    modify_hs_in_er_by_grad!(args...; kwargs...)
end

modify_hs_in_er!(hs_strategy::HSMinimize, args...; kwargs...) = 
    modify_hs_in_er_by_grad!(args...; kwargs...)

modify_hs_in_er!(hs_strategy::HSStale, args...; kwargs...) = nothing
   


"""
    modify_hs_in_er!

Updating hidden state in the experience replay buffer.
"""
function modify_hs_in_er_by_grad!(replay::ImageReplay, model, act_exp, exp_idx, hs, grads=nothing, opt=nothing, device=CPU())
    
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



function modify_hs_in_er_by_grad!(replay, model, exp, exp_idx, hs, grads = nothing, opt = nothing, device=CPU())

    if device isa GPU
        error("GPU Not yet supported for $(typeof(replay)) in modify_hs_in_er_by_grad!")
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

function modify_hs_in_er_by_grad!(::SequenceReplay, args...)
    throw("Please don't be here.")
end

