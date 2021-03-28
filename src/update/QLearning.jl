
function q_learning_loss(q_t, a_t, r, terminal, γ, q_tp1)
    target = dropgrad(r .+ γ*(1-terminal)*maximum(q_tp1))
    return (q_t[a_t] - target).^2
end

struct QLearning <: LearningUpdate
    γ::Float32
end

function update!(chain,
                 opt,
                 lu::QLearning,
                 h_init,
                 state_seq,
                 action_t,
                 reward,
                 terminal)

    ℒ = 0.0f0
    reset!(chain, h_init)

    grads = gradient(Flux.params(chain)) do
        preds = map(chain, state_seq)
        q_tp1 = dropgrad(preds[end])
        loss = q_learning_loss(preds[end-1], action_t, reward, terminal, lu.γ, q_tp1)
        ignore() do
            ℒ = loss
        end
        loss
    end
    
    Flux.reset!(chain)
    for weights in Flux.params(chain)
        if !(grads[weights] === nothing) && !(weights isa Flux.Zeros)
            Flux.update!(opt, weights, grads[weights])
        end
    end

    UpdateState(ℒ, grads, Flux.params(chain), opt)
end

q_learning_loss_batch_single(q_t_i, r, γ, term, q_tp1_i) = 
    (q_t_i - dropgrad((r + γ*(1-term)*maximum(q_tp1_i))))^2

qloss(preds, action_t, reward, γ, terminal, actual_seq_len) = begin
    s = 0.0f0
    for i ∈ 1:length(actual_seq_len)
        s += q_learning_loss_batch_single(
            preds[actual_seq_len[i]][action_t[i], i],
            reward[i],
            γ,
            terminal[i],
            preds[actual_seq_len[i] + 1][:, i])
    end
    s
end

function update_batch!(chain,
                       opt,
                       lu::QLearning,
                       h_init,
                       state_seq,
                       reward,
                       terminal,
                       action_t,
                       actual_seq_len;
                       hs_learnable=true)

    ℒ = 0.0f0
    γ = lu.γ
    reset!(chain, h_init)
    ps = get_params(chain, h_init, hs_learnable)
    
    grads = gradient(ps) do
        preds = map(chain, state_seq)
        loss = qloss(preds, action_t, reward, γ, terminal, actual_seq_len)
        ignore() do
            ℒ = loss
        end
        loss
    end
    reset!(chain, h_init)
    Flux.update!(opt, ps, grads)
    UpdateState(ℒ, grads, Flux.params(chain), opt)
end
