
using KernelAbstractions, Tullio


function q_learning_loss(q_t, a_t, r, terminal, γ, q_tp1)
    target = dropgrad(r .+ γ*(1-terminal)*maximum(q_tp1))
    return (q_t[a_t] - target).^2
end

struct RStepQLearning{F} <: ControlUpdate
    γ::Float32
    loss::F
end

QLearningMSE(γ) = QLearning(γ, Flux.mse)
QLearningSUM(γ) = QLearning(γ, (ŷ, y)->Flux.mse(ŷ, y; agg=sum))
QLearningHUBER(γ) = QLearning(γ, (ŷ, y)->Flux.huber_loss(ŷ, y; agg=sum))

function update!(chain,
                 opt,
                 lu::RStepQLearning,
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

qtargets(preds, action_t, r, γ, terminal, actual_seq_len) = begin
    @tullio q_tp1[i] := maximum(preds[actual_seq_len[i] + 1][:, i])
    (r) .+ γ * (1 .- (terminal)) .* q_tp1
end

function update_batch!(lu::RStepQLearning,
                       chain,
                       target_network,
                       opt,
                       h_init,
                       (state_seq,
                        action_t,
                        reward,
                        terminal,
                        actual_seq_len);
                       hs_learnable=true,
                       device=CPU())

    ℒ = 0.0f0
    γ = lu.γ

    reset!(chain, h_init)
    ps = get_params(chain, h_init, hs_learnable)

    trgt_preds = if target_network isa Nothing
        nothing
    else
        reset!(target_network, h_init)
    end

    m = fill(false, length(terminal), length(terminal))
    m[CartesianIndex.(1:length(terminal), 1:length(terminal))] .= true
    m_dev = device(m)
    
    grads = gradient(ps) do

        preds = map(chain, state_seq)
        pred_view = hcat([preds[actual_seq_len[i]][action_t[i], :] for i ∈ 1:length(actual_seq_len)]...)
        q_t = sum(pred_view .* m_dev; dims=2)[:, 1]

        qtrgts = typeof(q_t)()
        ignore() do
            if target_network isa Nothing
                for i ∈ 1:length(reward)
                    # qtrgts = device(dropgrad(qtargets(preds, action_t, reward, γ, terminal, actual_seq_len)), :qtargets)
                end
            else
                trgt_preds = map(target_network, state_seq)
                v_tp1 = trg_preds[end]
                maximum(preds[actual_seq_len[i] + 1][:, i])
                for i ∈ length(reward):-1:1
                    γ * max(v_tp1)
                end
                x = qtargets(trgt_preds, action_t, reward, γ, terminal, actual_seq_len)
                qtrgts = device(x, :qtargets)
            end
        end

        loss = lu.loss(q_t, qtrgts)
        
        ignore() do
            ℒ = loss
        end
        loss
    end

    Flux.update!(opt, ps, grads)
    UpdateState(ℒ, grads, Flux.params(chain), opt)
end



