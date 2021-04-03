

using KernelAbstractions, Tullio


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


qtargets(preds, action_t, r, γ, terminal, actual_seq_len) = begin
    @tullio q_tp1[i] := maximum(preds[actual_seq_len[i] + 1][:, i])
    trgts = (r) .+ γ * (1 .- (terminal)) .* q_tp1
    trgts
end

function get_pred_at_correct_time(preds, action, actual_seq_len, i)
    preds[actual_seq_len][action, i]
end



function update_batch!(chain,
                       opt,
                       lu::QLearning,
                       h_init,
                       state_seq,
                       reward,
                       terminal,
                       action_t,
                       actual_seq_len,
                       target_network=nothing;
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

    # one_hot_vecs = [Flux.OneHotMatrix(Flux.OneHotVector(action_t[i], num_actions), length(terminal)) for i in 1:length(state_seq)] |> gpu

    m = fill(false, length(terminal), length(terminal))
    m[CartesianIndex.(1:length(terminal), 1:length(terminal))] .= true
    m_dev = device(m)

    # return nothing
    grads = gradient(ps) do

        preds = map(chain, state_seq)
        pred_view = hcat([@view preds[actual_seq_len[i]][action_t[i], :] for i ∈ 1:length(actual_seq_len)]...)
        q_t = sum(pred_view .* m_dev; dims=2)[:, 1]

        qtrgts = typeof(q_t)()
        ignore() do
            if target_network isa Nothing
                qtrgts = device(dropgrad(qtargets(preds, action_t, reward, γ, terminal, actual_seq_len)), :qtargets)
            else
                trgt_preds = map(target_network, state_seq)
                x = qtargets(trgt_preds, action_t, reward, γ, terminal, actual_seq_len)
                qtrgts = device(x, :qtargets)
            end
        end
        loss = Flux.huber_loss(q_t, qtrgts; agg=sum)
        # loss = Flux.mse(q_t, qtrgts; agg=sum)
        # loss = sum((q_t .- qtrgts).^2)

        ignore() do
            ℒ = loss
        end
        loss
    end
    reset!(chain, h_init)
    Flux.update!(opt, ps, grads)
    UpdateState(ℒ, grads, Flux.params(chain), opt)
end


function update_batch!(model::VizBackbone,
                       opt,
                       lu::QLearning,
                       h_init,
                       state_seq,
                       reward,
                       terminal,
                       action_t,
                       actual_seq_len,
                       target_network=nothing;
                       hs_learnable=true,
                       device=CPU())

    ℒ = 0.0f0
    γ = lu.γ

    reset!(model, h_init)
    ps = get_params(model, h_init, hs_learnable)

    trgt_preds = if target_network isa Nothing
        nothing
    else
        reset!(target_network, h_init)
    end

    # one_hot_vecs = [Flux.OneHotMatrix(Flux.OneHotVector(action_t[i], num_actions), length(terminal)) for i in 1:length(state_seq)] |> gpu

    m = fill(false, length(terminal), length(terminal))
    m[CartesianIndex.(1:length(terminal), 1:length(terminal))] .= true
    m_dev = device(m)

    
    state_recon = if device isa GPU
        sm = if state_seq[1] isa Tuple
            get_mem!(()->zero(state_seq[1][2]), device, :state_recon)
        else
            get_mem!(()->zero(state_seq[1]), device, :state_recon)
        end
        for idx ∈ 1:length(terminal)
            if state_seq[1] isa Tuple
                set_batch_idx!(sm, get_batch_idx(state_seq[actual_seq_len[idx]][2], idx), idx)
            else
                set_batch_idx!(sm, get_batch_idx(state_seq[actual_seq_len[idx]], idx), idx)
            end
        end
        sm
    else
        if state_seq[1] isa Tuple
            Flux.batch([get_batch_idx(state_seq[actual_seq_len[idx]][2], idx) for idx ∈ 1:length(terminal)])
        else
            Flux.batch([get_batch_idx(state_seq[actual_seq_len[idx]], idx) for idx ∈ 1:length(terminal)])
        end
    end

    # return nothing
    grads = gradient(ps) do

        preds = map(model, state_seq)
        pred_view = hcat([@view preds[actual_seq_len[i]][action_t[i], :] for i ∈ 1:length(actual_seq_len)]...)
        q_t = sum(pred_view .* m_dev; dims=2)[:, 1]

        qtrgts = typeof(q_t)()
        ignore() do
            if target_network isa Nothing
                qtrgts = device(dropgrad(qtargets(preds, action_t, reward, γ, terminal, actual_seq_len)), :qtargets)
            else
                trgt_preds = map(target_network, state_seq)
                x = qtargets(trgt_preds, action_t, reward, γ, terminal, actual_seq_len)
                qtrgts = device(x, :qtargets)
            end
        end
        q_loss = sum((q_t .- qtrgts).^2)
        # q_loss = Flux.huber_loss(q_t, qtrgts)

        r_loss = Flux.mse(reconstruct(model, state_recon), state_recon)
        loss = q_loss + r_loss
        ignore() do
            ℒ = loss
        end
        loss
    end
    reset!(model, h_init)
    Flux.update!(opt, ps, grads)
    UpdateState(ℒ, grads, Flux.params(model), opt)
end
