
tderror(v_t, c, γ_tp1, ṽ_tp1) =
    (v_t .- (c .+ γ_tp1.*ṽ_tp1))


function offpolicy_tdloss(ρ_t, v_t, c, γ_tp1, ṽ_tp1)
    target = dropgrad(Float32.(c + γ_tp1.*ṽ_tp1))

    sum(ρ_t.*((v_t - target).^2)) * (1//(2*size(ρ_t)[1]))
end

function offpolicy_tdloss_batch(ρ_t, v_t, c, γ_tp1, ṽ_tp1)
    # println(size(ρ_t), size(v_t), size(c), size(γ_tp1), size(ṽ_tp1))
    target = dropgrad(Float32.(c + γ_tp1.*ṽ_tp1))
    sum(ρ_t.*((v_t - target).^2)) * (1//(2*size(ρ_t)[1]))
end


struct TD <: PredictionUpdate end

function update!(chain,
                 horde::H,
                 opt,
                 lu::TD,
                 h_init,
                 state_seq,
                 env_state_tp1,
                 action_t=nothing,
                 b_prob=1.0;
                 kwargs...) where {T, H<:GVFHordes.AbstractHorde}

    reset!(chain, h_init)
    n = length(state_seq)
    grads = gradient(Flux.params(chain)) do

        preds = map(chain, state_seq)
        v_tp1 = dropgrad(preds[n])
        cumulants, discounts, π_prob = dropgrad(get(horde, nothing, action_t, env_state_tp1, v_tp1))
        ρ = dropgrad(Float32.(π_prob./b_prob))
        offpolicy_tdloss(ρ, preds[n-1], cumulants, discounts, v_tp1)
        
    end
    reset!(chain, h_init)
    for weights in Flux.params(chain)
        if !(grads[weights] === nothing)
            Flux.update!(opt, weights, grads[weights])
        end
    end
end


function update_batch!(lu::TD,
                       chain,
                       horde::H,
                       opt,
                       h_init,
                       (state_seq,
                        env_state_tp1,
                        action_t,
                        r,
                        t,
                        b_prob,
                        actual_seq_lengths);
                       hs_learnable=true,
                       kwargs...) where {H<:GVFHordes.AbstractHorde}

    n = length(state_seq)
    ℒ = 0.0f0
    
    reset!(chain, h_init)
    ps = get_params(chain, h_init, hs_learnable)

    grads = gradient(ps) do

        preds = map(chain, state_seq)
        v_tp1 = dropgrad(preds[n])

        ρ = zeros(Float32, size(v_tp1)...)
        cumulants = zeros(Float32, size(v_tp1)...)
        discounts = zeros(Float32, size(v_tp1)...)

        ignore() do
            params = if length(size(v_tp1)) == 1
                [get(horde, nothing, action_t[1], env_state_tp1, v_tp1)]
            else
                [get(horde, nothing, action_t[i], env_state_tp1[:, i], v_tp1[:, i]) for i in 1:(size(v_tp1)[2])]
            end
            
            cumulants .= Flux.batch(Base.getindex.(params, 1))
            discounts .= Flux.batch(Base.getindex.(params, 2))
            ρ .= if length(size(v_tp1)) == 1 
                Flux.batch([Base.getindex(params[1], 3)./b_prob[1]])
            else
                Flux.batch([Base.getindex(params[i], 3)./b_prob[i] for i in 1:(size(v_tp1)[2])])
            end
        end
        
        loss = offpolicy_tdloss_batch(ρ, preds[n-1], cumulants, discounts, v_tp1)
        ignore() do
            ℒ = loss
        end
        loss
    end
    reset!(chain, h_init)
    Flux.update!(opt, ps, grads)
    UpdateState(ℒ, grads, Flux.params(chain), opt)
end


#old update batch
function update_batch!(chain,
                       horde::H,
                       opt,
                       lu::TD,
                       h_init,
                       state_seq,
                       env_state_tp1,
                       action_t=nothing,
                       b_prob=1.0;
                       kwargs...) where {H<:GVFHordes.AbstractHorde}

    reset!(chain, h_init)

    n = length(state_seq)
    preds = map(chain, state_seq)
    v_tp1 = preds[n]

    params = if length(size(v_tp1)) == 1
        dropgrad([get(horde, nothing, action_t[1], env_state_tp1, v_tp1)])
    elseif length(size(env_state_tp1)) == 1
        dropgrad([get(horde, nothing, action_t[i], env_state_tp1[i], v_tp1[:, i]) for i in 1:(size(env_state_tp1)[2])])
    else
        dropgrad([get(horde, nothing, action_t[i], env_state_tp1[:, i], v_tp1[:, i]) for i in 1:(size(env_state_tp1)[2])])
    end

    cumulants = dropgrad(Flux.batch(Base.getindex.(params, 1)))
    discounts = dropgrad(Flux.batch(Base.getindex.(params, 2)))
    ρ = if length(size(env_state_tp1)) == 1
        dropgrad(Flux.batch([Base.getindex(params[1], 3)./b_prob[1]]))
    else
        dropgrad(Flux.batch([Base.getindex(params[i], 3)./b_prob[i] for i in 1:(size(env_state_tp1)[2])]))
    end
    
    ℒ = 0.0f0
    ps = Flux.params(chain)
    reset!(chain, h_init)
    grads = gradient(ps) do
        
        preds = map(chain, state_seq)
        v_tp1 = dropgrad(preds[n])

        loss = offpolicy_tdloss(ρ, preds[n-1], cumulants, discounts, v_tp1)
        ignore() do
            ℒ = loss
        end
        loss
    end
    reset!(chain, h_init)
    for weights in ps
        if !(grads[weights] === nothing)
            Flux.update!(opt, weights, grads[weights])
        end
    end
    UpdateState(ℒ, grads, Flux.params(chain), opt)
end
