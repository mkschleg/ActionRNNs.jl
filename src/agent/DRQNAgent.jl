

"""
    Basic DRQNAgent.
"""

mutable struct DRQNAgent{LU, ER, TN, O, M, F, Φ,  Π, HS<:AbstractMatrix{Float32}} <: AbstractERAgent{LU, ER, TN}
    lu::LU
    opt::O
    model::M
    target_network::TN

    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}

    hidden_state_init::Dict{Symbol, HS}
    
    replay::ER
    update_timer::UpdateTimer
    target_update_timer::UpdateTimer

    batch_size::Int
    τ::Int
    
    s_t::Φ
    π::Π
    γ::Float32
    
    action::Int
    am1::Int
    action_prob::Float64
    
    hs_learnable::Bool
    beg::Bool
    cur_step::Int

    hs_tr_init::Dict{Symbol, HS}
    device::Device
end


function DRQNAgent(model,
                   opt,
                   τ,
                   γ,
                   feature_creator,
                   feature_size,
                   env_state_size,
                   replay_size,
                   warm_up,
                   batch_size,
                   update_time,
                   target_update_time,
                   acting_policy,
                   hs_learnable)

    dev = Device(model)
    @info dev
    
    state_list, init_state = make_state_list(model, dev)

    hidden_state_init = get_initial_hidden_state(model, 1)

    hs_type, hs_length, hs_symbol = ActionRNNs.get_hs_details_for_er(model)
    replay = EpisodicSequenceReplay(replay_size+τ-1,
                                    (Int, Float32, Int, Float32, Float32, Bool, Bool, hs_type...),
                                    (1, feature_size, 1, feature_size, 1, 1, 1, hs_length...),
                                    (:am1, :s, :a, :sp, :r, :t, :beg, hs_symbol...))

    update_timer = UpdateTimer(warm_up, update_time)
    trg_update_timer = UpdateTimer(0, target_update_time)

    DRQNAgent(QLearningSUM(γ),
              opt,
              model,
              deepcopy(model),
              # nothing,
              feature_creator,
              state_list,
              hidden_state_init,
              replay,
              update_timer,
              trg_update_timer,
              batch_size,
              τ,
              init_state,
              acting_policy,
              γ,
              1, 1, 0.0, hs_learnable, true, 0,
              typeof(hidden_state_init)(), dev)
end

function ImageDRQNAgent(model,
                        opt,
                        τ,
                        γ,
                        env_state_shape,
                        env_state_type,
                        replay_size,
                        warm_up,
                        batch_size,
                        update_time,
                        target_update_time,
                        acting_policy,
                        hs_learnable)

    dev = Device(model)
    @info dev
    
    state_list, init_state = begin
        if dev isa CPU
            if needs_action_input(model)
                (DataStructures.CircularBuffer{Tuple{Int64, Array{Float32, 4}}}(2), (0, zeros(Float32, 1,1,1,1)))
            else
                (DataStructures.CircularBuffer{Array{Float32, 4}}(2), zeros(Float32, 1,1,1,1))
            end
        else
            if needs_action_input(model)
                (DataStructures.CircularBuffer{Tuple{Int64, Flux.CUDA.CuArray{Float32, 4}}}(2), (0, zeros(Float32, 1, 1, 1, 1) |> gpu))
            else
                (DataStructures.CircularBuffer{Flux.CUDA.CuArray{Float32, 4}}(2), zeros(Float32, 1, 1, 1, 1) |> gpu)
            end
        end
    end

    hidden_state_init = get_initial_hidden_state(model, 1)

    hs_type, hs_length, hs_symbol = ActionRNNs.get_hs_details_for_er(model)
    replay = EpisodicSequenceReplay(replay_size+τ-1,
                                    (Int, Int, Int, Int, Float32, Bool, Bool, hs_type...),
                                    (1, 1, 1, 1, 1, 1, 1, hs_length...),
                                    (:am1, :s, :a, :sp, :r, :t, :beg, hs_symbol...))

    sb = StateBuffer{env_state_type}(replay_size+τ*2, env_state_shape)
    
    image_replay = ImageReplay(replay, sb, identity, (img) -> Float32.(img .// 255))

    update_timer = UpdateTimer(warm_up, update_time)
    trg_update_timer = UpdateTimer(0, target_update_time)

    DRQNAgent(QLearningSUM(γ),
              opt,
              model,
              deepcopy(model),
              # nothing,
              AddDimFeatureCreator(env_state_shape),
              state_list,
              hidden_state_init,
              image_replay,
              update_timer,
              trg_update_timer,
              batch_size,
              τ,
              init_state,
              acting_policy,
              γ,
              1, 1, 0.0, hs_learnable, true, 0,
              typeof(hidden_state_init)(), dev)
end


add_exp!(agent::DRQNAgent, env_s_tp1, r, terminal, hs_symbol, hs) = begin
    push!(agent.replay,
          (am1 = agent.am1,
           s = agent.s_t isa Tuple ? agent.s_t[2] : agent.s_t,
           a = agent.action,
           sp = agent.s_t isa Tuple ? build_new_feat(agent, env_s_tp1, agent.action)[2] : build_new_feat(agent, env_s_tp1, agent.action),
           r = r,
           t = terminal,
           beg = agent.beg,
           zip(hs_symbol, hs)...))
end

add_exp!(agent::DRQNAgent{LU, ER, TN}, env_s_tp1, r, terminal, hs_symbol, hs) where {LU, ER<:ImageReplay, TN} = begin
    push!(agent.replay,
          (am1 = agent.am1,
           s = agent.s_t isa Tuple ? agent.s_t[2] : agent.s_t,
           a = agent.action,
           sp = env_s_tp1,
           r = r,
           t = terminal,
           beg = agent.beg,
           zip(hs_symbol, hs)...))
end
