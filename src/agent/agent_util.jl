mutable struct UpdateTimer
    warm_up::Int
    update_wait::Int
    t::Int
    UpdateTimer(warm_up, update_wait) = new(warm_up, update_wait, 0)
end

(ut::UpdateTimer)(replay_len) =
    replay_len >= ut.warm_up && (ut.t % ut.update_wait == 0)

step!(ut::UpdateTimer) = ut.t += 1
reset!(ut::UpdateTimer) = ut.t = 0


get_replay_buffer(agent::AbstractERAgent) = agent.replay
get_learning_update(agent::AbstractERAgent) = agent.lu


####
# Construction helper functions
####


function make_state_list(model, dev)
    state_list, init_state = begin
        if dev isa CPU
            if needs_action_input(model)
                (DataStructures.CircularBuffer{Tuple{Int64, Array{Float32, 1}}}(2), (0, zeros(Float32, 1)))
            else
                (DataStructures.CircularBuffer{Array{Float32, 1}}(2), zeros(Float32, 1))
            end
        else
            if needs_action_input(model)
                (DataStructures.CircularBuffer{Tuple{Int64, Flux.CUDA.CuArray{Float32, 4}}}(2), (0, zeros(Float32, 1) |> gpu))
            else
                (DataStructures.CircularBuffer{Flux.CUDA.CuArray{Float32, 4}}(2), zeros(Float32, 1) |> gpu)
            end
        end
    end
end

function make_replay(model, feature_size, feature_type=Float32)
    hs_type, hs_length, hs_symbol = ActionRNNs.get_hs_details_for_er(model)
    replay = EpisodicSequenceReplay(replay_size+τ-1,
                                    (Int, feature_type, Int, feature_type, Float32, Bool, Bool, hs_type...),
                                    (1, feature_size, 1, feature_size, 1, 1, 1, hs_length...),
                                    (:am1, :s, :a, :sp, :r, :t, :beg, hs_symbol...))
end



####
# Get info from experience
####

get_state_from_experience(agent::AbstractERAgent, exp) =
    get_state_from_experience(agent.state_list[1], exp)

function get_state_from_experience(::Tuple, exp)
    get_state(seq) = seq.s
    s_1 = Flux.batchseq([[get_state.(seq); [seq[end].sp]] for seq in exp], zero(exp[1][1].s))
    a_1 = [rpad([[seqi_j.am1[] for seqi_j ∈ seq]; [seq[end].a[]]], length(s_1), 1) for seq in exp]
    [([a_1[b][t] for b ∈ 1:length(a_1)], st) for (t, st) ∈ enumerate(s_1)]
end

function get_state_from_experience(type, exp)
    Flux.batchseq([[getindex.(exp[i], :s); [exp[i][end].sp]] for i in 1:batch_size], zero(exp[1][1].s))
end

get_information_from_experience(agent::AbstractERAgent, exp) = 
    get_information_from_experience(get_replay_buffer(agent), get_learning_update(agent), agent.s_t, exp)

function get_information_from_experience(::EpisodicSequenceReplay, ::ControlUpdate, s_t, exp)
    s = get_state_from_experience(s_t, exp)

    batch_size = length(exp)
    
    t = [exp[i][end].t[1] for i in 1:batch_size]
    r = [exp[i][end].r[1] for i in 1:batch_size]
    a = [exp[i][end].a[1] for i in 1:batch_size]

    actual_seq_lengths = [length(exp[i]) for i in 1:batch_size]
    
    s, a, r, t, actual_seq_lengths
end

function get_information_from_experience(::ImageReplay{ER}, ::ControlUpdate, s_t, actllen_exp) where {ER<:EpisodicSequenceReplay}
    # s = get_state_from_experience(s_t, exp)

    exp = actllen_exp[2]
    s = if s_t isa Tuple
        [(exp.am1[i], exp.s[i]) for i in 1:length(exp.s)]
    else
        device.(exp.s, [Symbol("s$(i)") for i ∈ 1:length(exp.s)])
    end

    t = [exp.t[i][end] for i in 1:length(exp.t)]
    r = [exp.r[i][end] for i in 1:length(exp.t)]
    a = [exp.a[i][end] for i in 1:length(exp.t)]

    actual_seq_lengths = length.(exp.t)
    
    s, a, r, t, actual_seq_lengths
end


function get_information_from_experience(::EpisodicSequenceReplay, ::PredictionUpdate, s_t, exp)


    s = get_state_from_experience(s_t, exp)

    
    batch_size = length(exp)

    t = [exp[i][end].t[1] for i in 1:batch_size]
    r = [exp[i][end].r[1] for i in 1:batch_size]
    a = [exp[i][end].a[1] for i in 1:batch_size]
    bprob = [exp[i][end].ap[1] for i in 1:batch_size]
    sp1 = hcat([exp[i][end].esp for i in 1:batch_size]...)

    actual_seq_lengths = [length(exp[i]) for i in 1:batch_size]
    
    s, sp1, a, r, t, bprob, actual_seq_lengths
end

function get_information_from_experience(::SequenceReplay, ::PredictionUpdate, s_t, exp)

    state_list = if s_t isa Tuple
        sl = [(collect(exp[i].am1), collect(exp[i].s)) for i in 1:length(exp)]
        push!(sl, (collect(exp[end].a), collect(exp[end].sp)))
    else
        sl = [exp[i].s for i in 1:length(exp)]
        push!(sl, exp[end].sp)
    end
    
    batch_size = length(exp)

    sp1 = collect(exp[end].esp')
    bprob = exp[end].ap
    
    t = exp[end].t
    r = exp[end].r
    a = exp[end].a

    actual_seq_lengths = [length(exp[i]) for i in 1:batch_size]
    
    state_list, sp1, a, r, t, bprob, actual_seq_lengths
end

####
# Build features for the model
####

build_new_feat(agent::AbstractERAgent, state, action) = begin
    if eltype(agent.state_list) <: Tuple
        (action, agent.build_features(state, action))
    else
        agent.build_features(state, action)
    end
end

####
# Building State Lists
####

mutable struct IdentityFeatureCreator{ES} <: AbstractFeatureConstructor
    env_state_shape::ES
end

MinimalRLCore.create_features(fc::IdentityFeatureCreator, s, a) = s
MinimalRLCore.feature_size(fc::IdentityFeatureCreator) = fc.env_state_size

(fc::IdentityFeatureCreator)(s, a) = s

mutable struct AddDimFeatureCreator{ES} <: AbstractFeatureConstructor
    env_state_shape::ES
end

MinimalRLCore.create_features(fc::AddDimFeatureCreator, s, a) = reshape(s, fc.env_state_shape..., 1)
MinimalRLCore.feature_size(fc::AddDimFeatureCreator) = (fc.env_state_shape..., 1)

(fc::AddDimFeatureCreator)(s, a) = MinimalRLCore.create_features(fc, s, a)

