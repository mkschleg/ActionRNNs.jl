

mutable struct ImageReplay{R, SB, PRE, POST} <: AbstractReplay
    replay::R
    state_buffer::SB

    image_preproc::PRE
    image_postproc::POST
    function ImageReplay(replay::R,
                         state_buffer::SB,
                         image_preproc::PRE=identity,
                         image_postproc::POST=identity) where {R, SB, PRE, POST}
        new{R, SB, PRE, POST}(replay, state_buffer, image_preproc, image_postproc)
    end
end


function get_image(er::ImageReplay, s)
    er.image_postproc(er.state_buffer[s])
end

function get_raw_image(er::ImageReplay, s)
    er.state_buffer[s]
end

Base.length(er::ImageReplay) = length(er.replay)
function Base.getindex(er::ImageReplay, idx)
    experience = er.replay[idx]
    get_exp = (k, exp) -> begin
        if k == :s
            get_image(er, exp[k])
        elseif k == :sp
            get_image(er, exp[k])
        else
            exp[k]
        end
    end
    ks = keys(experience)
    nt_exp = (; zip(ks, [get_exp(k, experience) for k ∈ ks])...)
end
Base.view(er::ImageReplay, idx) = Base.view(er, idx)

function start_statebuffer!(er::ImageReplay, s)
    push!(er.state_buffer, er.image_preproc(s))
end

function Base.push!(er::ImageReplay, experience)

    s_idx = laststate(er.state_buffer)
    push!(er.state_buffer, er.image_preproc(experience.sp))
    sp_idx = laststate(er.state_buffer)
    ks = keys(experience)
    get_exp = (k, exp) -> begin
        if k == :s
            s_idx
        elseif k == :sp
            sp_idx
        else
            exp[k]
        end
    end
    nt_exp = (; zip(ks, [get_exp(k, experience) for k ∈ ks])...)
    push!(er.replay, nt_exp)
end

# lets just go ahead and deal w/ batching here.
function make_experience(er::ImageReplay{<:AbstractSequenceReplay}, expr)
    get_expr = (k, expr) -> begin
        if k == :s
            get_state(e) = get_image(er, e.s)
            Flux.batchseq([[get_state.(seq); [get_image(er, seq[end].sp)]] for seq in expr], zero(get_state(expr[1][1])))
        elseif k == :sp
            [get_raw_image(er, seq[end].sp) for seq ∈ expr]
        elseif k == :a
            l = maximum(length.(expr))
            [rpad([[seqi_j.am1[] for seqi_j ∈ seq]; [seq[end].a[]]], l, one(expr[1][1].a[])) for seq in expr]
        else
            l = maximum(length.(expr))
            [rpad([seqi_j[k][] for seqi_j ∈ seq], l, one(expr[1][1][k][])) for seq in expr]
        end
    end


    
    # do batching.
    # get_state(seq) = get_image(er, seq.s)
    # s_1 = Flux.batchseq([[get_state.(seq); [seq[end].sp]] for seq in expr], zero(expr[1][1].s))
    # a_1 = [rpad([[seqi_j.am1[] for seqi_j ∈ seq]; [seq[end].a[]]], length(s_1), 1) for seq in expr]
    # [([a_1[b][t] for b ∈ 1:length(a_1)], st) for (t, st) ∈ enumerate(s_1)]
    ks = keys(expr[1][1])
    (; zip(ks, [get_expr(k, expr) for k ∈ ks])...)
end

function make_experience(er::ImageReplay, experience)
    throw("Not Implemented!")
end

sample(er::ImageReplay, batch_size, args...) = sample(Random.GLOBAL_RNG, er, batch_size, args...)
function sample(rng::Random.AbstractRNG, er::ImageReplay, batch_size, args...)
    exp_s_idx, experience = sample(rng, er.replay, batch_size, args...)
    make_experience(er, experience)
end

