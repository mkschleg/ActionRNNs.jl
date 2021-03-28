include("buffer.jl")
include("sumtree.jl")

import Random
# import Base.getindex, Base.size
import DataStructures
import MacroTools: @forward

abstract type AbstractReplay end

Base.keys(er::AbstractReplay) = 1:length(er)

function Base.iterate(asr::AbstractReplay)

    state = 1
    result = asr[state]
    state += 1
    (result, state)
end

function Base.iterate(asr::AbstractReplay, state::Integer)

    if state > length(asr)
        return nothing
    end
    
    result = asr[state]
    state += 1
    
    (result, state)
end

mutable struct ExperienceReplay{CB<:CircularBuffer} <: AbstractReplay
    buffer::CB
end

const ExperienceReplayDef{TPL, TPS} = ExperienceReplay{CircularBuffer{TPL, TPS, (:s, :a, :sp, :r, :t)}}

ExperienceReplay(size, types, shapes, column_names) = begin
    cb = CircularBuffer(size, types, shapes, column_names)
    ExperienceReplay(cb)
end

ExperienceReplayDef(size, obs_size, obs_type) =
    ExperienceReplay(size,
                     (obs_type, Int, obs_type, Float32, Bool),
                     (obs_size, 1, obs_size, 1, 1),
                     (:s, :a, :sp, :r, :t))

Base.length(er::ExperienceReplay) = length(er.buffer)
Base.getindex(er::ExperienceReplay, idx) = er.buffer[idx]
Base.view(er::ExperienceReplay, idx) = @view er.buffer[idx]

Base.push!(er::ExperienceReplay, experience) = push!(er.buffer, experience)

sample(er::ExperienceReplay, batch_size) = sample(Random.GLOBAL_RNG, er, batch_size)

function sample(rng::Random.AbstractRNG, er::ExperienceReplay, batch_size)
    idx = rand(rng, 1:length(er), batch_size)
    return er[idx]
end


abstract type AbstractSequenceReplay <: AbstractReplay end

mutable struct SequenceReplay{CB} <: AbstractSequenceReplay
    buffer::CB
    place::Int64
end

function SequenceReplay(size, types, shapes, column_names) 
    cb = CircularBuffer(size, types, shapes, column_names)
    SequenceReplay(cb, 1)
end

Base.length(er::SequenceReplay) = length(er.buffer)
Base.getindex(er::SequenceReplay, idx) =
    er.buffer[idx]

Base.view(er::SequenceReplay, idx) =
    @view er.buffer[idx]

function Base.push!(er::SequenceReplay, experience)
    if er.buffer._full
        er.place = (er.place % capacity(er.buffer)) + 1
    end
    push!(er.buffer, experience)
end

sample(er::SequenceReplay, batch_size, seq_length) = sample(Random.GLOBAL_RNG, er, batch_size, seq_length)

function sample(rng::Random.AbstractRNG, er::SequenceReplay, batch_size, seq_length)
    start_inx = rand(rng, 1:(length(er) + 1 - seq_length), batch_size)
    e = [view(er, start_inx .+ (i-1)) for i ∈ 1:seq_length]
    start_inx, e
end

mutable struct EpisodicSequenceReplay{CB} <: AbstractSequenceReplay
    buffer::CB
    place::Int64
    terminal_locs::Vector{Int}
    terminal_symbol::Symbol
end

function EpisodicSequenceReplay(size, types, shapes, column_names; terminal_symbol = :t)
    cb = CircularBuffer(size, types, shapes, column_names)
    EpisodicSequenceReplay(cb, 1, Int[], terminal_symbol)
end


Base.length(er::EpisodicSequenceReplay) = length(er.buffer)
Base.getindex(er::EpisodicSequenceReplay, idx) =
    er.buffer[idx]

Base.view(er::EpisodicSequenceReplay, idx) =
    @view er.buffer[idx]

function Base.push!(er::EpisodicSequenceReplay, experience)
    if er.buffer._full
        er.place = (er.place % capacity(er.buffer)) + 1
    end
    push!(er.buffer, experience)
end

function get_episode_ends(er::EpisodicSequenceReplay)
    # TODO: n-computations. Maybe store in a cache?
    findall((exp)->exp::Bool, er.buffer._stg_tuple[er.terminal_symbol])
end

function get_valid_starting_range(s, e, seq_length)
    if e - seq_length <= s
        s:s
    else
        (s:e-seq_length)
    end
end


function get_valid_indicies(er::EpisodicSequenceReplay, min_seq_length)
    # episode_ends = get_episode_ends(er)

    1:length(er)
    
    # if isempty(episode_ends)
    #     1:(length(er) + 1 - min_seq_length)
    # else

    #     if er.place == 1
    #         vcat([get_valid_starting_range(
    #             e_idx == 1 ? 1 : episode_ends[e_idx-1] + 1,
    #             e,
    #             min_seq_length)
    #               for (e_idx, e) in enumerate(episode_ends)]...)
    #     else
    #         idx_set = UnitRange{Int}[]
    #         boundary_crossed = false
    #         for (e_idx, e) in enumerate(episode_ends)
    #             if e < er.place
    #                 if e_idx == 1
    #                     push!(idx_set,
    #                           get_valid_starting_range(
    #                               e_idx == 1 ? 1 : episode_ends[e_idx-1] + 1,
    #                               e,
    #                               min_seq_length))
    #                 else
    #                     push!(idx_set,
    #                           get_valid_starting_range(
    #                               e_idx == 1 ? 1 : episode_ends[e_idx-1] + 1,
    #                               e,
    #                               min_seq_length))
    #                 end
    #             elseif boundary_crossed == false
    #                 push!(idx_set,
    #                       get_valid_starting_range(
    #                           episode_ends[e_idx-1],
    #                           er.place,
    #                           min_seq_length))
    #                 push!(idx_set,
    #                       get_valid_starting_range(
    #                           er.place,
    #                           e,
    #                           min_seq_length))
    #                 boundary_crossed = true
    #             else
    #                 push!(idx_set,
    #                       get_valid_starting_range(
    #                           er.place,
    #                           e,
    #                           min_seq_length))
    #             end
    #         end
    #     end
        
    #     # If place boundary between episodes
    #     # if episode spans end and beginning of the buffer.
    #     #


    # end
end

function get_sequence(er::EpisodicSequenceReplay, start_ind, max_seq_length)
    ret = [view(er, start_ind)]
    er_size = length(er)
    if ((start_ind + 1 - 1) % er_size) + 1 == er.place || ret[end][er.terminal_symbol][]::Bool
        return ret
    end
    for i ∈ 1:(max_seq_length-1)
        push!(ret, view(er, (((start_ind + i - 1) % er_size) + 1)))
        if ret[end][er.terminal_symbol][]::Bool || ((start_ind + i + 1 - 1) % er_size) + 1 == er.place 
            break
        end
    end
    ret
end

sample(er::EpisodicSequenceReplay,
       batch_size,
       max_seq_length) =
           sample(Random.GLOBAL_RNG, er, batch_size, max_seq_length)

function sample(rng::Random.AbstractRNG,
                er::EpisodicSequenceReplay,
                batch_size,
                max_seq_length)
    # get valid starting indicies
    valid_inx = get_valid_indicies(er, 1)
    start_inx = rand(rng, valid_inx, batch_size)
    exp = [get_sequence(er, si, max_seq_length) for si ∈ start_inx]
    start_inx, exp
    # padding and batching handled by agent.
end




# mutable struct OnlineReplay{CB<:DataStructures.CircularBuffer, T<:Tuple} <: AbstractReplay
#     buffer::CB
#     column_names::T
# end

# OnlineReplay(size, types, column_names) =
#     OnlineReplay(DataStructures.CircularBuffer{Tuple{types...}}(size), tuple(column_names...))

# # size(er::OnlineReplay) = size(er.buffer)
# @forward OnlineReplay.buffer Base.lastindex
# function getindex(er::OnlineReplay, idx)
#     data = er.buffer[idx]
#     NamedTuple{er.column_names}((getindex.(data, i) for i in 1:length(er.column_names)))
# end
# isfull(er::OnlineReplay) = DataStructures.isfull(er.buffer)
# add!(er::OnlineReplay, experience) = push!(er.buffer, experience)

# function sample(er::OnlineReplay, batch_size; rng=Random.GLOBAL_RNG)
#     @assert batch_size <= size(er.buffer)[1]
#     return er[(end-batch_size+1):end]
# end

# warmup(er::OnlineReplay, x) = x

# mutable struct WeightedExperienceReplay{CB<:CircularBuffer} <: AbstractWeightedReplay
#     buffer::CB
#     sumtree::SumTree
# end

# WeightedExperienceReplay(size, types, column_names) =
#     WeightedExperienceReplay(
#         CircularBuffer(size, types, column_names),
#         SumTree{Int64}(size))

# # size(er::WeightedExperienceReplay) = size(er.buffer)
# @forward WeightedExperienceReplay.buffer getindex

# function add!(er::WeightedExperienceReplay, experience, weight)
#     idx = add!(er.buffer, experience)
#     add!(er.sumtree, weight, idx)
#     return
# end

# function sample(er::WeightedExperienceReplay, batch_size; rng=Random.GLOBAL_RNG)
#     batch_idx, batch_priorities, idx = sample(er.sumtree, batch_size; rng=rng)
#     return er.buffer[idx]
# end
