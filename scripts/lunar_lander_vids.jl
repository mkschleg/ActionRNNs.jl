
import Pkg
Pkg.activate(@__DIR__)

using Distributed
using Reproduce, ArgParse

@everwhere include(joinpath(@__DIR__, "modules/LunarLanderVids.jl"))
# @everwhere using .LunarLanderVids

function save_ll_video(item)
    imgs = LunarLanderVids.get_video_from_item(item, 200)
    FileIO.save(joinpath(item.folder_str, "images.jld2"), "imgs", imgs)
end

function save_ll_videos(dir::AbstractString)
    ic = ItemCollection(dir)
    pmap(save_ll_video, ic.items)
end

s = ArgParseSettings()
@add_arg_table s begin
    "dir"
        help = "Directory of the lunar lander results to use to create videos."
        required = true
end

parsed_args = parse_args(ARGS, s)
save_ll_videos(parsed_args["dir"])

# using ActionRNNs, Reproduce, Colors, ProgressLogging


# function render_img_from_env(env)
#     arr = env.gym.pyenv.render("rgb_array")./255
#     img = fill(Colors.RGB{Colors.N0f8}(0,0,0), 400, 600)
#     for i in 1:400, j in 1:600
# 	img[i, j] = Colors.RGB{Colors.N0f8}(arr[i, j, :]...)
#     end
#     img
# end

# function split_actions(res)
#     actions = Vector{Int}[]
#     stp_cnt = 1
#     for i in 1:length(res["total_steps"])
#         # @info stp_cnt
#         push!(actions, res["Agent_action"][stp_cnt:(stp_cnt + res["total_steps"][i])])
#         stp_cnt += (res["total_steps"][i]) + 1
#     end
#     actions
# end



# function get_video_from_item(item::Reproduce.Item, num_eps_for_vids)

#     env = LunarLander(item.parsed_args["seed"])
#     res = FileIO.load(joinpath(item.folder_str, "results.jld2"))
    
#     @progress "Warm-up" for eps in 1:(length(actions)-num_eps_for_vids)
#         is_terminal = false
#         MinimalRLCore.start!(env)
#         rew = 0.0f0
#         i = 1
#         while !is_terminal
#             MinimalRLCore.step!(env, actions[eps][i])
#             is_terminal = MinimalRLCore.is_terminal(env)
#             rew += MinimalRLCore.get_reward(env)
#             i += 1
#         end
#         if i - 1 != res["total_steps"][eps]
#             @warn "Episode Ends Premeturly!!!"
#         end
#     end

#     eps_imgs = []

#     @progress "Get Img Seqs" for eps in (length(actions)- num_eps_for_vids + 1):length(actions)
#         imgs = []
#         is_terminal = false
#         MinimalRLCore.start!(env)
#         push!(imgs, render_img_from_env(env))
#         i = 1
#         while !is_terminal
#             MinimalRLCore.step!(env, actions[eps][i])
#             is_terminal = MinimalRLCore.is_terminal(env)
#             push!(imgs, render_img_from_env(env))
#             i += 1
#         end
#         if i - 1 != res["total_steps"][eps]
#             @warn "Episode Ends Premeturly!!!"
#         end
#         push!(eps_imgs, imgs)
#     end

#     eps_imgs
    
# end


# function save_ll_videos(ic::Reproduce.ItemCollection, num_eps_for_vids=100)
#     for item in ic.items
#         imgs = get_video_from_item(item, num_eps_for_vids)
#         FileIO.save(joinpath(item.folder_str, "images.jld2"), "imgs", imgs)
#     end
# end

# save_ll_videos(dir, args...) = save_ll_videos(ItemCollection(dir), args...)




