
module LunarLanderVids

using ActionRNNs, Reproduce, Colors, ProgressLogging


function render_img_from_env(env)
    arr = env.gym.pyenv.render("rgb_array")./255
    img = fill(Colors.RGB{Colors.N0f8}(0,0,0), 400, 600)
    for i in 1:400, j in 1:600
	img[i, j] = Colors.RGB{Colors.N0f8}(arr[i, j, :]...)
    end
    img
end

function split_actions(res)
    actions = Vector{Int}[]
    stp_cnt = 1
    for i in 1:length(res["total_steps"])
        # @info stp_cnt
        push!(actions, res["Agent_action"][stp_cnt:(stp_cnt + res["total_steps"][i])])
        stp_cnt += (res["total_steps"][i]) + 1
    end
    actions
end



function get_video_from_item(item::Reproduce.Item, num_eps_for_vids)

    env = LunarLander(item.parsed_args["seed"])
    res = FileIO.load(joinpath(item.folder_str, "results.jld2"))
    
    @progress "Warm-up" for eps in 1:(length(actions)-num_eps_for_vids)
        is_terminal = false
        MinimalRLCore.start!(env)
        rew = 0.0f0
        i = 1
        while !is_terminal
            MinimalRLCore.step!(env, actions[eps][i])
            is_terminal = MinimalRLCore.is_terminal(env)
            rew += MinimalRLCore.get_reward(env)
            i += 1
        end
        if i - 1 != res["total_steps"][eps]
            @warn "Episode Ends Premeturly!!!"
        end
    end

    # eps_imgs = []
    eps_imgs = Dict{Int, Vector}()

    @progress "Get Img Seqs" for eps in (length(actions)- num_eps_for_vids + 1):length(actions)

        is_terminal = false
        MinimalRLCore.start!(env)
        imgs = [render_img_from_env(env)]
        i = 1
        while !is_terminal
            MinimalRLCore.step!(env, actions[eps][i])
            is_terminal = MinimalRLCore.is_terminal(env)
            push!(imgs, render_img_from_env(env))
            i += 1
        end
        if i - 1 != res["total_steps"][eps]
            @warn "Episode $(eps) Ends Premeturly!!!"
        end
        eps_imgs[eps] = imgs
    end
    eps_imgs
end


function save_ll_videos(ic::Reproduce.ItemCollection, num_eps_for_vids=100)
    for item in ic.items
        imgs = get_video_from_item(item, num_eps_for_vids)
        FileIO.save(joinpath(item.folder_str, "images.jld2"), "imgs", imgs)
    end
end

save_ll_videos(dir, args...) = save_ll_videos(ItemCollection(dir), args...)

end #module LunarLanderVids
