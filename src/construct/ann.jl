
import Macros: @info_str, InfoStr


function get_help_strs_and_places(default_config, __module__)
    start_str = "# Automatically generated docs for $(__module__) config."
    help_str_strg = InfoStr[
        InfoStr(start_str)
    ]
    postwalk(default_config) do expr
        expr_str = string(expr)
        if length(expr_str) > 5 && (expr_str[1:5] == "help\"" || expr_str[1:5] == "info\"")
            push!(help_str_strg, InfoStr(string(expr)[6:end-1]))
        end
        expr
    end
    md_strs = [Markdown.parse(hs.str) for hs in help_str_strg]
    join(md_strs, "\n")
end


macro create_build_ann_func(layers_desc)

    
    

    
end


@create_build_ann_func begin
    info"ACTION_STREAM"
    Flux.Chain(
        (a)->Flux.onehotbatch(a, 1:actions),
        Flux.Dense(actions, internal_a, Flux.relu, initW=init_func),
    )
    
    info"OBS_STREAM"
    identity
    
    info"POST_NETWORK"
    Flux.Dense(nh, actions; initW=init_func)
end




function build_deep_action_rnn_layers(in, actions, out, parsed, rng)


    # Deep actions for RNNs from Zhu et al 2018
    internal_a = parsed["internal_a"]

    init_func, initb = ActionRNNs.get_init_funcs(rng)

    #= Custom part =#
    action_stream = Flux.Chain(
        (a)->Flux.onehotbatch(a, 1:actions),
        Flux.Dense(actions, internal_a, Flux.relu, initW=init_func),
    )
    
    obs_stream = identity

    ActionRNNs.DualStreams(action_stream, obs_stream)
    # (ActionRNNs.DualStreams(action_stream, obs_stream),
    #  ActionRNNs.build_rnn_layer(internal_o, internal_a, out, parsed, rng))
end


function build_ann(in, actions::Int, parsed, rng)
    
    nh = parsed["numhidden"]
    init_func, initb = ActionRNNs.get_init_funcs(rng)

    deep_action = get(parsed, "deep", false)
    rnn = if deep_action
        build_deep_action_rnn_layers(in, actions, nh, parsed, rng)
    else
        (ActionRNNs.build_rnn_layer(in, actions, nh, parsed, rng),)
    end

    #= Custom part =#
    Flux.Chain(rnn...,
               Flux.Dense(nh, actions; initW=init_func))
    
end

