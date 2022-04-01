

function build_network(in, actions, out, args, rng; init_func, key_dict = "network")
    @assert key_dict ∈ args


    net_args = args["network"]
    layer_args = if "layers" ∈ keys(net_args)
        net_args["layers"]
    else
        [net_args[ln] for ln in sort(collect(keys(net_args)))]
    end

    
    # ls = [build_layer(in, la; init_func=init_func, rng=rng, actions=actions, net_out=out) for la in layer_args]
    ls = []
    cur_in = in isa Tuple ? in : (in, )
    for la in layer_args
        l = build_layer(cur_in, la; init_func=init_func, rng=rng, actions=actions, net_out=out)
        push!(ls, l)
        cur_in = Flux.outputsize(l, cur_in)
    end
    
    Flux.Chain(ls...)
    
    # # gn_layers = copy(gn_args["layers"])
    # cur_in = in + numhidden
    # ls = Union{ActionDense, Dense}[]
    # push!(gn_layers, [num_experts, "linear"])
    # for (layer_idx, layer) in enumerate(gn_layers)
    #     lout = layer[1]
    #     lact = layer[2]
    #     if layer_idx == 1
    #         push!(ls, ActionDense(cur_in, actions, lout, ExpUtils.FluxUtils.get_activation(lact), init=init, bias=initb(lout)))
    #     else
    #         push!(ls, Dense(cur_in, lout, ExpUtils.FluxUtils.get_activation(lact), init=init, bias=initb(lout)))
    #     end
    #     cur_in = lout
    # end
    # Flux.Chain(ls..., softmax)
    
end

import ExpUtils.Flux: get_activation

function build_layer(in, args; net_out=nothing, actions=nothing, kwargs...)
    out = if args["out"] == "net_out"
        net_out
    elseif args["out"] == "actions"
        actions
    else
        args["out"]
    end

    build_layer(Val(Symbol(args["type"])), in, out, args; kwargs...)
end

function build_layer(::Val{:Dense}, in::Tuple, out, args; init_func, kwargs...)
    act = get_activation(get(args, "act", "identity"))
    Dense(in[1], out, act; init=init_func, bias = get(args, "bias", true))
end

function build_layer(::Val{:ActionDense}, in::Tuple, out, args; init_func, actions, kwargs...)
    act = get_activation(get(args, "act", "identity"))
    ActionDense(in[1], actions, out, act; init=init_func, bias = get(args, "bias", true))
end

function build_layer(::Val{:Recur}, in::Tuple, out, args; init_func, actions, rng, kwargs...)
    build_rnn_layer(args, in[1], actions out, rng)
end

function build_layer(::Val{:Conv}, in, out, args; init_func, kwargs...)

end

