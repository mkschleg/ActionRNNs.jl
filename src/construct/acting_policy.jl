

build_acting_policy(config, env::AbstractEnvironment) =
    build_acting_policy(config, MinimalRLCore.get_actions(env))

function build_acting_policy(config, actions; pre="", kwargs...)

    p_type = ActionRNNs.getproperty(ActionRNNs, config[join([pre, "policy"], "_")])

    build_acting_policy(p_type, config, actions; pre=pre, kwargs...)
end



function build_acting_policy(p_type, args...; kwargs...)
    @error "Implement `build_acting_policy` for $(p_type)"
end

function build_acting_policy(p_type::type{ϵGreedy}, config, actions; pre="", kwargs...)
    p_type(config[join([pre, "epsilon"])])
end
