import Random

rnn_types() = ["AARNN", "MARNN", "MAARNNadd", "AAGRU", "MAGRU", "AALSTM", "MALSTM"]
fac_rnn_types() = ["FacMARNN", "FacMAGRU"]
fac_tuc_rnn_types() = ["FacTucMARNN", "FacTucMAGRU"]
gated_rnn_types() = ["ActionGatedRNN", "GAIARNN", "GAIGRU", "GAIAGRU", "GAUGRU", "GAIALSTM"]
combo_add_rnn_types() = ["CaddRNN", "CaddGRU", "CaddAAGRU", "CaddMAGRU", "CaddElRNN"]
combo_cat_rnn_types() = ["CcatRNN", "CcatGRU"]

mixture_rnn_types() = ["MixRNN", "MixElRNN", "MixElGRU", "MixGRU"]


function get_init_funcs(rng=Random.GLOBAL_RNG)
    init_func = (dims...; kwargs...)->
        ActionRNNs.glorot_uniform(rng, dims...; kwargs...)
    initb = (dims...; kwargs...) -> Flux.zeros(dims...)
    init_func, initb
end

macro create_rnn_build_trait(rnn_func, trait)
    fn = :rnn_build_trait
    quote
        function $(esc(fn))(::typeof($rnn_func))
            $trait()
        end
    end
end

rnn_build_trait(rnn_func) = @error "$(rnn_func) not supported! Please implement `construct_rnn_layer`"


struct BuildActionRNN end

for cell_func in [AARNN, MARNN, AAGRU, MAGRU, AALSTM, MALSTM]
    @eval begin
        @create_rnn_build_trait $cell_func BuildActionRNN
    end
end

struct BuildFactored end

@create_rnn_build_trait FacMARNN BuildFactored
@create_rnn_build_trait FacMAGRU BuildFactored

struct BuildTucFactored end

@create_rnn_build_trait FacTucMAGRU BuildTucFactored
@create_rnn_build_trait FacTucMARNN BuildTucFactored

struct BuildComboCat end

@create_rnn_build_trait CcatRNN BuildComboCat
@create_rnn_build_trait CcatGRU BuildComboCat

struct BuildComboAdd end

for cell_func in [CaddRNN, CaddGRU, CaddAAGRU, CaddMAGRU, CaddElRNN]
    @eval begin
        @create_rnn_build_trait $cell_func BuildComboAdd
    end
end

struct BuildMixed end

for cell_func in [MixRNN, MixElRNN, MixElGRU, MixGRU]
    @eval begin
        @create_rnn_build_trait $cell_func BuildMixed
    end
end

struct BuildFlux end

for cell_func in [Flux.RNN, Flux.GRU, Flux.LSTM]
    @eval begin
        @create_rnn_build_trait $cell_func BuildFlux
    end
end

struct BuildAGMoE end

@create_rnn_build_trait AGMoERNNCell BuildAGMoE



"""
    build_rnn_layer(in, actions, out, parsed, rng)

Build an rnn layer according from parsed. This assumes the `"cell"` key is in the `parsed` dict. in, actions, and out are integers. must explicitly pass in a RNG.

Gets layer constructor from either the ActionRNNs or Flux namespaces.

Types of build types
- `BuildActionRNN`: [`AARNN`](@ref), [`MARNN`](@ref), [`AAGRU`](@ref), [`MAGRU`](@ref), [`AALSTM`](@ref), [`MALSTM`](@ref)
- `BuildFactored`: [`FacMARNN`](@ref), [`FacMAGRU`](@ref)
- `BuildTucFactored`: [`FacTucMARNN`](@ref), [`FacTucMAGRU`](@ref)
- `BuildComboCat`: [`CcatRNN`](@ref), [`CcatGRU`](@ref)
- `BuildComboAdd`: [`CaddRNN`](@ref), [`CaddGRU`](@ref), [`CaddAAGRU`](@ref), [`CaddMAGRU`](@ref), [`CaddElRNN`](@ref)
- `BuildMixed`: [`MixRNN`](@ref), [`MixElRNN`](@ref), [`MixElGRU`](@ref), [`MixGRU`](@ref)

"""
function build_rnn_layer(parsed::Dict, in::Int, actions::Int, out::Int, rng)
    rnn_type = if isdefined(ActionRNNs, Symbol(parsed["cell"]))
        getproperty(ActionRNNs, Symbol(parsed["cell"]))
    elseif isdefined(Flux, Symbol(parsed["cell"]))
        getproperty(Flux, Symbol(parsed["cell"]))
    else
        @error """$(parsed["cell"]) not supported."""
    end

    build_rnn_layer(rnn_type, in, actions, out, parsed, rng)
end

build_rnn_layer(in::Int, actions::Int, out::Int, parsed::Dict, rng) =
    build_rnn_layer(parsed, in, actions, out, rng)

build_rnn_layer(rnn_type, in, actions, out, parsed, rng; kwargs...) =
    build_rnn_layer(rnn_build_trait(rnn_type), rnn_type, in, actions, out, parsed, rng; kwargs...)

"""
    build_rnn_layer(::BuildActionRNN, args...; kwargs...)

Standard Additive and Multiplicative cells.
No extra parameters.
"""
function build_rnn_layer(::BuildActionRNN,
                         rnn_type,
                         in, actions, out,
                         parsed, rng;
                         init_func=get_init_funcs(rng)[1],
                         initb=get_init_funcs(rng)[2],
                         kwargs...)
    
    rnn_type(in, actions, out;
             init=init_func,
             initb=initb)
end

"""
    build_rnn_layer(::BuildFactored, args...; kwargs...)

Factored (not tucker) cells.
**Extra Config Options:**
- `init_style::String`: They style of init. Check your cell for possible options.
- `factors::Int`: Number of factors in factorization.
"""
function build_rnn_layer(::BuildFactored, rnn_type,
                         in, actions, out,
                         parsed, rng;
                         init_func=get_init_funcs(rng)[1],
                         initb=get_init_funcs(rng)[2],
                         default_init_style="standard", kwargs...)

    @assert "factors" ∈ keys(parsed)
    
    factors = parsed["factors"]
    init_style = get(parsed, "init_style", default_init_style)

    rnn_type(in, actions, out, factors;
             init_style=init_style,
             init=init_func,
             initb=initb,
             rng=rng)

end

"""
    build_rnn_layer(::BuildTucFactored, args...; kwargs...)

Tucker Factored cells:
**Extra Config Options:**
- `in_factors::Int`: Number of factors in input matrix
- `action_factors::Int`: Number of factors in action matrix
- `out_factors::Int`: Number of factors in out matrix
"""
function build_rnn_layer(::BuildTucFactored,
                         rnn_type,
                         in, actions, out,
                         parsed,
                         rng;
                         init_func=get_init_funcs(rng)[1],
                         initb=get_init_funcs(rng)[2],
                         default_init_style="standard", kwargs...)

    @assert "action_factors" ∈ keys(parsed)
    @assert "out_factors" ∈ keys(parsed)
    @assert "in_factors" ∈ keys(parsed)

    action_factors = parsed["action_factors"]
    out_factors = parsed["out_factors"]
    in_factors = parsed["in_factors"]
    init_style = get(parsed, "init_style", default_init_style)

    rnn_type(in, actions, out, action_factors, out_factors, in_factors;
             init_style=init_style,
             init=init_func,
             initb=initb)
     
end

"""
    build_rnn_layer(::BuildComboCat, args...; kwargs...)

Combo cat AA/MA cells. No Extra Params.
"""
function build_rnn_layer(::BuildComboCat, rnn_type,
                         in, actions, out,
                         parsed, rng;
                         init_func=get_init_funcs(rng)[1],
                         initb=get_init_funcs(rng)[2], kwargs...)

    @assert out % 2 == 0
    
    rnn_type(in, actions, out÷2;
             init=init_func,
             initb=initb)
end

"""
    build_rnn_layer(::BuildComboAdd, args...; kwargs...)

Combo add AA/MA cells. No Extra Params.
"""
function build_rnn_layer(::BuildComboAdd, rnn_type,
                         in, actions, out,
                         parsed, rng;
                         init_func=get_init_funcs(rng)[1],
                         initb=get_init_funcs(rng)[2], kwargs...)
    
    rnn_type(in, actions, out;
             init=init_func,
             initb=initb)
end

"""
    build_rnn_layer(::BuildMixed, args...; kwargs...)

Mixed layers.
**Extra Config Options**
-`num_experts::Int`: number of parallel cells in mixture.
"""
function build_rnn_layer(::BuildMixed, rnn_type,
                         in, actions, out,
                         parsed, rng;
                         init_func=get_init_funcs(rng)[1],
                         initb=get_init_funcs(rng)[2], kwargs...)

    @assert "num_experts" ∈ keys(parsed)
    
    ne = parsed["num_experts"]
    rnn_type(in, actions, out, ne;
        init=init_func,
        initb=initb)

end

"""
    build_rnn_layer(::BuildFlux, args...; kwargs...)

Flux cell. No extra parameters.
"""
function build_rnn_layer(::BuildFlux, rnn_type,
                         in, actions, out,
                         parsed, rng;
                         init_func=get_init_funcs(rng)[1], kwargs...)
    rnn_type(in, out; init=init_func)
end

"""
    build_rnn_layer(::BuildAGMoE, args...; kwargs...)
"""

function build_rnn_layer(::BuildAGMoE, rnn_type, in, actions, out, parsed, rng;
                         init_func=get_init_funcs(rng)[1],
                         initb=get_init_funcs(rng)[2], kwargs...)

    @assert "gating_network" ∈ keys(parsed)
    @assert "num_experts" ∈ keys(parsed)

    num_experts = parsed["num_experts"]
    gating_network = build_gating_network(
        Val(Symbol(parsed["gating_network"])),
        in, actions, out,
        num_experts, parsed,
        init, initb)

    rnn_type(in, actions, out, num_experts, gating_network; init=init_func, initb=initb, kwargs...)
    
end

"""
    build_gating_network

    [[out, activation]]
"""
function build_gating_network(::Val{:default}, in, actions, numhidden, num_experts, parsed, init, initb)
    @assert "gn_layers" ∈ keys(parsed)

    gn_layers = copy(parsed["gn_layers"])
    cur_in = in + numhidden
    ls = Union{ActionDense, Dense}[]
    push!(gn_layers, [num_experts, "linear"])
    for (layer_idx, layer) in enumerate(gn_layers)
        lout = layer[1]
        lact = layer[2]
        if layer_idx == 1
            push!(ls, ActionDense(cur_in, actions, lout, ExpUtils.FluxUtils.get_activation(lact), init=init, bias=initb(lout)))
        else
            push!(ls, Dense(cur_in, lout, ExpUtils.FluxUtils.get_activation(lact), init=init, bias=initb(lout)))
        end
        cur_in = lout
    end
    Flux.Chain(ls..., softmax)
end
