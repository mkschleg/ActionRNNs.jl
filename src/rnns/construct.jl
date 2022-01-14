
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

function build_rnn_layer(in::Int, actions::Int, out::Int, parsed, rng)
    rnn_type = if isdefined(ActionRNNs, Symbol(parsed["cell"]))
        getproperty(ActionRNNs, Symbol(parsed["cell"]))
    elseif isdefined(Flux, Symbol(parsed["cell"]))
        getproperty(Flux, Symbol(parsed["cell"]))
    else
        @error """$(parsed["cell"]) not supported."""
    end

    build_rnn_layer(rnn_type, in, actions, out, parsed, rng)
end

build_rnn_layer(rnn_type, in, actions, out, parsed, rng; kwargs...) =
    build_rnn_layer(rnn_build_trait(rnn_type), rnn_type, in, actions, out, parsed, rng; kwargs...)

struct BuildActionRNN end

for cell_func in [AARNN, MARNN, AAGRU, MAGRU, AALSTM, MALSTM]
    @eval begin
        @create_rnn_build_trait $cell_func BuildActionRNN
    end
end

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


struct BuildFactored end

@create_rnn_build_trait FacMARNN BuildFactored
@create_rnn_build_trait FacMAGRU BuildFactored

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
             initb=initb)

end

struct BuildTucFactored end

@create_rnn_build_trait FacTucMAGRU BuildTucFactored
@create_rnn_build_trait FacTucMARNN BuildTucFactored

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

struct BuildComboCat end

@create_rnn_build_trait CcatRNN BuildComboCat
@create_rnn_build_trait CcatGRU BuildComboCat

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

struct BuildComboAdd end

for cell_func in [CaddRNN, CaddGRU, CaddAAGRU, CaddMAGRU, CaddElRNN]
    @eval begin
        @create_rnn_build_trait $cell_func BuildComboAdd
    end
end

function build_rnn_layer(::BuildComboAdd, rnn_type,
                         in, actions, out,
                         parsed, rng;
                         init_func=get_init_funcs(rng)[1],
                         initb=get_init_funcs(rng)[2], kwargs...)
    
    rnn_type(in, actions, out;
             init=init_func,
             initb=initb)
end

struct BuildMixed end

for cell_func in [MixRNN, MixElRNN, MixElGRU, MixGRU]
    @eval begin
        @create_rnn_build_trait $cell_func BuildMixed
    end
end

function build_rnn_layer(::BuildMixed, rnn_type,
                         in, actions, out,
                         parsed, rng;
                         init_func=get_init_funcs(rng)[1],
                         initb=get_init_funcs(rng)[2], kwargs...)

    @assert "num_experts" ∈ keys(parsed)
    
    ne = parsed["num_experts"]
    rnn_type(in, na, nh, ne;
        init=init_func,
        initb=initb)

end


struct BuildFlux end

for cell_func in [Flux.RNN, Flux.GRU, Flux.LSTM]
    @eval begin
        @create_rnn_build_trait $cell_func BuildFlux
    end
end

function build_rnn_layer(::BuildFlux, rnn_type,
                         in, actions, out,
                         parsed, rng;
                         init_func=get_init_funcs(rng)[1], kwargs...)
    rnn_type(in, nh; init=init_func)
end
