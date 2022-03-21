ret = let
    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:983 =#
    (issubset)((Tullio.extremerange)(a), (axes)(mid, 1)) || (throw)("extrema of index a[k] must fit within mid")
    (ndims)(mid) == 3 || (throw)("expected a 3-array mid")
    begin
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1041 =#
        local function 𝒜𝒸𝓉!(::Type, ℛ::AbstractArray{𝒯}, mid, a, 𝒶𝓍i, 𝒶𝓍k, ♻️ = nothing, 💀 = true) where 𝒯
            $(Expr(:meta, :inline))
            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1041 =#
            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1042 =#
            begin
                $(Expr(:inbounds, true))
                local var"#477#val" = begin
                    nothing
                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1042 =#
                    for k = 𝒶𝓍k
                        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                        for i = 𝒶𝓍i
                            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                            begin
                                nothing
                                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                                ℛ[i, k] = mid[a[k], i, k]
                                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                                nothing
                            end
                        end
                    end
                end
                $(Expr(:inbounds, :pop))
                var"#477#val"
            end
        end
    end
    
    begin
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1090 =#
        local function 𝒜𝒸𝓉!(::Type{<:Array{<:LoopVectorization.NativeTypes}}, ℛ::AbstractArray{𝒯}, mid, a, 𝒶𝓍i, 𝒶𝓍k, ♻️ = nothing, 💀 = true) where 𝒯
            $(Expr(:meta, :inline))
            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1090 =#
            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1091 =#
            nothing
            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1092 =#
            begin
                #= logging.jl:370 =#
                let
                    #= logging.jl:371 =#
                    var"#255#level" = Base.CoreLogging.Info
                    #= logging.jl:372 =#
                    var"#256#std_level" = Base.CoreLogging.convert(Base.CoreLogging.LogLevel, var"#255#level")
                    #= logging.jl:373 =#
                    if var"#256#std_level" >= Base.CoreLogging._min_enabled_level[]
                        #= logging.jl:374 =#
                        var"#257#group" = :macro
                        #= logging.jl:375 =#
                        var"#258#_module" = Main
                        #= logging.jl:376 =#
                        var"#259#logger" = Base.CoreLogging.current_logger_for_env(var"#256#std_level", var"#257#group", var"#258#_module")
                        #= logging.jl:377 =#
                        if !(var"#259#logger" === Base.CoreLogging.nothing)
                            #= logging.jl:378 =#
                            var"#260#id" = 0xa3820e29d8c54d18
                            #= logging.jl:381 =#
                            if Base.CoreLogging._invoked_shouldlog(var"#259#logger", var"#255#level", var"#258#_module", var"#257#group", var"#260#id")
                                #= logging.jl:382 =#
                                var"#261#file" = "/Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl"
                                #= logging.jl:383 =#
                                var"#262#line" = 1083
                                #= logging.jl:384 =#
                                local var"#263#msg", var"#264#kwargs"
                                #= logging.jl:385 =#
                                begin
                                    #= logging.jl:359 =#
                                    try
                                        #= logging.jl:360 =#
                                        var"#263#msg" = "running LoopVectorization actor "
                                        #= logging.jl:361 =#
                                        var"#264#kwargs" = (; maxlog = 3)
                                        #= logging.jl:362 =#
                                        true
                                    catch var"#277#err"
                                        #= logging.jl:364 =#
                                        Base.CoreLogging.logging_error(var"#259#logger", var"#255#level", var"#258#_module", var"#257#group", var"#260#id", var"#261#file", var"#262#line", var"#277#err", true)
                                        #= logging.jl:365 =#
                                        false
                                    end
                                end && Base.CoreLogging.handle_message(var"#259#logger", var"#255#level", var"#263#msg", var"#258#_module", var"#257#group", var"#260#id", var"#261#file", var"#262#line"; var"#264#kwargs"...)
                            end
                        end
                    end
                    #= logging.jl:391 =#
                    Base.CoreLogging.nothing
                end
            end
            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1093 =#
            LoopVectorization.check_args(mid, a) || begin
                #= logging.jl:370 =#
                let
                    #= logging.jl:371 =#
                    var"#278#level" = Base.CoreLogging.Error
                    #= logging.jl:372 =#
                    var"#279#std_level" = Base.CoreLogging.convert(Base.CoreLogging.LogLevel, var"#278#level")
                    #= logging.jl:373 =#
                    if var"#279#std_level" >= Base.CoreLogging._min_enabled_level[]
                        #= logging.jl:374 =#
                        var"#280#group" = :macro
                        #= logging.jl:375 =#
                        var"#281#_module" = Main
                        #= logging.jl:376 =#
                        var"#282#logger" = Base.CoreLogging.current_logger_for_env(var"#279#std_level", var"#280#group", var"#281#_module")
                        #= logging.jl:377 =#
                        if !(var"#282#logger" === Base.CoreLogging.nothing)
                            #= logging.jl:378 =#
                            var"#283#id" = 0xa3820e29d8c54d18
                            #= logging.jl:381 =#
                            if Base.CoreLogging._invoked_shouldlog(var"#282#logger", var"#278#level", var"#281#_module", var"#280#group", var"#283#id")
                                #= logging.jl:382 =#
                                var"#284#file" = "/Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl"
                                #= logging.jl:383 =#
                                var"#285#line" = 1084
                                #= logging.jl:384 =#
                                local var"#286#msg", var"#287#kwargs"
                                #= logging.jl:385 =#
                                begin
                                    #= logging.jl:359 =#
                                    try
                                        #= logging.jl:360 =#
                                        var"#286#msg" = "rejected by LoopVectorization's check_args! "
                                        #= logging.jl:361 =#
                                        var"#287#kwargs" = (; maxlog = 3)
                                        #= logging.jl:362 =#
                                        true
                                    catch var"#300#err"
                                        #= logging.jl:364 =#
                                        Base.CoreLogging.logging_error(var"#282#logger", var"#278#level", var"#281#_module", var"#280#group", var"#283#id", var"#284#file", var"#285#line", var"#300#err", true)
                                        #= logging.jl:365 =#
                                        false
                                    end
                                end && Base.CoreLogging.handle_message(var"#282#logger", var"#278#level", var"#286#msg", var"#281#_module", var"#280#group", var"#283#id", var"#284#file", var"#285#line"; var"#287#kwargs"...)
                            end
                        end
                    end
                    #= logging.jl:391 =#
                    Base.CoreLogging.nothing
                end
            end
            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1094 =#
            begin
                begin
                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                    nothing
                end
                begin
                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                    nothing
                end
                begin
                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                    nothing
                end
                begin
                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                    nothing
                end
                begin
                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1094 =#
                    nothing
                end
                var"###looprangek###1###" = LoopVectorization.canonicalize_range(begin
                                                                                 $(Expr(:inbounds, true))
                                                                                 local var"#301#val" = 𝒶𝓍k
                                                                                 $(Expr(:inbounds, :pop))
                                                                                 var"#301#val"
                                                                                 end)
                var"###looplenk###2###" = ArrayInterface.static_length(var"###looprangek###1###")
                var"###k_loop_lower_bound###3###" = LoopVectorization.maybestaticfirst(var"###looprangek###1###")
                var"###k_loop_upper_bound###4###" = LoopVectorization.maybestaticlast(var"###looprangek###1###")
                var"###k_loop_step###5###" = LoopVectorization.static_step(var"###looprangek###1###")
                var"###looprangei###6###" = LoopVectorization.canonicalize_range(begin
                                                                                 $(Expr(:inbounds, true))
                                                                                 local var"#302#val" = 𝒶𝓍i
                                                                                 $(Expr(:inbounds, :pop))
                                                                                 var"#302#val"
                                                                                 end)
                var"###loopleni###7###" = ArrayInterface.static_length(var"###looprangei###6###")
                var"###i_loop_lower_bound###8###" = LoopVectorization.maybestaticfirst(var"###looprangei###6###")
                var"###i_loop_upper_bound###9###" = LoopVectorization.maybestaticlast(var"###looprangei###6###")
                var"###i_loop_step###10###" = LoopVectorization.static_step(var"###looprangei###6###")
                if LoopVectorization.check_args(ℛ, mid, a)
                    (var"##vptr##_ℛ", var"#ℛ#preserve#buffer#") = LoopVectorization.stridedpointer_preserve(ℛ)
                    (var"##vptr##_mid", var"#mid#preserve#buffer#") = LoopVectorization.stridedpointer_preserve(mid)
                    (var"##vptr##_a", var"#a#preserve#buffer#") = LoopVectorization.stridedpointer_preserve(a)
                    var"####grouped#strided#pointer####14###" = Core.getfield(LoopVectorization.grouped_strided_pointer((LoopVectorization.densewrapper(LoopVectorization.gespf1(var"##vptr##_a", (var"###k_loop_lower_bound###3###",)), a), LoopVectorization.densewrapper(LoopVectorization.gespf1(var"##vptr##_mid", (LoopVectorization.similardims(var"###looprangek###1###", static(0)), var"###i_loop_lower_bound###8###", var"###k_loop_lower_bound###3###")), mid), LoopVectorization.densewrapper(LoopVectorization.gespf1(var"##vptr##_ℛ", (var"###i_loop_lower_bound###8###", var"###k_loop_lower_bound###3###")), ℛ)), Val{()}()), 1)
                    $(Expr(:gc_preserve, quote
                           var"##vargsym#315" = ((LoopVectorization.zerorangestart(var"###looprangek###1###"), LoopVectorization.zerorangestart(var"###looprangei###6###")), (var"####grouped#strided#pointer####14###",))
                           var"##Tloopeltype##" = LoopVectorization.promote_type(LoopVectorization.eltype(ℛ), LoopVectorization.eltype(mid), LoopVectorization.eltype(a))
                           var"##Wvecwidth##" = LoopVectorization.pick_vector_width(var"##Tloopeltype##")
                           LoopVectorization._turbo_!(LoopVectorization.avx_config_val(Val{(false, 0, 0, 0, false, 0x0000000000000001)}(), var"##Wvecwidth##"), Val{(:LoopVectorization, :getindex, LoopVectorization.OperationStruct(0x00000000000000000000000000000001, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, LoopVectorization.memload, 0x0001, 0x01), :LoopVectorization, :getindex, LoopVectorization.OperationStruct(0x00000000000000000000000000000121, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000001, 0x00000000000000000000000000000000, LoopVectorization.memload, 0x0002, 0x02), :LoopVectorization, :identity, LoopVectorization.OperationStruct(0x00000000000000000000000000000012, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000002, 0x00000000000000000000000000000000, LoopVectorization.compute, 0x0003, 0x00), :LoopVectorization, :setindex!, LoopVectorization.OperationStruct(0x00000000000000000000000000000021, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000003, 0x00000000000000000000000000000000, LoopVectorization.memstore, 0x0004, 0x03))}(), Val{(LoopVectorization.ArrayRefStruct{:a, Symbol("##vptr##_a")}(0x00000000000000000000000000000001, 0x00000000000000000000000000000001, 0x00000000000000000000000000000000, 0x00000000000000000000000000000001), LoopVectorization.ArrayRefStruct{:mid, Symbol("##vptr##_mid")}(0x00000000000000000000000000020101, 0x00000000000000000000000000010201, 0x00000000000000000000000000000000, 0x00000000000000000000000000010101), LoopVectorization.ArrayRefStruct{:ℛ, Symbol("##vptr##_ℛ")}(0x00000000000000000000000000000101, 0x00000000000000000000000000000201, 0x00000000000000000000000000000000, 0x00000000000000000000000000000101))}(), Val{(0, (), (), (), (), (), ())}(), Val{(:k, :i)}(), Base.Val(Base.typeof(var"##vargsym#315")), LoopVectorization.flatten_to_tuple(var"##vargsym#315")...)
                           end, Symbol("#a#preserve#buffer#"), Symbol("#mid#preserve#buffer#"), Symbol("#ℛ#preserve#buffer#")))
                    nothing
                else
                    begin
                        #= logging.jl:370 =#
                        let
                            #= logging.jl:371 =#
                            var"#303#level" = Base.CoreLogging.Warn
                            #= logging.jl:372 =#
                            var"#304#std_level" = Base.CoreLogging.convert(Base.CoreLogging.LogLevel, var"#303#level")
                            #= logging.jl:373 =#
                            if var"#304#std_level" >= Base.CoreLogging._min_enabled_level[]
                                #= logging.jl:374 =#
                                var"#305#group" = :condense_loopset
                                #= logging.jl:375 =#
                                var"#306#_module" = Main
                                #= logging.jl:376 =#
                                var"#307#logger" = Base.CoreLogging.current_logger_for_env(var"#304#std_level", var"#305#group", var"#306#_module")
                                #= logging.jl:377 =#
                                if !(var"#307#logger" === Base.CoreLogging.nothing)
                                    #= logging.jl:378 =#
                                    var"#308#id" = :Main_1fbe9ed4
                                    #= logging.jl:381 =#
                                    if Base.CoreLogging._invoked_shouldlog(var"#307#logger", var"#303#level", var"#306#_module", var"#305#group", var"#308#id")
                                        #= logging.jl:382 =#
                                        var"#309#file" = "/Users/Matt/.julia/packages/LoopVectorization/x4G96/src/condense_loopset.jl"
                                        #= logging.jl:383 =#
                                        var"#310#line" = 825
                                        #= logging.jl:384 =#
                                        local var"#312#msg", var"#313#kwargs"
                                        #= logging.jl:385 =#
                                        begin
                                            #= logging.jl:346 =#
                                            let var"#311#err" = nothing
                                                #= logging.jl:347 =#
                                                if var"#311#err" === Base.CoreLogging.nothing
                                                    #= logging.jl:348 =#
                                                    var"#312#msg" = "#= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1094 =#:\n`LoopVectorization.check_args` on your inputs failed; running fallback `@inbounds @fastmath` loop instead.\nUse `warn_check_args=false`, e.g. `@turbo warn_check_args=false ...`, to disable this warning."
                                                    #= logging.jl:349 =#
                                                    var"#313#kwargs" = (; maxlog = 1)
                                                    #= logging.jl:350 =#
                                                    true
                                                else
                                                    #= logging.jl:352 =#
                                                    Base.CoreLogging.logging_error(var"#307#logger", var"#303#level", var"#306#_module", var"#305#group", var"#308#id", var"#309#file", var"#310#line", var"#311#err", false)
                                                    #= logging.jl:353 =#
                                                    false
                                                end
                                            end
                                        end && Base.CoreLogging.handle_message(var"#307#logger", var"#303#level", var"#312#msg", var"#306#_module", var"#305#group", var"#308#id", var"#309#file", var"#310#line"; var"#313#kwargs"...)
                                    end
                                end
                            end
                            #= logging.jl:391 =#
                            Base.CoreLogging.nothing
                        end
                    end
                    begin
                        $(Expr(:inbounds, true))
                        local var"#328#val" = for k = 𝒶𝓍k
                            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                            for i = 𝒶𝓍i
                                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                                begin
                                    nothing
                                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                                    ℛ[i, k] = mid[a[k], i, k]
                                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                                    nothing
                                end
                            end
                        end
                        $(Expr(:inbounds, :pop))
                        var"#328#val"
                    end
                end
            end
            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1095 =#
            nothing
        end
    end
begin
    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1041 =#
    local function ∇𝒜𝒸𝓉!(::Type, 𝛥mid, 𝛥a, 𝛥ℛ::AbstractArray{𝒯}, ℛ, mid, a, 𝒶𝓍i, 𝒶𝓍k, ♻️ = nothing, 💀 = true) where 𝒯
        $(Expr(:meta, :inline))
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1041 =#
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1042 =#
        begin
            $(Expr(:inbounds, true))
            local var"#478#val" = begin
                nothing
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1042 =#
                for k = 𝒶𝓍k
                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                    for i = 𝒶𝓍i
                        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                        begin
                            nothing
                            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                            begin
                                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:327 =#
                                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:328 =#
                                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:53 =#
                                𝛥mid[a[k], i, k] = Base.FastMath.add_fast(𝛥mid[a[k], i, k], 𝛥ℛ[i, k])
                            end
                            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                            nothing
                        end
                    end
                end
            end
            $(Expr(:inbounds, :pop))
            var"#478#val"
        end
    end
end
begin
    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1090 =#
    local function ∇𝒜𝒸𝓉!(::Type{<:Array{<:LoopVectorization.NativeTypes}}, 𝛥mid, 𝛥a, 𝛥ℛ::AbstractArray{𝒯}, ℛ, mid, a, 𝒶𝓍i, 𝒶𝓍k, ♻️ = nothing, 💀 = true) where 𝒯
        $(Expr(:meta, :inline))
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1090 =#
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1091 =#
        nothing
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1092 =#
        begin
            #= logging.jl:370 =#
            let
                #= logging.jl:371 =#
                var"#329#level" = Base.CoreLogging.Info
                #= logging.jl:372 =#
                var"#330#std_level" = Base.CoreLogging.convert(Base.CoreLogging.LogLevel, var"#329#level")
                #= logging.jl:373 =#
                if var"#330#std_level" >= Base.CoreLogging._min_enabled_level[]
                    #= logging.jl:374 =#
                    var"#331#group" = :macro
                    #= logging.jl:375 =#
                    var"#332#_module" = Main
                    #= logging.jl:376 =#
                    var"#333#logger" = Base.CoreLogging.current_logger_for_env(var"#330#std_level", var"#331#group", var"#332#_module")
                    #= logging.jl:377 =#
                    if !(var"#333#logger" === Base.CoreLogging.nothing)
                        #= logging.jl:378 =#
                        var"#334#id" = 0xa3820e29d8c54d18
                        #= logging.jl:381 =#
                        if Base.CoreLogging._invoked_shouldlog(var"#333#logger", var"#329#level", var"#332#_module", var"#331#group", var"#334#id")
                            #= logging.jl:382 =#
                            var"#335#file" = "/Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl"
                            #= logging.jl:383 =#
                            var"#336#line" = 1083
                            #= logging.jl:384 =#
                            local var"#337#msg", var"#338#kwargs"
                            #= logging.jl:385 =#
                            begin
                                #= logging.jl:359 =#
                                try
                                    #= logging.jl:360 =#
                                    var"#337#msg" = "running LoopVectorization actor (symbolic gradient)"
                                    #= logging.jl:361 =#
                                    var"#338#kwargs" = (; maxlog = 3)
                                    #= logging.jl:362 =#
                                    true
                                catch var"#351#err"
                                    #= logging.jl:364 =#
                                    Base.CoreLogging.logging_error(var"#333#logger", var"#329#level", var"#332#_module", var"#331#group", var"#334#id", var"#335#file", var"#336#line", var"#351#err", true)
                                    #= logging.jl:365 =#
                                    false
                                end
                            end && Base.CoreLogging.handle_message(var"#333#logger", var"#329#level", var"#337#msg", var"#332#_module", var"#331#group", var"#334#id", var"#335#file", var"#336#line"; var"#338#kwargs"...)
                        end
                    end
                end
                #= logging.jl:391 =#
                Base.CoreLogging.nothing
            end
        end
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1093 =#
        LoopVectorization.check_args(mid, a) || begin
            #= logging.jl:370 =#
            let
                #= logging.jl:371 =#
                var"#352#level" = Base.CoreLogging.Error
                #= logging.jl:372 =#
                var"#353#std_level" = Base.CoreLogging.convert(Base.CoreLogging.LogLevel, var"#352#level")
                #= logging.jl:373 =#
                if var"#353#std_level" >= Base.CoreLogging._min_enabled_level[]
                    #= logging.jl:374 =#
                    var"#354#group" = :macro
                    #= logging.jl:375 =#
                    var"#355#_module" = Main
                    #= logging.jl:376 =#
                    var"#356#logger" = Base.CoreLogging.current_logger_for_env(var"#353#std_level", var"#354#group", var"#355#_module")
                    #= logging.jl:377 =#
                    if !(var"#356#logger" === Base.CoreLogging.nothing)
                        #= logging.jl:378 =#
                        var"#357#id" = 0xa3820e29d8c54d18
                        #= logging.jl:381 =#
                        if Base.CoreLogging._invoked_shouldlog(var"#356#logger", var"#352#level", var"#355#_module", var"#354#group", var"#357#id")
                            #= logging.jl:382 =#
                            var"#358#file" = "/Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl"
                            #= logging.jl:383 =#
                            var"#359#line" = 1084
                            #= logging.jl:384 =#
                            local var"#360#msg", var"#361#kwargs"
                            #= logging.jl:385 =#
                            begin
                                #= logging.jl:359 =#
                                try
                                    #= logging.jl:360 =#
                                    var"#360#msg" = "rejected by LoopVectorization's check_args! (symbolic gradient)"
                                    #= logging.jl:361 =#
                                    var"#361#kwargs" = (; maxlog = 3)
                                    #= logging.jl:362 =#
                                    true
                                catch var"#374#err"
                                    #= logging.jl:364 =#
                                    Base.CoreLogging.logging_error(var"#356#logger", var"#352#level", var"#355#_module", var"#354#group", var"#357#id", var"#358#file", var"#359#line", var"#374#err", true)
                                    #= logging.jl:365 =#
                                    false
                                end
                            end && Base.CoreLogging.handle_message(var"#356#logger", var"#352#level", var"#360#msg", var"#355#_module", var"#354#group", var"#357#id", var"#358#file", var"#359#line"; var"#361#kwargs"...)
                        end
                    end
                end
                #= logging.jl:391 =#
                Base.CoreLogging.nothing
            end
        end
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1094 =#
        begin
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                nothing
            end
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:53 =#
                nothing
            end
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:328 =#
                nothing
            end
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:327 =#
                nothing
            end
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                nothing
            end
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                nothing
            end
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                nothing
            end
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1094 =#
                nothing
            end
            var"###looprangek###1###" = LoopVectorization.canonicalize_range(begin
                                                                             $(Expr(:inbounds, true))
                                                                             local var"#375#val" = 𝒶𝓍k
                                                                             $(Expr(:inbounds, :pop))
                                                                             var"#375#val"
                                                                             end)
            var"###looplenk###2###" = ArrayInterface.static_length(var"###looprangek###1###")
            var"###k_loop_lower_bound###3###" = LoopVectorization.maybestaticfirst(var"###looprangek###1###")
            var"###k_loop_upper_bound###4###" = LoopVectorization.maybestaticlast(var"###looprangek###1###")
            var"###k_loop_step###5###" = LoopVectorization.static_step(var"###looprangek###1###")
            var"###looprangei###6###" = LoopVectorization.canonicalize_range(begin
                                                                             $(Expr(:inbounds, true))
                                                                             local var"#376#val" = 𝒶𝓍i
                                                                             $(Expr(:inbounds, :pop))
                                                                             var"#376#val"
                                                                             end)
            var"###loopleni###7###" = ArrayInterface.static_length(var"###looprangei###6###")
            var"###i_loop_lower_bound###8###" = LoopVectorization.maybestaticfirst(var"###looprangei###6###")
            var"###i_loop_upper_bound###9###" = LoopVectorization.maybestaticlast(var"###looprangei###6###")
            var"###i_loop_step###10###" = LoopVectorization.static_step(var"###looprangei###6###")
            if LoopVectorization.check_args(𝛥mid, a, 𝛥ℛ)
                (var"##vptr##_𝛥mid", var"#𝛥mid#preserve#buffer#") = LoopVectorization.stridedpointer_preserve(𝛥mid)
                (var"##vptr##_a", var"#a#preserve#buffer#") = LoopVectorization.stridedpointer_preserve(a)
                (var"##vptr##_𝛥ℛ", var"#𝛥ℛ#preserve#buffer#") = LoopVectorization.stridedpointer_preserve(𝛥ℛ)
                var"####grouped#strided#pointer####15###" = Core.getfield(LoopVectorization.grouped_strided_pointer((LoopVectorization.densewrapper(LoopVectorization.gespf1(var"##vptr##_a", (var"###k_loop_lower_bound###3###",)), a), LoopVectorization.densewrapper(LoopVectorization.gespf1(var"##vptr##_𝛥mid", (LoopVectorization.similardims(var"###looprangek###1###", static(0)), var"###i_loop_lower_bound###8###", var"###k_loop_lower_bound###3###")), 𝛥mid), LoopVectorization.densewrapper(LoopVectorization.gespf1(var"##vptr##_𝛥ℛ", (var"###i_loop_lower_bound###8###", var"###k_loop_lower_bound###3###")), 𝛥ℛ)), Val{()}()), 1)
                $(Expr(:gc_preserve, quote
                       var"##vargsym#319" = ((LoopVectorization.zerorangestart(var"###looprangek###1###"), LoopVectorization.zerorangestart(var"###looprangei###6###")), (var"####grouped#strided#pointer####15###",))
                       var"##Tloopeltype##" = LoopVectorization.promote_type(LoopVectorization.eltype(𝛥mid), LoopVectorization.eltype(a), LoopVectorization.eltype(𝛥ℛ))
                       var"##Wvecwidth##" = LoopVectorization.pick_vector_width(var"##Tloopeltype##")
                       LoopVectorization._turbo_!(LoopVectorization.avx_config_val(Val{(false, 0, 0, 0, false, 0x0000000000000001)}(), var"##Wvecwidth##"), Val{(:LoopVectorization, :getindex, LoopVectorization.OperationStruct(0x00000000000000000000000000000001, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, LoopVectorization.memload, 0x0001, 0x01), :LoopVectorization, :getindex, LoopVectorization.OperationStruct(0x00000000000000000000000000000121, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000001, 0x00000000000000000000000000000000, LoopVectorization.memload, 0x0002, 0x02), :LoopVectorization, :getindex, LoopVectorization.OperationStruct(0x00000000000000000000000000000021, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, LoopVectorization.memload, 0x0003, 0x03), :LoopVectorization, :add_fast, LoopVectorization.OperationStruct(0x00000000000000000000000000000012, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000020003, 0x00000000000000000000000000000000, LoopVectorization.compute, 0x0002, 0x00), :LoopVectorization, :setindex!, LoopVectorization.OperationStruct(0x00000000000000000000000000000121, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000040001, 0x00000000000000000000000000000000, LoopVectorization.memstore, 0x0004, 0x02))}(), Val{(LoopVectorization.ArrayRefStruct{:a, Symbol("##vptr##_a")}(0x00000000000000000000000000000001, 0x00000000000000000000000000000001, 0x00000000000000000000000000000000, 0x00000000000000000000000000000001), LoopVectorization.ArrayRefStruct{:𝛥mid, Symbol("##vptr##_𝛥mid")}(0x00000000000000000000000000020101, 0x00000000000000000000000000010201, 0x00000000000000000000000000000000, 0x00000000000000000000000000010101), LoopVectorization.ArrayRefStruct{:𝛥ℛ, Symbol("##vptr##_𝛥ℛ")}(0x00000000000000000000000000000101, 0x00000000000000000000000000000201, 0x00000000000000000000000000000000, 0x00000000000000000000000000000101))}(), Val{(0, (), (), (), (), (), ())}(), Val{(:k, :i)}(), Base.Val(Base.typeof(var"##vargsym#319")), LoopVectorization.flatten_to_tuple(var"##vargsym#319")...)
                       end, Symbol("#a#preserve#buffer#"), Symbol("#𝛥mid#preserve#buffer#"), Symbol("#𝛥ℛ#preserve#buffer#")))
                nothing
            else
                begin
                    #= logging.jl:370 =#
                    let
                        #= logging.jl:371 =#
                        var"#377#level" = Base.CoreLogging.Warn
                        #= logging.jl:372 =#
                        var"#378#std_level" = Base.CoreLogging.convert(Base.CoreLogging.LogLevel, var"#377#level")
                        #= logging.jl:373 =#
                        if var"#378#std_level" >= Base.CoreLogging._min_enabled_level[]
                            #= logging.jl:374 =#
                            var"#379#group" = :condense_loopset
                            #= logging.jl:375 =#
                            var"#380#_module" = Main
                            #= logging.jl:376 =#
                            var"#381#logger" = Base.CoreLogging.current_logger_for_env(var"#378#std_level", var"#379#group", var"#380#_module")
                            #= logging.jl:377 =#
                            if !(var"#381#logger" === Base.CoreLogging.nothing)
                                #= logging.jl:378 =#
                                var"#382#id" = :Main_1fbe9ed5
                                #= logging.jl:381 =#
                                if Base.CoreLogging._invoked_shouldlog(var"#381#logger", var"#377#level", var"#380#_module", var"#379#group", var"#382#id")
                                    #= logging.jl:382 =#
                                    var"#383#file" = "/Users/Matt/.julia/packages/LoopVectorization/x4G96/src/condense_loopset.jl"
                                    #= logging.jl:383 =#
                                    var"#384#line" = 825
                                    #= logging.jl:384 =#
                                    local var"#386#msg", var"#387#kwargs"
                                    #= logging.jl:385 =#
                                    begin
                                        #= logging.jl:346 =#
                                        let var"#385#err" = nothing
                                            #= logging.jl:347 =#
                                            if var"#385#err" === Base.CoreLogging.nothing
                                                #= logging.jl:348 =#
                                                var"#386#msg" = "#= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1094 =#:\n`LoopVectorization.check_args` on your inputs failed; running fallback `@inbounds @fastmath` loop instead.\nUse `warn_check_args=false`, e.g. `@turbo warn_check_args=false ...`, to disable this warning."
                                                #= logging.jl:349 =#
                                                var"#387#kwargs" = (; maxlog = 1)
                                                #= logging.jl:350 =#
                                                true
                                            else
                                                #= logging.jl:352 =#
                                                Base.CoreLogging.logging_error(var"#381#logger", var"#377#level", var"#380#_module", var"#379#group", var"#382#id", var"#383#file", var"#384#line", var"#385#err", false)
                                                #= logging.jl:353 =#
                                                false
                                            end
                                        end
                                    end && Base.CoreLogging.handle_message(var"#381#logger", var"#377#level", var"#386#msg", var"#380#_module", var"#379#group", var"#382#id", var"#383#file", var"#384#line"; var"#387#kwargs"...)
                                end
                            end
                        end
                        #= logging.jl:391 =#
                        Base.CoreLogging.nothing
                    end
                end
                begin
                    $(Expr(:inbounds, true))
                    local var"#402#val" = for k = 𝒶𝓍k
                        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                        for i = 𝒶𝓍i
                            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                            begin
                                nothing
                                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                                begin
                                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:327 =#
                                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:328 =#
                                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:53 =#
                                    𝛥mid[a[k], i, k] = Base.FastMath.add_fast(𝛥mid[a[k], i, k], 𝛥ℛ[i, k])
                                end
                                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                                nothing
                            end
                        end
                    end
                    $(Expr(:inbounds, :pop))
                    var"#402#val"
                end
            end
        end
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1095 =#
        nothing
    end
end
begin
    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1041 =#
    local function ∇𝒜𝒸𝓉!(::Type, 𝛥mid, 𝛥a, 𝛥ℛ::Zygote.Fill{𝒯}, ℛ, mid, a, 𝒶𝓍i, 𝒶𝓍k, ♻️ = nothing, 💀 = true) where 𝒯
        $(Expr(:meta, :inline))
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1041 =#
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1042 =#
        begin
            $(Expr(:inbounds, true))
            local var"#479#val" = begin
                𝛥ℛ_value = 𝛥ℛ.value
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1042 =#
                for k = 𝒶𝓍k
                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                    for i = 𝒶𝓍i
                        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                        begin
                            nothing
                            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                            begin
                                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:327 =#
                                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:328 =#
                                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:53 =#
                                𝛥mid[a[k], i, k] = Base.FastMath.add_fast(𝛥mid[a[k], i, k], 𝛥ℛ_value)
                            end
                            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                            nothing
                        end
                    end
                end
            end
            $(Expr(:inbounds, :pop))
            var"#479#val"
        end
    end
end
begin
    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1090 =#
    local function ∇𝒜𝒸𝓉!(::Type{<:Array{<:LoopVectorization.NativeTypes}}, 𝛥mid, 𝛥a, 𝛥ℛ::Zygote.Fill{𝒯}, ℛ, mid, a, 𝒶𝓍i, 𝒶𝓍k, ♻️ = nothing, 💀 = true) where 𝒯
        $(Expr(:meta, :inline))
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1090 =#
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1091 =#
        𝛥ℛ_value = 𝛥ℛ.value
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1092 =#
        begin
            #= logging.jl:370 =#
            let
                #= logging.jl:371 =#
                var"#403#level" = Base.CoreLogging.Info
                #= logging.jl:372 =#
                var"#404#std_level" = Base.CoreLogging.convert(Base.CoreLogging.LogLevel, var"#403#level")
                #= logging.jl:373 =#
                if var"#404#std_level" >= Base.CoreLogging._min_enabled_level[]
                    #= logging.jl:374 =#
                    var"#405#group" = :macro
                    #= logging.jl:375 =#
                    var"#406#_module" = Main
                    #= logging.jl:376 =#
                    var"#407#logger" = Base.CoreLogging.current_logger_for_env(var"#404#std_level", var"#405#group", var"#406#_module")
                    #= logging.jl:377 =#
                    if !(var"#407#logger" === Base.CoreLogging.nothing)
                        #= logging.jl:378 =#
                        var"#408#id" = 0xa3820e29d8c54d18
                        #= logging.jl:381 =#
                        if Base.CoreLogging._invoked_shouldlog(var"#407#logger", var"#403#level", var"#406#_module", var"#405#group", var"#408#id")
                            #= logging.jl:382 =#
                            var"#409#file" = "/Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl"
                            #= logging.jl:383 =#
                            var"#410#line" = 1083
                            #= logging.jl:384 =#
                            local var"#411#msg", var"#412#kwargs"
                            #= logging.jl:385 =#
                            begin
                                #= logging.jl:359 =#
                                try
                                    #= logging.jl:360 =#
                                    var"#411#msg" = "running LoopVectorization actor (gradient method for FillArrays)"
                                    #= logging.jl:361 =#
                                    var"#412#kwargs" = (; maxlog = 3)
                                    #= logging.jl:362 =#
                                    true
                                catch var"#425#err"
                                    #= logging.jl:364 =#
                                    Base.CoreLogging.logging_error(var"#407#logger", var"#403#level", var"#406#_module", var"#405#group", var"#408#id", var"#409#file", var"#410#line", var"#425#err", true)
                                    #= logging.jl:365 =#
                                    false
                                end
                            end && Base.CoreLogging.handle_message(var"#407#logger", var"#403#level", var"#411#msg", var"#406#_module", var"#405#group", var"#408#id", var"#409#file", var"#410#line"; var"#412#kwargs"...)
                        end
                    end
                end
                #= logging.jl:391 =#
                Base.CoreLogging.nothing
            end
        end
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1093 =#
        LoopVectorization.check_args(mid, a) || begin
            #= logging.jl:370 =#
            let
                #= logging.jl:371 =#
                var"#426#level" = Base.CoreLogging.Error
                #= logging.jl:372 =#
                var"#427#std_level" = Base.CoreLogging.convert(Base.CoreLogging.LogLevel, var"#426#level")
                #= logging.jl:373 =#
                if var"#427#std_level" >= Base.CoreLogging._min_enabled_level[]
                    #= logging.jl:374 =#
                    var"#428#group" = :macro
                    #= logging.jl:375 =#
                    var"#429#_module" = Main
                    #= logging.jl:376 =#
                    var"#430#logger" = Base.CoreLogging.current_logger_for_env(var"#427#std_level", var"#428#group", var"#429#_module")
                    #= logging.jl:377 =#
                    if !(var"#430#logger" === Base.CoreLogging.nothing)
                        #= logging.jl:378 =#
                        var"#431#id" = 0xa3820e29d8c54d18
                        #= logging.jl:381 =#
                        if Base.CoreLogging._invoked_shouldlog(var"#430#logger", var"#426#level", var"#429#_module", var"#428#group", var"#431#id")
                            #= logging.jl:382 =#
                            var"#432#file" = "/Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl"
                            #= logging.jl:383 =#
                            var"#433#line" = 1084
                            #= logging.jl:384 =#
                            local var"#434#msg", var"#435#kwargs"
                            #= logging.jl:385 =#
                            begin
                                #= logging.jl:359 =#
                                try
                                    #= logging.jl:360 =#
                                    var"#434#msg" = "rejected by LoopVectorization's check_args! (gradient method for FillArrays)"
                                    #= logging.jl:361 =#
                                    var"#435#kwargs" = (; maxlog = 3)
                                    #= logging.jl:362 =#
                                    true
                                catch var"#448#err"
                                    #= logging.jl:364 =#
                                    Base.CoreLogging.logging_error(var"#430#logger", var"#426#level", var"#429#_module", var"#428#group", var"#431#id", var"#432#file", var"#433#line", var"#448#err", true)
                                    #= logging.jl:365 =#
                                    false
                                end
                            end && Base.CoreLogging.handle_message(var"#430#logger", var"#426#level", var"#434#msg", var"#429#_module", var"#428#group", var"#431#id", var"#432#file", var"#433#line"; var"#435#kwargs"...)
                        end
                    end
                end
                #= logging.jl:391 =#
                Base.CoreLogging.nothing
            end
        end
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1094 =#
        begin
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                nothing
            end
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:53 =#
                nothing
            end
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:328 =#
                nothing
            end
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:327 =#
                nothing
            end
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                nothing
            end
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                nothing
            end
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                nothing
            end
            begin
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1094 =#
                nothing
            end
            var"###looprangek###1###" = LoopVectorization.canonicalize_range(begin
                                                                             $(Expr(:inbounds, true))
                                                                             local var"#449#val" = 𝒶𝓍k
                                                                             $(Expr(:inbounds, :pop))
                                                                             var"#449#val"
                                                                             end)
            var"###looplenk###2###" = ArrayInterface.static_length(var"###looprangek###1###")
            var"###k_loop_lower_bound###3###" = LoopVectorization.maybestaticfirst(var"###looprangek###1###")
            var"###k_loop_upper_bound###4###" = LoopVectorization.maybestaticlast(var"###looprangek###1###")
            var"###k_loop_step###5###" = LoopVectorization.static_step(var"###looprangek###1###")
            var"###looprangei###6###" = LoopVectorization.canonicalize_range(begin
                                                                             $(Expr(:inbounds, true))
                                                                             local var"#450#val" = 𝒶𝓍i
                                                                             $(Expr(:inbounds, :pop))
                                                                             var"#450#val"
                                                                             end)
            var"###loopleni###7###" = ArrayInterface.static_length(var"###looprangei###6###")
            var"###i_loop_lower_bound###8###" = LoopVectorization.maybestaticfirst(var"###looprangei###6###")
            var"###i_loop_upper_bound###9###" = LoopVectorization.maybestaticlast(var"###looprangei###6###")
            var"###i_loop_step###10###" = LoopVectorization.static_step(var"###looprangei###6###")
            if LoopVectorization.check_args(𝛥mid, a)
                (var"##vptr##_𝛥mid", var"#𝛥mid#preserve#buffer#") = LoopVectorization.stridedpointer_preserve(𝛥mid)
                (var"##vptr##_a", var"#a#preserve#buffer#") = LoopVectorization.stridedpointer_preserve(a)
                var"####grouped#strided#pointer####14###" = Core.getfield(LoopVectorization.grouped_strided_pointer((LoopVectorization.densewrapper(LoopVectorization.gespf1(var"##vptr##_a", (var"###k_loop_lower_bound###3###",)), a), LoopVectorization.densewrapper(LoopVectorization.gespf1(var"##vptr##_𝛥mid", (LoopVectorization.similardims(var"###looprangek###1###", static(0)), var"###i_loop_lower_bound###8###", var"###k_loop_lower_bound###3###")), 𝛥mid)), Val{()}()), 1)
                $(Expr(:gc_preserve, quote
                       var"##vargsym#323" = ((LoopVectorization.zerorangestart(var"###looprangek###1###"), LoopVectorization.zerorangestart(var"###looprangei###6###")), (var"####grouped#strided#pointer####14###", 𝛥ℛ_value))
                       var"##Tloopeltype##" = LoopVectorization.promote_type(LoopVectorization.eltype(𝛥mid), LoopVectorization.eltype(a))
                       var"##Wvecwidth##" = LoopVectorization.pick_vector_width(var"##Tloopeltype##")
                       LoopVectorization._turbo_!(LoopVectorization.avx_config_val(Val{(false, 0, 0, 0, false, 0x0000000000000001)}(), var"##Wvecwidth##"), Val{(:LoopVectorization, :getindex, LoopVectorization.OperationStruct(0x00000000000000000000000000000001, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, LoopVectorization.memload, 0x0001, 0x01), :LoopVectorization, :getindex, LoopVectorization.OperationStruct(0x00000000000000000000000000000121, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000001, 0x00000000000000000000000000000000, LoopVectorization.memload, 0x0002, 0x02), :LoopVectorization, :LOOPCONSTANTINSTRUCTION, LoopVectorization.OperationStruct(0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, LoopVectorization.constant, 0x0003, 0x00), :LoopVectorization, :add_fast, LoopVectorization.OperationStruct(0x00000000000000000000000000000012, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000020003, 0x00000000000000000000000000000000, LoopVectorization.compute, 0x0002, 0x00), :LoopVectorization, :setindex!, LoopVectorization.OperationStruct(0x00000000000000000000000000000121, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000040001, 0x00000000000000000000000000000000, LoopVectorization.memstore, 0x0004, 0x02))}(), Val{(LoopVectorization.ArrayRefStruct{:a, Symbol("##vptr##_a")}(0x00000000000000000000000000000001, 0x00000000000000000000000000000001, 0x00000000000000000000000000000000, 0x00000000000000000000000000000001), LoopVectorization.ArrayRefStruct{:𝛥mid, Symbol("##vptr##_𝛥mid")}(0x00000000000000000000000000020101, 0x00000000000000000000000000010201, 0x00000000000000000000000000000000, 0x00000000000000000000000000010101))}(), Val{(0, (), (3,), (), (), (), ())}(), Val{(:k, :i)}(), Base.Val(Base.typeof(var"##vargsym#323")), LoopVectorization.flatten_to_tuple(var"##vargsym#323")...)
                       end, Symbol("#a#preserve#buffer#"), Symbol("#𝛥mid#preserve#buffer#")))
                nothing
            else
                begin
                    #= logging.jl:370 =#
                    let
                        #= logging.jl:371 =#
                        var"#451#level" = Base.CoreLogging.Warn
                        #= logging.jl:372 =#
                        var"#452#std_level" = Base.CoreLogging.convert(Base.CoreLogging.LogLevel, var"#451#level")
                        #= logging.jl:373 =#
                        if var"#452#std_level" >= Base.CoreLogging._min_enabled_level[]
                            #= logging.jl:374 =#
                            var"#453#group" = :condense_loopset
                            #= logging.jl:375 =#
                            var"#454#_module" = Main
                            #= logging.jl:376 =#
                            var"#455#logger" = Base.CoreLogging.current_logger_for_env(var"#452#std_level", var"#453#group", var"#454#_module")
                            #= logging.jl:377 =#
                            if !(var"#455#logger" === Base.CoreLogging.nothing)
                                #= logging.jl:378 =#
                                var"#456#id" = :Main_1fbe9ed6
                                #= logging.jl:381 =#
                                if Base.CoreLogging._invoked_shouldlog(var"#455#logger", var"#451#level", var"#454#_module", var"#453#group", var"#456#id")
                                    #= logging.jl:382 =#
                                    var"#457#file" = "/Users/Matt/.julia/packages/LoopVectorization/x4G96/src/condense_loopset.jl"
                                    #= logging.jl:383 =#
                                    var"#458#line" = 825
                                    #= logging.jl:384 =#
                                    local var"#460#msg", var"#461#kwargs"
                                    #= logging.jl:385 =#
                                    begin
                                        #= logging.jl:346 =#
                                        let var"#459#err" = nothing
                                            #= logging.jl:347 =#
                                            if var"#459#err" === Base.CoreLogging.nothing
                                                #= logging.jl:348 =#
                                                var"#460#msg" = "#= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1094 =#:\n`LoopVectorization.check_args` on your inputs failed; running fallback `@inbounds @fastmath` loop instead.\nUse `warn_check_args=false`, e.g. `@turbo warn_check_args=false ...`, to disable this warning."
                                                #= logging.jl:349 =#
                                                var"#461#kwargs" = (; maxlog = 1)
                                                #= logging.jl:350 =#
                                                true
                                            else
                                                #= logging.jl:352 =#
                                                Base.CoreLogging.logging_error(var"#455#logger", var"#451#level", var"#454#_module", var"#453#group", var"#456#id", var"#457#file", var"#458#line", var"#459#err", false)
                                                #= logging.jl:353 =#
                                                false
                                            end
                                        end
                                    end && Base.CoreLogging.handle_message(var"#455#logger", var"#451#level", var"#460#msg", var"#454#_module", var"#453#group", var"#456#id", var"#457#file", var"#458#line"; var"#461#kwargs"...)
                                end
                            end
                        end
                        #= logging.jl:391 =#
                        Base.CoreLogging.nothing
                    end
                end
                begin
                    $(Expr(:inbounds, true))
                    local var"#476#val" = for k = 𝒶𝓍k
                        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                        for i = 𝒶𝓍i
                            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1212 =#
                            begin
                                nothing
                                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                                begin
                                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:327 =#
                                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:328 =#
                                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/symbolic.jl:53 =#
                                    𝛥mid[a[k], i, k] = Base.FastMath.add_fast(𝛥mid[a[k], i, k], 𝛥ℛ_value)
                                end
                                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1037 =#
                                nothing
                            end
                        end
                    end
                    $(Expr(:inbounds, :pop))
                    var"#476#val"
                end
            end
        end
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1095 =#
        nothing
    end
end
begin
    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1310 =#
    local ∇ℳ𝒶𝓀ℯ = let ∇𝒜𝒸𝓉! = ∇𝒜𝒸𝓉!
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1311 =#
        begin
            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1298 =#
            local function ∇ℳ𝒶𝓀ℯ(𝛥ℛ::AbstractArray{𝒯}, ℛ, mid, a) where 𝒯
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1298 =#
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1299 =#
                local 𝛥mid = fill!(similar(mid, Base.promote_type(eltype(mid), 𝒯)), 0)
                local 𝛥a = nothing
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1300 =#
                local 𝒶𝓍k = (axes)(mid, 3)
                (Tullio.linearindex)(a) == (axes)(mid, 3) || (throw)("range of index k must agree")
                local 𝒶𝓍i = (axes)(mid, 2)
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1301 =#
                (Tullio.∇threader)(∇𝒜𝒸𝓉!, (Tullio.storage_type)(𝛥mid, 𝛥a, mid, a), tuple(𝛥mid, 𝛥a, 𝛥ℛ, ℛ, mid, a), tuple(𝒶𝓍i, 𝒶𝓍k), tuple(), 262144)
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:1304 =#
                return (𝛥mid, 𝛥a)
            end
        end
    end
end
#= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:984 =#
begin
    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:975 =#
    let 𝒜𝒸𝓉! = 𝒜𝒸𝓉!
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:976 =#
        begin
            #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:969 =#
            local function ℳ𝒶𝓀ℯ(mid, a)
                $(Expr(:meta, :inline))
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:969 =#
                #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:970 =#
                local 𝒶𝓍k = (axes)(mid, 3)
                (Tullio.linearindex)(a) == (axes)(mid, 3) || (throw)("range of index k must agree")
                local 𝒶𝓍i = (axes)(mid, 2)
                begin
                    #= logging.jl:370 =#
                    let
                        #= logging.jl:371 =#
                        var"#480#level" = Base.CoreLogging.Info
                        #= logging.jl:372 =#
                        var"#481#std_level" = Base.CoreLogging.convert(Base.CoreLogging.LogLevel, var"#480#level")
                        #= logging.jl:373 =#
                        if var"#481#std_level" >= Base.CoreLogging._min_enabled_level[]
                            #= logging.jl:374 =#
                            var"#482#group" = :macro
                            #= logging.jl:375 =#
                            var"#483#_module" = Main
                            #= logging.jl:376 =#
                            var"#484#logger" = Base.CoreLogging.current_logger_for_env(var"#481#std_level", var"#482#group", var"#483#_module")
                            #= logging.jl:377 =#
                            if !(var"#484#logger" === Base.CoreLogging.nothing)
                                #= logging.jl:378 =#
                                var"#485#id" = :Main_c3261e42
                                #= logging.jl:381 =#
                                if Base.CoreLogging._invoked_shouldlog(var"#484#logger", var"#480#level", var"#483#_module", var"#482#group", var"#485#id")
                                    #= logging.jl:382 =#
                                    var"#486#file" = "/Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl"
                                    #= logging.jl:383 =#
                                    var"#487#line" = 718
                                    #= logging.jl:384 =#
                                    local var"#488#msg", var"#489#kwargs"
                                    #= logging.jl:385 =#
                                    begin
                                        #= logging.jl:359 =#
                                        try
                                            #= logging.jl:360 =#
                                            var"#488#msg" = "left index ranges"
                                            #= logging.jl:361 =#
                                            var"#489#kwargs" = (; i = (axes)(mid, 2), k = (axes)(mid, 3))
                                            #= logging.jl:362 =#
                                            true
                                        catch var"#502#err"
                                            #= logging.jl:364 =#
                                            Base.CoreLogging.logging_error(var"#484#logger", var"#480#level", var"#483#_module", var"#482#group", var"#485#id", var"#486#file", var"#487#line", var"#502#err", true)
                                            #= logging.jl:365 =#
                                            false
                                        end
                                    end && Base.CoreLogging.handle_message(var"#484#logger", var"#480#level", var"#488#msg", var"#483#_module", var"#482#group", var"#485#id", var"#486#file", var"#487#line"; var"#489#kwargs"...)
                                end
                            end
                        end
                        #= logging.jl:391 =#
                        Base.CoreLogging.nothing
                    end
                end
                local 𝓇𝒽𝓈(mid, a, i, k) = begin
                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:793 =#
                    identity(mid[a[k], i, k])
                end
                begin
                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:802 =#
                    local 𝒯1 = Core.Compiler.return_type(𝓇𝒽𝓈, (typeof)((mid, a, (first)(𝒶𝓍i), (first)(𝒶𝓍k))))
                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:803 =#
                    local 𝒯2 = if Base.isconcretetype(𝒯1)
                        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:804 =#
                        𝒯1
                    else
                        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:806 =#
                        begin
                            #= logging.jl:370 =#
                            let
                                #= logging.jl:371 =#
                                var"#503#level" = Base.CoreLogging.Warn
                                #= logging.jl:372 =#
                                var"#504#std_level" = Base.CoreLogging.convert(Base.CoreLogging.LogLevel, var"#503#level")
                                #= logging.jl:373 =#
                                if var"#504#std_level" >= Base.CoreLogging._min_enabled_level[]
                                    #= logging.jl:374 =#
                                    var"#505#group" = :macro
                                    #= logging.jl:375 =#
                                    var"#506#_module" = Main
                                    #= logging.jl:376 =#
                                    var"#507#logger" = Base.CoreLogging.current_logger_for_env(var"#504#std_level", var"#505#group", var"#506#_module")
                                    #= logging.jl:377 =#
                                    if !(var"#507#logger" === Base.CoreLogging.nothing)
                                        #= logging.jl:378 =#
                                        var"#508#id" = :Main_cc2cf7d3
                                        #= logging.jl:381 =#
                                        if Base.CoreLogging._invoked_shouldlog(var"#507#logger", var"#503#level", var"#506#_module", var"#505#group", var"#508#id")
                                            #= logging.jl:382 =#
                                            var"#509#file" = "/Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl"
                                            #= logging.jl:383 =#
                                            var"#510#line" = 800
                                            #= logging.jl:384 =#
                                            local var"#511#msg", var"#512#kwargs"
                                            #= logging.jl:385 =#
                                            begin
                                                #= logging.jl:328 =#
                                                var"#511#msg" = "unable to infer eltype from RHS"
                                                #= logging.jl:329 =#
                                                var"#512#kwargs" = (;)
                                                #= logging.jl:330 =#
                                                true
                                            end && Base.CoreLogging.handle_message(var"#507#logger", var"#503#level", var"#511#msg", var"#506#_module", var"#505#group", var"#508#id", var"#509#file", var"#510#line"; var"#512#kwargs"...)
                                        end
                                    end
                                end
                                #= logging.jl:391 =#
                                Base.CoreLogging.nothing
                            end
                        end
                        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:807 =#
                        (typeof)(𝓇𝒽𝓈(mid, a, (first)(𝒶𝓍i), (first)(𝒶𝓍k)))
                    end
                end
                local 𝒯3 = 𝒯2
                local 𝒯 = 𝒯3
                local ret = similar(parent(mid), 𝒯, tuple(𝒶𝓍i, 𝒶𝓍k))
                begin
                    (Tullio.threader)(𝒜𝒸𝓉!, (Tullio.storage_type)(ret, mid, a), ret, tuple(mid, a), tuple(𝒶𝓍i, 𝒶𝓍k), tuple(), +, 262144, nothing)
                    #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:961 =#
                    ret
                end
            end
        end
        #= /Users/Matt/.julia/packages/Tullio/qPZkO/src/macro.jl:977 =#
        ((Tullio.Eval)(ℳ𝒶𝓀ℯ, ∇ℳ𝒶𝓀ℯ))(mid, a)
    end
end
end
