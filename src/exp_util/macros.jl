module Macros
using MacroTools: prewalk, postwalk, @capture
import Markdown: Markdown, MD, @md_str
import TOML
greet() = print("Hello World!")

struct HelpStr
    str::String
end

# export @help_str
macro help_str(str)
end

# export @info_str
macro info_str(str)
end

function get_args_and_order(default_config)

    arg_order = String[]
    args = Expr[]
    postwalk(default_config) do ex
        chk = @capture(ex, k_ => v_)
        # @show k
        if chk
            push!(arg_order, k)
            push!(args, :($k=>$v))
        end
        ex
    end
    args, arg_order
    
end

function get_help_str(default_config, __module__)
    start_str = "# Automatically generated docs for $(__module__) config."
    help_str_strg = HelpStr[
        HelpStr(start_str)
    ]
    postwalk(default_config) do expr
        expr_str = string(expr)
        if length(expr_str) > 5 && (expr_str[1:5] == "help\"" || expr_str[1:5] == "info\"")
            push!(help_str_strg, HelpStr(string(expr)[6:end-1]))
        end
        expr
    end
    md_strs = [Markdown.parse(hs.str) for hs in help_str_strg]
    join(md_strs, "\n")
end


macro generate_config_funcs(default_config)
    # println(default_config)

    func_name = :default_config
    help_func_name = :help
    create_toml_func_name = :create_toml_template
    mdstrings = String[]
    src_file = relpath(String(__source__.file))

    
    docs = get_help_str(default_config, __module__)
    args, arg_order = get_args_and_order(default_config)
    # @show args

    create_toml_docs = """
        create_toml_template(save_file=nothing; database=false)

    Used to create toml template. If save_file is nothing just return toml string. 
    If database is true, then generate using mysql backend otherwise generate using file backend.
    """
    quote
        @doc $(docs)
        function $(esc(func_name))()
            Dict{String, Any}(
                $(args...)
            )

        end

        function $(esc(help_func_name))()
            local docs = Markdown.parse($(docs))
            # InteractiveUtils.less(docs)
            display(docs)
        end

        function $(esc(create_toml_func_name))(save_file=nothing; database=false)
            local ao = filter((str)->str!="save_dir", $arg_order)
            cnfg = $(esc(func_name))()
            cnfg_filt = filter((p)->p.first != "save_dir", cnfg)

            mod = $__module__

            save_info = if database
                """
                save_backend="mysql" # mysql only database backend supported
                database="test_iter" # Database name
                save_dir="$(cnfg["save_dir"])" # Directory name for exceptions, settings, and more!"""
            else
                """
                save_backend="file" # file saving mode
                file_type = "jld2" # using JLD2 as save type
                save_dir="$(cnfg["save_dir"])" # save location"""
            end
            
            toml_str = """
            Config generated automatically from default_config. When you have finished 
            making changes to this config for your experiment comment out this line.

            [config]
            $(save_info)
            exp_file = "$($src_file)"
            exp_module_name = "$(mod)"
            exp_func_name = "main_experiment"
            arg_iter_type = "iter"

            [static_args]
            """
            buf = IOBuffer()

            TOML.print(buf,
                cnfg_filt, sorted=true, by=(str)->findfirst((strinner)->str==strinner, ao)
                       )
            toml_str *= String(take!(buf))

            toml_str *= """\n[sweep_args]
            # Put args to sweep over here.
            """

            if save_file === nothing
                toml_str
            else
                open(save_file, "w") do io
                    write(io, toml_str)
                end
            end
            
        end
    end
end



end # module



