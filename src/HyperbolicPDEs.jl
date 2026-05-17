module HyperbolicPDEs
export approxless

export display_settings
export print_error_table_latex, print_error_table_ascii
export print_convergence_table_latex, print_convergence_table_ascii

using LinearAlgebra
using Printf
using StaticArrays

import DrWatson

DrWatson.default_allowed(::Dict) = (
    DrWatson.default_allowed(nothing)...,
    Type,
    GlaubitzEtal2024
)

include("mesh.jl")
include("operators/multiblock.jl")
include("operators/upwind-LGL.jl")
include("operators/firstorder.jl")
include("operators/dissipation.jl")
include("operators/measure.jl")

include("io/makie.jl")

include("pde-utils.jl")
include("symutils.jl")

approxless(a, b) = a ≈ b || a < b

##########
# Output #
##########

function print_error_table_latex(pdes, xgrids, error)
    print_error_table_latex(stdout, pdes, xgrids, error)
end
function print_error_table_latex(io::IO, pdes, xgrids, error)
    Ks  = getindex.(xgrids, 1)
    Ns  = getindex.(xgrids, 2)

    println(io, "\\begin{tabular}{@{}rr*{$(length(pdes))}{r}@{}}")
    println(io, "\t\\toprule")
    print(io, "\tK & N")
    for pde in pdes
        print(io, " & $(pde.label)")
    end
    println(io, " \\\\")
    println(io, "\t\\midrule")

    for i in eachindex(xgrids)
        @printf io "\t%4d & %4d" Ks[i] Ns[i]
        for j in eachindex(pdes)
            @printf io " & \\num{%.2e}" error[j][i]
        end
        println(io, " \\\\")
    end
    println(io, "\t\\bottomrule")
    println(io, "\\end{tabular}")
end

function print_error_table_ascii(pdes, xgrids, error)
    print_error_table_ascii(stdout, pdes, xgrids, error)
end
function print_error_table_ascii(io::IO, pdes, xgrids, error)
    Ks  = getindex.(xgrids, 1)
    Ns  = getindex.(xgrids, 2)

    @printf io " %3s  %3s │" "K" "N"
    for pde in pdes
        @printf io "  %10s " pde.label
    end
    @printf io "\n"

    @printf io " %s  %s │" "―"^3 "―"^3
    for pde in pdes
        @printf io "  %s " "―"^max(length(pde.label), 10)
    end
    @printf io "\n"

    for i in eachindex(xgrids)
        @printf io "%4d %4d │" Ks[i] Ns[i]
        for j in eachindex(pdes)
            col_width = length(pdes[j].label)
            @printf io "  %10s " lpad(@sprintf("%.3e", error[j][i]), col_width)
        end
        @printf io "\n"
    end
end

function print_convergence_table_latex(pdes, xs, error)
    print_convergence_table_latex(stdout, pdes, xs, error)
end

function print_convergence_table_latex(io::IO, pdes, xgrids, error)
    Ks  = getindex.(xgrids, 1)
    Ns  = getindex.(xgrids, 2)
    Δxs = @. 1 / (Ks * (Ns - 1)) # only needs to be proportional

    println(io, "\\begin{tabular}{@{}rr*{$(length(pdes))}{rr}@{}}")

    println(io, "\t\\toprule")
    println(io, "\t&")
    for pde in pdes
        print(io, "\t& \\multicolumn{2}{r}{$(pde.label)}")
    end
    println(io, " \\\\")

    for i in eachindex(pdes)
        print(io, "\t \\cmidrule(lr){$(2i+1)-$(2i+2)}")
    end
    println(io)

    print(io, "\tK & N")
    for _ in pdes
        print(io, " & Error & EOC")
    end
    println(io, " \\\\")
    println(io, "\t\\midrule")

    for i in eachindex(xgrids)
        @printf io "\t%4d & %4d" Ks[i] Ns[i]
        for j in eachindex(pdes)
            @printf io " & \\num{%.2e}" error[j][i]
            if i > 1
                estimated_order = log(error[j][i] / error[j][i - 1]) /
                                  log(Δxs[i] / Δxs[i - 1])
                @printf io " & %6.2f " estimated_order
            else
                @printf io " & %6s " "-"
            end
        end
        println(io, " \\\\")
    end
    println(io, "\t\\bottomrule")
    println(io, "\\end{tabular}")
end

function print_convergence_table_ascii(pdes, xs, error)
    print_convergence_table_ascii(stdout, pdes, xs, error)
end

function print_convergence_table_ascii(io::IO, pdes, xgrids, error)
    Ks  = getindex.(xgrids, 1)
    Ns  = getindex.(xgrids, 2)
    Δxs = @. 1 / (Ks * (Ns - 1)) # only needs to be proportional

    @printf io " %3s  %3s │" "" ""
    for pde in pdes
        @printf io "  %22s " pde.label
    end
    println(io)

    @printf io " %3s  %3s │" "K" "N"
    for pde in pdes
        @printf io "  %10s %11s " "Error" "Est. Order"
    end
    println(io)

    @printf io " %3s  %3s │" "―"^3 "―"^3
    for pde in pdes
        @printf io "  %10s %11s " "―"^10 "―"^11
    end
    println(io)

    for i in eachindex(xgrids)
        @printf io "%4d %4d │" Ks[i] Ns[i]
        for j in eachindex(pdes)
            @printf io "  %10.3e" error[j][i]
            if i > 1
                estimated_order = log(error[j][i] / error[j][i - 1]) /
                                  log(Δxs[i] / Δxs[i - 1])
                @printf io " %11.4f " estimated_order
            else
                @printf io " %11s " "-"
            end
        end
        println(io)
    end
end

_keywidth(k, d) = length(string(k))
function _display_keyval(depth, k, v, keywidth; prefix="")
    printstyled(prefix; color=:white)
    kstr = string(k)
    print(kstr)
    printstyled(": ", "…"^(keywidth - length(kstr) - length(prefix)); color=:white)
    println(" ", repr(v; context=:compact=>true))
end

function _keywidth(k, d::Union{AbstractDict, NamedTuple})
    return max(length(string(k)) - 1, 2 + maximum(splat(_keywidth), pairs(d); init=0))
end
function _display_keyval(depth, k, d::Union{AbstractDict, NamedTuple}, keywidth; prefix="")
    printstyled(prefix; color=:white)
    print(k)
    printstyled(":\n"; color=:white)
    for (k,v) in pairs(d)
        nextprefix = get(["╎", "┆", "┊"], depth, "⦙")
        _display_keyval(depth + 1, k, v, keywidth; prefix=prefix * nextprefix * " ")
    end
end

function display_settings(d)
    keywidth = 3 + maximum(splat(_keywidth), pairs(d))
    for (k,v) in pairs(d)
        _display_keyval(1, k, v, keywidth; prefix = "│ ")
    end
end

end
