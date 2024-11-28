module HyperbolicPDEs
export approxless,
       meshgrid,
       meshgrid!,
       energy_norm,
       periodic_range,
       bounded_range

export print_error_table_latex, print_error_table_ascii
export print_convergence_table_latex, print_convergence_table_ascii

using LinearAlgebra
using Printf
using Reexport
using StaticArrays

include("SbpOperators.jl")
@reexport using .SbpOperators

approxless(a, b) = a < b || a ≈ b

periodic_range(start, stop, length) = start .+ (stop - start) / length * (0:(length - 1))
bounded_range(start, stop, length)  = range(start, stop, length)

"""
    meshgrid!(grid::AbstractArray, f, xs...)

Fills `grid` by calling `f` with the coordinate of each grid point.
"""
function meshgrid!(f, grid::AbstractArray, xs...)
    grid .= Iterators.map(Base.splat(f), Iterators.product(xs...))
    return grid
end

"""
    meshgrid(f, xs...)

Like `meshgrid!` but constructs a new array instead of updating inplace.
"""
meshgrid(f, xs...) = meshgrid!(f, Array{Float64}(undef, length.(xs)...), xs...)

"""
    energy_norm(f, state, Hs...)

Approximates `∫ f(state[x]) dx` using a weighted sum where Hs defines the
quadrature weights along each dimension.
"""
function energy_norm(f, state, H::Diagonal)
    mapreduce(*, +, parent(H), (f(s) for s in zip(state...)))
end,
function energy_norm(f, state, Hs...)
    initH..., tailH = Hs
    lastdim = length(Hs)
    return sum((tailH[i, i] * energy_norm(f, selectdim.(state, lastdim, i), initH...)
    for i in axes(state[1], lastdim)))
end

##########
# Output #
##########

function print_error_table_latex(pdes, xs, error)
    print_error_table_latex(stdout, pdes, xs, error)
end
function print_error_table_latex(io::IO, pdes, xs, error)
    println(io, "\\begin{tabular}{r|*{$(length(pdes))}{r}}")
    print(io, "\tN")
    for pde in pdes
        print(io, " & $(pde.label)")
    end
    println(io, "\\\\")
    println(io, "\t\\hline")

    for i in eachindex(xs)
        @printf io "\t%4d" length(xs[i])
        for j in eachindex(pdes)
            @printf io " & %10.2e" error[j][i]
        end
        println(io, " \\\\")
    end
    println(io, "\\end{tabular}")
end

function print_error_table_ascii(pdes, xs, error)
    print_error_table_ascii(stdout, pdes, xs, error)
end
function print_error_table_ascii(io::IO, pdes, xs, error)
    @printf io "%5s │" "N"
    for pde in pdes
        @printf io "  %10s " pde.label
    end
    @printf io "\n"

    @printf io " %s │" "―"^4
    for pde in pdes
        @printf io "  %s " "―"^max(length(pde.label), 10)
    end
    @printf io "\n"

    for i in eachindex(xs)
        @printf io "%5d │" length(xs[i])
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

function print_convergence_table_latex(io::IO, pdes, xs, error)
    println(io, "\\begin{tabular}{r*{$(length(pdes))}{|rr}|}")
    for pde in pdes
        print(io, "\t& \\multicolumn{2}{r|}{$(pde.label)}")
    end
    println(io, "\\\\")

    print(io, "\tN")
    for _ in pdes
        print(io, " & Error & EOC")
    end
    println(io, "\\\\")
    println(io, "\t\\hline")

    Δxs = step.(xs)
    for i in eachindex(xs)
        @printf io "\t%4d" length(xs[i])
        for j in eachindex(pdes)
            @printf io " & %10.2e" error[j][i]
            if i > 1
                estimated_order = log(error[j][i] / error[j][i - 1]) /
                                  log(Δxs[i] / Δxs[i - 1])
                @printf io " & %11.2f " estimated_order
            else
                @printf io " & %11s " "-"
            end
        end
        println(io, " \\\\")
    end
    println(io, "\\end{tabular}")
end

function print_convergence_table_ascii(pdes, xs, error)
    print_convergence_table_ascii(stdout, pdes, xs, error)
end

function print_convergence_table_ascii(io::IO, pdes, xs, error)
    @printf io "%4s │" ""
    for pde in pdes
        @printf io "  %22s " pde.label
    end
    println(io)

    @printf io "%4s │" "N"
    for pde in pdes
        @printf io "  %10s %11s " "Error" "Est. Order"
    end
    println(io)

    @printf io " %3s │" "―"^3
    for pde in pdes
        @printf io "  %10s %11s " "―"^10 "―"^11
    end
    println(io)

    Δxs = step.(xs)
    for i in eachindex(xs)
        @printf io "%4d │" length(xs[i])
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

end
