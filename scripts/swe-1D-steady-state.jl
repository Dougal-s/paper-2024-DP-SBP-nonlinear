using DrWatson
@quickactivate :HyperbolicPDEs
using SummationByPartsOperators
using CairoMakie
using OrdinaryDiffEqSSPRK
using LaTeXStrings
using Base.Threads: @threads
using Base.Iterators: product
using UnPack: @unpack
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include(srcdir("PDEs", "swe.jl"))
include(srcdir("plotting-utils.jl"))
include(srcdir("makie-theme.jl"))

const tspan = (0.0, 20.0)
const xspan = (0.0, 25.0)
const Δt    = Δx -> 0.1Δx

const g = 9.81

b(x) = 8 < x < 12 ? 0.2 - 0.05 * (x - 10)^2 : 0.0

const exact = (
    (t, (x,)) -> 0.5 - b(x), # h
    (t, (x,)) -> 0.0,        # u
    (t, (x,)) -> b(x)        # b
)

# steady state with flow
# c::Float64 = 9
# using Polynomials
# function fluxexact(t,xs)
# 	p = xs .|> _ -> 1.0
# 	bt = b.(xs)
# 	h = xs .|> _ -> 0.0
# 	for i in eachindex(xs)
# 		u = Polynomial([p[i]^2 / (2g), 0.0, bt[i] - c/g, 1], :h)
# 		h[i] = maximum(roots(u))
# 	end
# 	(h, p)
# end

let # plot exact solution
    filename = joinpath("swe-1D", "steady-state", "immersed-bump-exact.pdf")
    print("Plotting '", filename, "'")

    fig = Figure(size = (360, 210))
    ax = Axis(fig[1, 1],
        ylabel            = "Surface level",
        xlabel            = L"x",
        xticks            = 0:5:25,
        xminorticks       = IntervalsBetween(5),
        xminorgridvisible = true,
        yticks            = 0.0:0.1:0.5,
        yminorticks       = IntervalsBetween(5),
        yminorgridvisible = true
    )
    xs = range(xspan..., 1024)
    lines!(ax, collect(xspan), x -> 0.5, label = L"$h+b$")
    lines!(ax, xs, b, label = L"$b$")
    axislegend(position = :rc)

    filepath = plotsdir(filename)
    wsave(filepath, fig)
    print("\n")
end

const schemes = [
    # (; label   = "Flux Form",
    #     pdeinfo = ShallowWaterFluxForm1D(g, nothing),
    # ),
    (; label   = "lin. stable DP DG/FD",
        pdeinfo = ShallowWaterFluxForm1D(FluxLaxFriedrichs(); g = g),
    ),
    (; label   = "DGSEM/SBP FD",
        pdeinfo = ShallowWaterSkewSym1D(; g = g),
    ),
    (; label   = "DP DG/FD",
        pdeinfo = ShallowWaterSkewSym1D(FluxEntropyStable(); g = g),
    )
]

const opt_list = vcat(
    dict_list(Dict(
        :order       => Derived(:nodes, Ns -> only(Ns) - 1),
        :nodes       => [[N] for N in 5:7],
        :elems       => [[16, 32, 64]],
        :deriv_type  => GlaubitzEtal2024(-0.1)
    )),
    dict_list(Dict(
        :order       => 4:7 |> collect,
        :nodes       => [[32, 64, 128, 256] .+ 1],
        :elems       => [[4]],
        :deriv_type  => Mattsson2017
    ))
)

for opts in opt_list
    println("Running with settings:")
    display(opts)
    println()

    @unpack nodes, elems = opts

    xgrids = Iterators.product(elems, nodes) |> collect
    errors = [fill(NaN64, length(xgrids)) for _ in eachindex(schemes)]
    runs = product(eachindex(schemes), eachindex(xgrids)) |> collect
    @threads for (scheme_idx, xgrid_idx) in runs
        # initialize
        @unpack pdeinfo, label = schemes[scheme_idx]
        nb, N = xgrids[xgrid_idx]

        mesh = CartesianMesh((xspan,), (nb,))
        cell = local_dp_operator(opts[:deriv_type], opts[:order], (N,))
        xs, fdop = couple_operators(mesh, cell)

        prob = let
            s0   = map(f -> f.(tspan[begin], xs), exact)
            pde! = semidiscretise(pdeinfo, xs, fdop)
            ODEProblem{true, SciMLBase.NoSpecialize}(
                pde!, from_primitive_vars(pdeinfo, s0), tspan)
        end

        # solve
        sol = solve(prob, SSPRK54();
            dt = Δt(step(xs)...),
            save_end = true,
            save_everystep = false,
            progress = true,
            progress_steps = 250,
            progress_id = Symbol(scheme_idx * length(xgrids) + xgrid_idx),
            progress_name = rpad(label, 20) * lpad(join(size(xs), "×"), 6)
        )

        # compute error
        final_exact = map(f -> f.(tspan[end], xs), exact)
        final_numerical = to_primitive_vars(pdeinfo, sol.u[end])
        errors[scheme_idx][xgrid_idx] = mapreduce(
            (hₙᵤₘ, h) -> abs((h - hₙᵤₘ) / h),
            max,
            final_numerical[1],
            final_exact[1]
        )
    end

    tag = savename(opts) * "-" * string(hash(opts); base = 60)
    tablepath = datadir("swe-1D", "steady-state", "error-$tag.tex")
    mkpath(dirname(tablepath))
    open(tablepath; write = true) do io
        print_error_table_latex(io, schemes, xgrids, errors)
    end
    print_error_table_ascii(schemes, xgrids, errors)
    println()
end
