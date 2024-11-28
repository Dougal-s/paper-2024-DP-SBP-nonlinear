using DrWatson
@quickactivate :HyperbolicPDEs
using SummationByPartsOperators
using CairoMakie
using OrdinaryDiffEqSSPRK
using LaTeXStrings
using Base.Threads: @threads
using Base.Iterators: product
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include(srcdir("PDEs", "swe.jl"))
include(srcdir("plotting-utils.jl"))
include(srcdir("makie-theme.jl"))
set_theme!(hpdes_makie_theme)

const tspan  = (0.0, 20.0)
const xspan  = (0.0, 25.0)
const Ns     = [32, 64, 128, 256]
const Δt     = Δx -> 0.1Δx
const xgrids = [bounded_range(xspan..., N) for N in Ns]
const Δxs    = step.(xgrids)
const g      = 9.81

b(x) = 8 < x < 12 ? 0.2 - 0.05 * (x - 10)^2 : 0.0

const exact = (
    (t, x) -> 0.5 - b(x), # h
    (t, x) -> 0.0,        # u
    (t, x) -> b(x)        # b
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
    filename = "swe-1D/steady-state/immersed-bump-exact.pdf"
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
    #     marker  = :cross
    # ),
    (; label   = "linearly stable",
        pdeinfo = ShallowWaterFluxForm1D(FluxLaxFriedrichs(); g = g),
        marker  = :circle
    ),
    (; label   = "entropy conserving",
        pdeinfo = ShallowWaterSkewSym1D(; g = g),
        marker  = :rect
    ),
    (; label   = "entropy stable",
        pdeinfo = ShallowWaterSkewSym1D(FluxEntropyStable(); g = g),
        marker  = :utriangle
    )
]

const opt_list = dict_list(Dict(
    :deriv_order => [4, 5, 6, 7],
    :deriv_type => Mattsson2017,
    :deriv_bc => PeriodicSAT()
))

for opts in opt_list
    println("Running with settings:")
    display(opts)
    println()

    errors = [fill(NaN64, length(xgrids)) for _ in eachindex(schemes)]
    runs = product(eachindex(schemes), eachindex(xgrids)) |> collect
    @threads for (scheme_idx, xgrid_idx) in runs
        # initialize
        scheme  = schemes[scheme_idx]
        xs      = xgrids[xgrid_idx]
        pdeinfo = scheme[:pdeinfo]

        s0        = map(f -> f.(Ref(tspan[begin]), xs), exact)
        D₋, D₊, H = dp_operator(opts[:deriv_type], opts[:deriv_bc], opts[:deriv_order], xs)
        pde!      = semidiscretise(pdeinfo, xs, (D₊, D₋))
        prob      = ODEProblem(pde!, from_primitive_vars(pdeinfo, s0), tspan)

        # solve
        sol = solve(prob, SSPRK54();
            dt = Δt(step(xs)),
            save_end = true,
            save_everystep = false,
            progress = true,
            progress_steps = 250,
            progress_id = Symbol(scheme_idx * length(xgrids) + xgrid_idx),
            progress_name = rpad("$(scheme[:label]) N=$(length(xs))", 30)
        )

        # compute error
        exactf = map(f -> f.(Ref(tspan[end]), xs), exact)
        sf = to_primitive_vars(pdeinfo, sol.u[end])
        errors[scheme_idx][xgrid_idx] = mapreduce(max, sf[1] - exactf[1], sf[1]) do δh, h
            return abs(δh / h)
        end
    end

    tag = savename(opts) * "-" * string(hash(opts); base = 60)
    tablepath = datadir("swe-1D", "steady-state", "convergence-$tag.tex")
    mkpath(dirname(tablepath))
    open(tablepath; write = true) do io
        print_error_table_latex(io, schemes, xgrids, errors)
    end
    print_error_table_ascii(schemes, xgrids, errors)
    println()
end
