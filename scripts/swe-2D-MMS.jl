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

const tspan  = (0.0, 2.0)
const xspan  = (-1.0, 1.0)
const Ns     = [32, 64, 128, 256]
const Δt     = Δx -> 0.05Δx
const xgrids = [bounded_range(xspan..., N) for N in Ns]
const Δxs    = step.(xgrids)
const f      = 0.0
const g      = 1.0

# wavespeed = |u| + √(gh), |v| + √(gh)
const exact = (
    (t, x, y) -> 2 + 0.2 * sinpi(2 * (x - t)) * sinpi(2 * (y - t)), # h
    (t, x, y) -> 2 + 0.2 * sinpi(2 * (x + t)) * sinpi(2 * (y + t)), # u
    (t, x, y) -> 2 + 0.2 * sinpi(2 * (x + t)) * sinpi(2 * (y + t)), # v
    (t, x, y) -> 0.0                                                # b
)

const schemes = [
    # (; label = "Flux Form",
    #     pde = ShallowWaterFluxForm1D(SourceMMS(exact); f = f, g = g),
    #     marker = :cross
    # ),
    (; label   = "linearly stable",
        pdeinfo = ShallowWaterFluxForm2D(SourceMMS(exact), FluxLaxFriedrichs(); f = f, g = g),
        marker  = :circle
    ),
    (; label   = "entropy conserving",
        pdeinfo = ShallowWaterSkewSym2D(SourceMMS(exact); f = f, g = g),
        marker  = :rect
    ),
    (; label   = "entropy stable",
        pdeinfo = ShallowWaterSkewSym2D(SourceMMS(exact), FluxEntropyStable(); f = f, g = g),
        marker  = :utriangle
    )
]

const opt_list = dict_list(Dict(
    :deriv_order => [4, 5, 6, 7],
    :deriv_type => Mattsson2017,
    :boundary => PeriodicSAT()
))

for opts in opt_list
    errors = [fill(NaN64, length(xgrids)) for _ in eachindex(schemes)]
    runs = product(eachindex(schemes), eachindex(xgrids)) |> collect
    @threads for (scheme_idx, xgrid_idx) in runs
        # initialize
        scheme  = schemes[scheme_idx]
        xs      = xgrids[xgrid_idx]
        ys      = xs
        pdeinfo = scheme[:pdeinfo]

        s0           = map(f -> meshgrid((x, y) -> f(tspan[begin], x, y), xs, ys), exact)
        Dx₋, Dx₊, Hx = dp_operator(opts[:deriv_type], opts[:boundary], opts[:deriv_order], xs)
        Dy₋, Dy₊, Hy = dp_operator(opts[:deriv_type], opts[:boundary], opts[:deriv_order], ys)

        pde! = semidiscretise(pdeinfo, (xs, ys), ((Dx₊, Dx₋), (Dy₊, Dy₋)))

        prob = ODEProblem(pde!, from_primitive_vars(pdeinfo, s0), tspan)

        # solve
        sol = solve(prob, SSPRK54();
            dt = Δt(step(xs)),
            save_end = true,
            save_everystep = false,
            progress = true,
            progress_steps = 100,
            progress_id = Symbol(scheme_idx * length(xgrids) + xgrid_idx),
            progress_name = rpad(scheme[:label], 20) *
                            rpad("$(length(xs))×$(length(ys))", 8)
        )

        # compute error
        exactf = map(f -> meshgrid((x, y) -> f(tspan[end], x, y), xs, ys), exact)
        sf = to_primitive_vars(pdeinfo, sol.u[end])
        errors[scheme_idx][xgrid_idx] = √(energy_norm(
            sf .- exactf, Hx, Hy) do (δh, δu, δv, δb)
            return δh^2
        end)
    end

    tag = savename(opts) * "-" * string(hash(opts); base = 60)
    plot_convergence(
        "swe-2D/MMS/convergence-$tag.pdf",
        get.(schemes, :label, ""),
        errors,
        Δxs;
        figargs = (; size = (280, 200)),
        markers = getfield.(schemes, (:marker,)),
        legendname = joinpath("swe-2D", "MMS", "convergence-legend-$tag.pdf")
    )
    tablepath = datadir("swe-2D/MMS/convergence-$tag.tex")
    mkpath(dirname(tablepath))
    open(tablepath; write = true) do io
        print_convergence_table_latex(io, schemes, xgrids, errors)
    end
    print_convergence_table_ascii(schemes, xgrids, errors)
end
