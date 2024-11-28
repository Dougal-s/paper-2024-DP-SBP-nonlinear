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
const Ns     = [32, 64, 128, 256, 512]
const Δt     = Δx -> 0.1Δx
const xgrids = [bounded_range(xspan..., N) for N in Ns]
const Δxs    = step.(xgrids)
const g      = 1.0

# wavespeed = |u| + √(gh)
const exact = (
    (t, x) -> 2 + 0.3 * sinpi(2 * (x - t)), # h
    (t, x) -> 2 + 0.3 * sinpi(2 * (x + t)), # u
    (t, x) -> 0.0                           # b
)

const schemes = [
    # (; label = "Flux Form",
    #     pde = ShallowWaterFluxForm1D(SourceMMS(exact); g = g),
    #     marker = :cross
    # ),
    (; label   = "linearly stable",
        pdeinfo = ShallowWaterFluxForm1D(SourceMMS(exact), FluxLaxFriedrichs(); g = g),
        marker  = :circle
    ),
    (; label   = "entropy conserving",
        pdeinfo = ShallowWaterSkewSym1D(SourceMMS(exact); g = g),
        marker  = :rect
    ),
    (; label   = "entropy stable",
        pdeinfo = ShallowWaterSkewSym1D(SourceMMS(exact), FluxEntropyStable(); g = g),
        marker  = :utriangle
    )
]

const opt_list = dict_list(Dict(
    :deriv_order => [4, 5, 6, 7],
    :deriv_type => Mattsson2017,
    :deriv_bc => PeriodicSAT(),
    :dp_operator => Derived(
        [:deriv_order, :deriv_type, :deriv_bc],
        (p, type, bc) -> (xs -> dp_operator(type, bc, p, xs))
    )
))

for opts in opt_list
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
            progress_name = rpad(sprint(print, scheme[:label]), 20)
        )

        # compute error
        exactf = map(f -> f.(Ref(tspan[end]), xs), exact)
        sf = to_primitive_vars(pdeinfo, sol.u[end])
        errors[scheme_idx][xgrid_idx] = √(energy_norm(sf .- exactf, H) do (δh, δu, δb)
            return δh^2
        end)
    end

    tag = savename(opts) * "-" * string(hash(opts); base = 60)
    plot_convergence(
        "swe-1D/MMS/convergence-$tag.pdf",
        get.(schemes, :label, ""),
        errors,
        Δxs;
        figargs = (; size = (280, 200)),
        markers = getfield.(schemes, (:marker,)),
        legendname = joinpath("swe-1D", "MMS", "convergence-legend-$tag.pdf")
    )
    tablepath = datadir("swe-1D", "MMS", "convergence-$tag.tex")
    mkpath(dirname(tablepath))
    open(tablepath; write = true) do io
        print_convergence_table_latex(io, schemes, xgrids, errors)
    end
    print_convergence_table_ascii(schemes, xgrids, errors)
end
