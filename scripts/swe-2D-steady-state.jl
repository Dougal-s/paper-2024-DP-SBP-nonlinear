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

include(srcdir("PDEs/swe.jl"))
include(srcdir("plotting-utils.jl"))
include(srcdir("makie-theme.jl"))
set_theme!(hpdes_makie_theme)

const tspan  = (0.0, 10.0)
const xspan  = (0.0, 25.0)
const Ns     = [32, 64, 128]
const Δt     = Δx -> 0.05 * Δx
const xgrids = [bounded_range(xspan..., N) for N in Ns]
const Δxs    = step.(xgrids)
const f      = 0.0
const g      = 9.81

b(x, y) = (x - 10)^2 + (y - 10)^2 < 4 ? 0.2 - 0.05 * ((x - 10)^2 + (y - 10)^2) : 0.0

const exact = (
    (t, x, y) -> 0.5 - b(x, y), # h
    (t, x, y) -> 0.0,           # u
    (t, x, y) -> 0.0,           # v
    (t, x, y) -> b(x, y)        # b
)

let # plot exact solution
    filename = "swe-2D/steady-state/immersed-bump-exact.png"
    print("Plotting '", filename, "'")

    fig = Figure(size = (480, 300))
    ax  = Axis3(fig[1, 1], zlabel = "Surface level")
    xs  = range(xspan..., 512)
    ys  = range(xspan..., 512)

    surface!(ax, xs, ys, (x, y) -> 0.5; label = L"$h+b$")
    surface!(ax, xs, ys, b; label = L"$b$")

    filepath = plotsdir(filename)
    wsave(filepath, fig; pt_per_unit = 1, px_per_unit = 4)
    print("\n")
end

const schemes = [
    (; label   = "Flux Form",
        pdeinfo = ShallowWaterFluxForm2D(; f = f, g = g),
        marker  = :cross
    ),
    (; label   = "Lax-Fried.",
        pdeinfo = ShallowWaterFluxForm2D(FluxLaxFriedrichs(); f = f, g = g),
        marker  = :circle
    ),
    (; label   = L"Skew Symm. $(\gamma=0)$",
        pdeinfo = ShallowWaterSkewSym2D(; f = f, g = g),
        marker  = :rect
    ),
    (; label   = L"Skew Symm. $(\gamma>0)$",
        pdeinfo = ShallowWaterSkewSym2D(FluxEntropyStable(); f = f, g = g),
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
    @sync @threads for (scheme_idx, xgrid_idx) in runs
        # initialize
        scheme  = schemes[scheme_idx]
        xs      = xgrids[xgrid_idx]
        ys      = xgrids[xgrid_idx]
        pdeinfo = scheme[:pdeinfo]

        s0           = map(f -> meshgrid((x, y) -> f(tspan[begin], x, y), xs, ys), exact)
        Dx₋, Dx₊, Hx = dp_operator(opts[:deriv_type], opts[:deriv_bc], opts[:deriv_order], xs)
        Dy₋, Dy₊, Hy = dp_operator(opts[:deriv_type], opts[:deriv_bc], opts[:deriv_order], ys)

        pde! = semidiscretise(
            pdeinfo,
            (xs, ys),
            ((Dx₊, Dx₋), (Dy₊, Dy₋))
        )

        prob = ODEProblem(pde!, from_primitive_vars(pdeinfo, s0), tspan)

        # solve
        sol = solve(prob, SSPRK54();
            adaptive = false,
            dt = Δt(step(xs)),
            save_end = true,
            save_everystep = false,
            progress = true,
            progress_steps = 250,
            progress_id = Symbol(scheme_idx * length(xgrids) + xgrid_idx),
            progress_name = rpad("$(scheme[:label]) N=$(length(xs))", 30)
        )

        # compute error
        exactf = map(f -> meshgrid((x, y) -> f(tspan[end], x, y), xs, ys), exact)
        sf = to_primitive_vars(pdeinfo, sol.u[end])
        errors[scheme_idx][xgrid_idx] = mapreduce(max, sf[1] - exactf[1], sf[1]) do δh, h
            return abs(δh / h)
        end
    end

    tag = savename(opts) * "-" * string(hash(opts); base = 60)
    tablepath = datadir("swe-2D/steady-state/convergence-$tag.tex")
    mkpath(dirname(tablepath))
    open(tablepath; write = true) do io
        print_error_table_latex(io, schemes, xgrids, errors)
    end
    print_error_table_ascii(schemes, xgrids, errors)
    println()
end
