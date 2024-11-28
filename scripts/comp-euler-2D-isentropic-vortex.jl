using DrWatson
@quickactivate :HyperbolicPDEs
using SummationByPartsOperators
using CairoMakie
using OrdinaryDiffEqSSPRK
using DiffEqCallbacks
using JLD2
using WriteVTK
using LaTeXStrings
using Base.Threads: @threads
using Base.Iterators: product
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include(srcdir("PDEs", "compressible-euler.jl"))
include(srcdir("PDEs", "compressible-euler-analytic.jl"))
include(srcdir("plotting-utils.jl"))
include(srcdir("makie-theme.jl"))
set_theme!(hpdes_makie_theme)

const tspan  = (0.0, 16.0)
const save_times = [0.0, 4.0, 8.0, 12.0, 16.0]
const xspan  = (-8.0, 8.0)
const Ns     = [32, 48, 64, 96, 128, 256]
const Δt     = Δx -> 0.1Δx
const xgrids = [bounded_range(xspan..., N) for N in Ns]
const Δxs    = step.(xgrids)
const γ      = 1.4

exact = IsentropicVortex(γ = γ, xrange = xspan, yrange = xspan)

const schemes = [
    # (; label = "Flux Form",
    #     pdeinfo = CompEulerFluxForm2D(γ = γ, flux_splitting = nothing),
    #     marker = :cross
    # ),
    (; label   = "linearly stable",
         pdeinfo = CompEulerFluxForm2D(γ = γ, flux_splitting = FluxLaxFriedrichs()),
        marker  = :circle,
        color   = Makie.wong_colors()[1]
    ),
    (; label   = "entropy conserving",
        pdeinfo = NordstromCompEuler2D(γ = γ, flux_splitting = nothing),
        marker  = :rect,
        color   = Makie.wong_colors()[2]
    ),
    (; label   = "entropy stable",
        pdeinfo = NordstromCompEuler2D(γ = γ, flux_splitting = FluxEntropyStable()),
        marker  = :utriangle,
        color   = Makie.wong_colors()[3]
    ),
    # (; label   = "Van Leer-Hanel",
    #     pdeinfo = VanLeerHanelCompEuler2D(γ = γ),
    #     marker  = :rect,
    #     color   = Makie.wong_colors()[5]
    # )
]

const opt_list = dict_list(Dict(
    :deriv_order => [7, 6, 5, 4],
    :deriv_type => Mattsson2017,
    :bc => PeriodicSAT()
))

# save exact solution
let
    println("generating exact solution")
    odir = joinpath("comp-euler-2D", "isentropic-vortex", "exact")
    mkpath(datadir(odir))
    xs = bounded_range(xspan..., 1024)
    for t in save_times
        state = exact(t, xs, xs)
        vtk_grid(datadir(odir, "time_$t.vti"), xs, xs) do vtk
            vtk["ϱ"] = state[1]
            vtk["u"] = @views (state[2], state[3])
            vtk["p"] = state[4]
        end
    end
end

# run solvers
for opts in opt_list
    odir = joinpath("comp-euler-2D", "isentropic-vortex")
    tag = savename(opts) * "-" * string(hash(opts); base = 60)

    runs = product(eachindex(schemes), eachindex(xgrids)) |> collect

    display(opts)
    @threads for (scheme_idx, xgrid_idx) in runs
        # initialize
        scheme  = schemes[scheme_idx]
        xs      = xgrids[xgrid_idx]
        ys      = xs
        pdeinfo = scheme[:pdeinfo]

        runtag = "$tag-$(scheme.label)-$(length(xs))"

        s0           = exact(tspan[begin], xs, ys)
        Dx₋, Dx₊, Hx = dp_operator(opts[:deriv_type], opts[:bc], opts[:deriv_order], xs)
        Dy₋, Dy₊, Hy = dp_operator(opts[:deriv_type], opts[:bc], opts[:deriv_order], ys)

        pde! = semidiscretise(pdeinfo, s0, Dx₊, Dx₋, Dy₊, Dy₋)
        prob = ODEProblem(pde!, from_primitive_vars(pdeinfo, s0), tspan)

        # solve
        error = SavedValues(Float64, Float64)
        prim_vars = similar.(s0)
        solve(prob, SSPRK54();
            dt = Δt(step(xs)),
            save_on = false,
            callback = SavingCallback(error;
                save_everystep = true,
                save_start = true,
                save_end = true
            ) do u, t, _
                to_primitive_vars!(prim_vars, pdeinfo, u)
                if any(≈(t; atol = 0.5Δt(step(xs)) ), save_times)
                    mkpath(datadir(odir, runtag))
                    vtk_grid(datadir(odir, runtag, "time_$t.vti"), xs, ys) do vtk
                        vtk["ϱ"] = prim_vars[1]
                        vtk["u"] = @views (prim_vars[2], prim_vars[3])
                        vtk["p"] = prim_vars[4]
                    end
                end
                exactf = exact(t, xs, ys)
                √energy_norm(prim_vars .- exactf, Hx, Hy) do (δϱ, δu, δp)
                    return δϱ^2 + δu^2 + δp^2
                end
            end,
            progress = true,
            progress_steps = 250,
            progress_id = Symbol(scheme_idx * length(xgrids) + xgrid_idx),
            progress_name = rpad("$(scheme[:label]) ", 24, "─") *
                              lpad(" $(length(xs))×$(length(xs))", 8)
        )

        wsave(datadir(odir, "error-$runtag.jld2"),
            Dict(
                "l2 error" => error.saveval,
                "t" => error.t
            ))
    end

    errors = [fill(NaN64, length(xgrids)) for _ in eachindex(schemes)]
    for scheme_idx in eachindex(schemes)
        scheme = schemes[scheme_idx]

        fig = Figure()
        ax = Axis(fig[1, 1],
            ylabel = "Error",
            xlabel = L"t",
            yscale = pseudolog_tol(1e-7),
            yticks = [0.0; 10.0 .^ (-6:0)],
            yminorticks = IntervalsBetween(5),
            width = 200,
            height = 140
        )

        for xgrid_idx in eachindex(xgrids)
            xs = xgrids[xgrid_idx]
            runtag = "$tag-$(scheme.label)-$(length(xs))"

            error = load(datadir(odir, "error-$runtag.jld2"))
            if error["t"][end] ≈ tspan[end]
                errors[scheme_idx][xgrid_idx] = error["l2 error"][end]
            else
                @warn "mismatched final times" error["t"][end] tspan[end]
            end

            lines!(ax, error["t"], error["l2 error"], label = "N=$(length(xs))")
        end
        ylims!(ax, 0.0, 3.0)
        axislegend(ax)
        resize_to_layout!(fig)
        wsave(plotsdir(odir, "error-$tag-$(scheme.label).pdf"), fig)
    end

    plot_convergence(
        plotsdir(odir, "convergence-$tag.pdf"),
        get.(schemes, :label, ""),
        errors,
        Δxs;
        figargs = (; size = (280, 200)),
        markers = getfield.(schemes, (:marker,)),
        legendname = plotsdir(odir, "convergence-legend-$tag.pdf")
    )
    tablepath = datadir(odir, "convergence-$tag.tex")
    mkpath(dirname(tablepath))
    open(tablepath; write = true) do io
        print_convergence_table_latex(io, schemes, xgrids, errors)
    end
    print_convergence_table_ascii(schemes, xgrids, errors)
end
