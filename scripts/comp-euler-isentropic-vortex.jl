using DrWatson
@quickactivate :HyperbolicPDEs
using SummationByPartsOperators
using CairoMakie
using OrdinaryDiffEqSSPRK
using DiffEqCallbacks
using JLD2
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

const tspan = (0.0, 16.0)
const xspan = (-8.0, 8.0)
const Δt    = (Δx, Δy) -> 4e-2Δx

const γ = 1.4

exact = IsentropicVortex(γ = γ, xrange = xspan, yrange = xspan)

odir = joinpath("comp-euler-2D", string(exact))

const schemes = [
    # (; label = "entropy conserving",
    #     pdeinfo = NordstromCompEuler2D(; γ)
    # ),
    (; label = "entropy stable",
        pdeinfo = NordstromCompEuler2D(FluxDSL2025(); γ)
    ),
    # (; label = "Lax-Friedrichs",
    #     pdeinfo = CompEulerFluxForm2D(FluxLaxFriedrichs(); γ)
    # ),
    # (; label = "van Leer-Hanel",
    #     pdeinfo = CompEulerFluxForm2D(FluxVanLeerHanel(); γ)
    # )
]

const opt_list = mapreduce(dict_list, vcat,
    [
        Dict(
            :deriv_type => Mattsson2017,
            :order      => [8, 6, 4, 2],
            :nodes      => [[16 * 2^i + 1 for i in 0:3]],
            :elems      => [[4]]
        ),
        Dict(
            :deriv_type => GlaubitzEtal2024(-0.1),
            :order      => collect(2:6),
            :nodes      => Derived(:order, p -> [p + 1]),
            :elems      => [[16 * 2^i for i in 0:2]]
        )
    ])

function generate_results(tag, scheme_idx, opts, (K, N))
    # initialize
    @unpack pdeinfo, label = schemes[scheme_idx]

    runtag = "$tag-$label-K=$K-N=$N"

    mesh = CartesianMesh((xspan, xspan), (K, K))
    cell = local_dp_operator(opts[:deriv_type], opts[:order], (N, N))
    xs, fdop = couple_operators(mesh, cell)
    ∫_Ω = VolumeMeasure(xs, fdop)

    s0   = exact(tspan[begin], xs)
    prob = let
        pde! = semidiscretise(pdeinfo, xs, fdop)
        ODEProblem{true, SciMLBase.NoSpecialize}(pde!, from_primitive_vars(pdeinfo, s0), tspan)
    end

    # solve
    error = SavedValues(Float64, Float64)
    primvars = similar.(s0)
    solve(prob, SSPRK54();
        dt = Δt(step(xs)...),
        save_on = false,
        callback = SavingCallback(error; saveat = first(tspan):0.2:last(tspan)) do u, t, _
            to_primitive_vars!(primvars, pdeinfo, u)
            analytic = exact(t, xs)
            √∫_Ω((primvars..., analytic...)) do s
                ϵ = s[1:4] .- s[5:8]
                ϵ ⋅ ϵ
            end
        end,
        progress = true,
        progress_steps = 200,
        progress_id = Symbol(1e4scheme_idx + 1e2K + 1e0N),
        progress_name = rpad(label, 20) * " " * lpad(join(size(xs), "×"), 8)
    )

    wsave(datadir(odir, "error-$runtag.jld2"),
        Dict(
            "l2 error" => error.saveval,
            "t" => error.t
        ))
end

# run solvers
for opts in opt_list
    println()
    display_settings(opts)

    @unpack nodes, elems = opts

    tag = savename(opts) * "-" * string(hash(opts); base = 60)

    xgrids = product(elems, nodes) |> collect
    runs = product(eachindex(schemes), xgrids) |> collect

    @threads for (scheme_idx, xgrid) in runs
        generate_results(tag, scheme_idx, opts, xgrid)
    end

    errors = [fill(NaN64, length(xgrids)) for _ in eachindex(schemes)]
    for scheme_idx in eachindex(schemes)
        @unpack pdeinfo, label = schemes[scheme_idx]

        fig = Figure()
        ax = Axis(fig[1, 1],
            ylabel = "Error",
            xlabel = L"t",
            yscale = pseudolog_tol(1e-6),
            ytickformat = pows10_ytickformatter,
            yticks = [0.0; 10.0 .^ (-5:0)],
            yminorticks = IntervalsBetween(5),
            width = 200,
            height = 140
        )

        for xgrid_idx in eachindex(xgrids)
            nb, N = xgrids[xgrid_idx]
            runtag = "$tag-$label-K=$nb-N=$N"

            error = load(datadir(odir, "error-$runtag.jld2"))
            if error["t"][end] ≈ tspan[end]
                errors[scheme_idx][xgrid_idx] = error["l2 error"][end]
            else
                @warn "mismatched final times" error["t"][end] tspan[end]
            end

            lines!(ax, error["t"], error["l2 error"], label = "K=$nb N=$N")
        end
        ylims!(ax, 0.0, 1.0)
        Legend(fig[1, 2], ax)
        resize_to_layout!(fig)
        wsave(plotsdir(odir, "error-$tag-$label.pdf"), fig)
    end

    tablepath = datadir(odir, "convergence-$tag.tex")
    mkpath(dirname(tablepath))
    open(tablepath; write = true) do io
        print_convergence_table_latex(io, schemes, xgrids, errors)
    end
    print_convergence_table_ascii(schemes, xgrids, errors)
end
