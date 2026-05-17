using DrWatson
@quickactivate :HyperbolicPDEs
using Printf
using SummationByPartsOperators
using CairoMakie
using OrdinaryDiffEqSSPRK
using DiffEqCallbacks
using Base.Threads: @spawn
using LinearAlgebra
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include(srcdir("PDEs", "compressible-euler-analytic.jl"))
include(srcdir("PDEs", "compressible-euler.jl"))
include(srcdir("makie-theme.jl"))
include(srcdir("plotting-utils.jl"))

const γ = 1.4

# const saveat = 0:0.01:1.5
const saveat = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5]
const tspan  = (0.0, 2.0)
const Δt     = (Δx,) -> 0.005Δx
const domain = ((-6.0, 6.0),)

const exact = SodShockTube(; γ)

const outpath = joinpath("comp-euler-1D", string(exact))

const testname = string(exact)

const cons_qtys = (
    (; type = :rel_diff, qtyname = "mass",
        fn = ((ϱ, v..., p),) -> ϱ
    ),
    (; type = :abs_diff, qtyname = "momentum",
        fn = ((ϱ, v..., p),) -> ϱ * v[1]
    ),
    (; type = :rel_diff, qtyname = "energy",
        fn = ((ϱ, v..., p),) -> p / (γ - 1) + ϱ * (v ⋅ v) / 2
    )
)

const entr_qtys = (
    (; qtyname = "entropy", ylimits = (-0.5, -0.42),
        fn = Base.Fix1(thermodynamic_entropy, γ)
    ),
    (; qtyname = L"Harten entropy $(α=1-2γ)$", ylimits = (6.14, 6.18),
        fn = Base.Fix1(harten_entropy, (γ, 1 - 2γ))
    ),
    (; qtyname = L"Harten entropy $(α=1)$", ylimits = (-41.93, -41.75),
        fn = Base.Fix1(harten_entropy, (γ, 1))
    )
)

const schemes = Dict(
    "Nordstrom (γ=0)"  => NordstromCompEuler1D(; γ),
    "Nordstrom (γ>0)"  => NordstromCompEuler1D(FluxDSL2025(); γ),
    "van Leer-Hänel"   => CompEulerFluxForm1D(FluxVanLeerHanel(); γ),
    "Lax-Fried. (γ>0)" => CompEulerFluxForm1D(FluxLaxFriedrichs(); γ),
    "Flux Form"        => CompEulerFluxForm1D(; γ)
)

const label_colors = Dict(
    "DGSEM/SBP FD"   => Makie.wong_colors()[7],
    "DP DG/FD"       => Makie.wong_colors()[2],
    "van Leer-Hänel" => Makie.wong_colors()[4],
    "Lax-Fried."     => Makie.wong_colors()[1],
    "Flux Form"      => Makie.wong_colors()[5]
)

function make_label(opts)
    Dict(
        "Nordstrom (γ=0)" => "DGSEM/SBP FD",
        "Nordstrom (γ>0)" => "DP DG/FD",
        "van Leer-Hänel" => "van Leer-Hänel",
        "Lax-Fried. (γ>0)" => "Lax-Fried.",
        "Flux Form" => "Flux Form"
    )[opts[:scheme]]
end

function make_model(opts)
    @unpack elems, nodes = opts
    mesh = CartesianMesh(domain, (elems,))
    cell = local_dp_operator(opts[:deriv_type], opts[:order], (nodes,))
    xs, fdop = couple_operators(mesh, cell)
    return xs, fdop
end

function solve_scheme(opts; runindex)
    pdeinfo = schemes[opts[:scheme]]

    xs, fdop = make_model(opts)
    ∫_Ω = VolumeMeasure(xs, fdop)

    s₀ = exact(tspan[begin], xs)

    pde! = semidiscretise(pdeinfo, xs, fdop)
    prob = ODEProblem{true, SciMLBase.NoSpecialize}(
        pde!, from_primitive_vars(pdeinfo, s₀), tspan)

    primvars = similar.(s₀)
    qtys = SavedValues(Float64, NTuple{length(cons_qtys) + length(entr_qtys), Float64})
    crash_time = Ref(NaN64)
    sol = solve(prob, SSPRK54();
        saveat,
        dt = Δt(step(xs)...),
        unstable_check = (_, u, _, t) -> any(x -> isinf(x) || isnan(x), u) ?
                                         (crash_time[] = t; true) : false,
        callback = SavingCallback(qtys) do u, t, _
            to_primitive_vars!(primvars, pdeinfo, u)
            cons = map(qty -> ∫_Ω(qty.fn, primvars), cons_qtys)
            entr = map(qty -> ∫_Ω(qty.fn, primvars), entr_qtys)
            return (cons..., entr...)
        end,
        progress = true,
        progress_steps = 250,
        progress_id = Symbol(runindex),
        progress_name = rpad(make_label(opts), 30)
    )

    return (;
        t = sol.t,
        u = [to_primitive_vars(pdeinfo, u) for u in sol.u],
        qtys = qtys,
        crash_time = crash_time[]
    )
end

function generate_results(testopt, runopts)
    results = map(enumerate(runopts)) do (runindex, runopt)
        @spawn solve_scheme(merge(runopt, testopt); runindex)
    end .|> fetch

    for (result, runopt) in zip(results, runopts)
        opts = merge(runopt, testopt)
        params_str = savename(opts)
        mkpath(datadir(outpath))
        open(datadir(outpath, "crash_time-$params_str.txt"); write = true) do io
            println(io, "crash time = ", result.crash_time)
        end
    end

    results
end

# Compare Schemes

function make_animation(testopt, runopts, results)
    testhash = hash(testopt)
    testtag  = savename(testopt) * "_" * string(testhash; base = 60)
    testdir  = joinpath(outpath, testtag)

    names = [make_label(merge(runopt, testopt)) for runopt in runopts]

    (qty, ylimits, fn) = ("ϱ", (0.0, 1.2), ((ϱ, u, p),) -> ϱ)
    for (n, time) in enumerate(saveat)
        analytic_xs = range(first(domain)..., 2048)
        analytic_sol = exact(time, analytic_xs)

        fig = Figure()
        ax = Axis(
            fig[1, 1],
            xlabel = L"x",
            ylabel = latexstring(qty),
            title  = latexstring(@sprintf "t = %.3f" time),
            limits = ((-3.0, 3.0), ylimits),
            width  = 400, height = 200
        )
        for (name, sol, runopt) in zip(names, results, runopts)
            xs, _ = make_model(merge(runopt, testopt))
            if checkbounds(Bool, sol.u, n)
                lines!(ax, xs, fn(sol.u[n]); label = name, color = label_colors[name])
            end
        end
        lines!(ax, analytic_xs, fn(analytic_sol);
            linestyle = :dash, color = :black, label = "exact")
        axislegend(ax; merge = true)

        resize_to_layout!(fig)
        wsave(plotsdir(testdir, "frames", @sprintf "%05d.png" n), fig)
    end
end

function compare_state(testopt, runopts, results)
    testhash = hash(testopt)
    testtag  = savename(testopt) * "_" * string(testhash; base = 60)
    testdir  = joinpath(outpath, testtag)

    names = [make_label(merge(runopt, testopt)) for runopt in runopts]

    for (n, time) in enumerate(saveat)
        analytic_xs = range(first(domain)..., 2048)
        analytic_sol = exact(time, analytic_xs)

        qtys = [
            ("ϱ", (0.0, 1.2), ((ϱ, u, p),) -> ϱ),
            # ("u", (-0.1, 1.1), ((ϱ, u, p),) -> ϱ .* u),
            ("ϱv", (-0.1, 0.6), ((ϱ, u, p),) -> ϱ .* u),
            # ("p", (-0.1, 1.1), ((ϱ, u, p),) -> p),
            ("ϱe", (0.1, 3.0), ((ϱ, u, p),) -> @. p / (γ - 1) + ϱ * u^2)
        ]
        fig = Figure(; figure_padding = 8)
        axs = map(enumerate(qtys)) do (qty_i, (qty, ylimits, fn))
            ax = Axis(
                fig[qty_i, 1],
                xlabel = L"x",
                ylabel = latexstring(qty),
                limits = ((-3, 3), ylimits),
                width = 220, height = 80
            )
            for (name, sol, runopt) in zip(names, results, runopts)
                xs, _ = make_model(merge(runopt, testopt))
                if checkbounds(Bool, sol.u, n)
                    lines!(ax, xs, fn(sol.u[n]); label = name, color = label_colors[name])
                end
            end
            lines!(ax, analytic_xs, fn(analytic_sol);
                linestyle = :dash, color = :black, label = "Analytic")
            ax
        end
        for ax in axs[begin:(end - 1)]
            hidexdecorations!(ax, ticks = false, grid = false, minorticks = false)
        end
        yspace = maximum(tight_yticklabel_spacing!, axs)
        for ax in axs
            ax.yticklabelspace = yspace
        end
        resize_to_layout!(fig)
        wsave(plotsdir(testdir, @sprintf "t=%f.pdf" time), fig)
    end
end

function compare_cons_globals(testopt, runopts, results)
    testhash = hash(testopt)
    testtag  = savename(testopt) * "_" * string(testhash; base = 60)
    testdir  = joinpath(outpath, testtag)

    names = [make_label(merge(runopt, testopt)) for runopt in runopts]

    fig = Figure()
    volΩ = abs(prod(splat(-), domain))
    for (qty_i, cons_qty) in enumerate(cons_qtys)
        @unpack type, qtyname, fn = cons_qty
        println("plotting total $qtyname")

        ax = Axis(fig[1, qty_i];
            xlabel = L"t",
            title = qtyname,
            yscale = pseudolog_tol(1e-14),
            ytickformat = pows10_ytickformatter,
            yticks = [0; 10.0 .^ (-12:3:-3)],
            yminorticks = IntervalsBetween(4),
            yminorgridvisible = true,
            limits = ((-0.05, 1.55), (-1e-13, 1e-3)),
            width = 105, height = 100)
        if qty_i ≠ 1
            hideydecorations!(ax;
                minorgrid = false, grid = false, ticks = false, minorticks = false)
        end

        for i in eachindex(results)
            sol = results[i]
            label = names[i]
            ts = sol.qtys.t
            qty = getindex.(sol.qtys.saveval, qty_i)
            ys = type == :rel_diff ? (qty .- qty[begin]) ./ qty[begin] :
                 type == :abs_diff ? (qty .- qty[begin]) ./ volΩ :
                 qty
            lines!(ax, ts, abs.(ys); label, color = label_colors[label])
        end
        if qty_i == firstindex(cons_qtys)
            save_legend(plotsdir(testdir, "legend-cons-horizontal.pdf"), ax, :horizontal)
            save_legend(plotsdir(testdir, "legend-cons-vertical.pdf"), ax, :vertical)
        end
    end
    resize_to_layout!(fig)
    wsave(plotsdir(testdir, "cons-qtys.pdf"), fig)
end

function compare_entr_globals(testopt, runopts, results)
    testhash = hash(testopt)
    testtag  = savename(testopt) * "_" * string(testhash; base = 60)
    testdir  = joinpath(outpath, testtag)

    names = [make_label(merge(runopt, testopt)) for runopt in runopts]

    tend = 1.5
    analytic_ts = range(0.0, 1.55, 100)

    for (qty_i, entr_qty) in enumerate(entr_qtys)
        @unpack (qtyname, ylimits, fn) = entr_qty
        println("plotting total $qtyname")

        fig = Figure()
        ax = Axis(fig[1, 1];
            xlabel = L"t",
            title = qtyname,
            width = 120, height = 100)

        for i in eachindex(results)
            sol = results[i]
            label = names[i]
            ts = sol.qtys.t
            ys = getindex.(sol.qtys.saveval, length(cons_qtys) + qty_i)
            lines!(ax, ts, ys; label, color = label_colors[label])
        end

        let ys = [2integrate_solution(fn, exact, t, domain[1] ./ 2) for t in analytic_ts]
            lines!(ax, analytic_ts, ys;
                label = "exact", linestyle = :dash, color = :black)
        end

        if all(isfinite, ylimits)
            yrange = last(ylimits) - first(ylimits)
            ylims!(ax, (@. ylimits + yrange * (-0.1, 0.1))...)
        end
        xlims!(ax, -0.05, 1.55)
        resize_to_layout!(fig)
        wsave(plotsdir(testdir, "entr-$qtyname.pdf"), fig)

        if qty_i == firstindex(entr_qtys)
            save_legend(plotsdir(testdir, "legend-entr-horizontal.pdf"), ax, :horizontal)
            save_legend(plotsdir(testdir, "legend-entr-vertical.pdf"), ax, :vertical)
        end
    end
end

## generate snapshots and conservation plots

testopts = mapreduce(dict_list, vcat, [
    Dict(
        :elems      => 64,
        :order      => 6,
        :nodes      => Derived(:order, p -> p + 1),
        :deriv_type => GlaubitzEtal2024(-0.1)
    ),
    Dict(
        :elems      => 12,
        :order      => 6,
        :nodes      => 33,
        :deriv_type => Mattsson2017
    )
])

runopts = dict_list(Dict(
    :scheme => ["Nordstrom (γ=0)", "Nordstrom (γ>0)", "Lax-Fried. (γ>0)", "van Leer-Hänel"],
))

for testopt in testopts
    println("generating solutions for:")
    display(testopt)
    results = generate_results(testopt, runopts)
    # make_animation(testopt, runopts, results)
    compare_state(testopt, runopts, results)
    compare_cons_globals(testopt, runopts, results)
    compare_entr_globals(testopt, runopts, results)
end

## generate crash times

testopts = mapreduce(dict_list, vcat, [
    Dict(
        :elems      => [32, 64, 128],
        :order      => Derived(:nodes, N -> N - 1),
        :nodes      => 7:-1:4 |> collect,
        :deriv_type => [GlaubitzEtal2024(-0.1)]
    ),
    Dict(
        :elems      => [16],
        :order      => 9:-1:6 |> collect,
        :nodes      => [16, 32, 64] .+ 1,
        :deriv_type => [Mattsson2017]
    )
])

runopts = dict_list(Dict(
    :scheme => ["Nordstrom (γ=0)", "Nordstrom (γ>0)", "Lax-Fried. (γ>0)", "van Leer-Hänel"],
))

for testopt in testopts
    println("computing crash times for:")
    display_settings(testopt)
    generate_results(testopt, runopts)
end
