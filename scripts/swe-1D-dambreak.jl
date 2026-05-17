using DrWatson
@quickactivate :HyperbolicPDEs
using Printf
using SummationByPartsOperators
using Makie, CairoMakie
using OrdinaryDiffEqSSPRK
using DiffEqCallbacks
using LaTeXStrings
using Base.Threads: @spawn
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include(srcdir("PDEs", "swe-analytic.jl"))
include(srcdir("PDEs", "swe.jl"))
include(srcdir("makie-theme.jl"))
include(srcdir("plotting-utils.jl"))

const g = 1.0

const exact = DambreakPeriodic(; hₗ = 1.2, hᵣ = 0.2, x₀ = 15.0, g)

const outpath = joinpath("swe-1D", string(exact))

const saveat = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0]
const tspan  = (0.0, 10.0)
const Δt     = (Δx,) -> 1e-3Δx
const domain = ((-30.0, 30.0),)

const cons_qtys = (
    (; type = :rel_diff, qtyname = "mass",
        fn = ((h, u, b),) -> h),
    (; type = :abs_diff, qtyname = "momentum",
        fn = ((h, u, b),) -> h * u)
)

const entr_qtys = (
    (; qtyname = "energy", title = "energy/entropy", ylimits = (-2.1e-2, 1e-3),
        fn = ((h, u, b),) -> h * g * (h / 2 + b) + h * u^2 / 2),
)

const schemes = Dict(
    "Lax-Fried." => ShallowWaterFluxForm1D(FluxLaxFriedrichs(); g),
    "SkewSym (γ=0)" => ShallowWaterSkewSym1D(; g),
    "SkewSym (γ>0)" => ShallowWaterSkewSym1D(FluxEntropyStable(); g),
)

const label_colors = Dict(
    "DGSEM/SBP FD" => Makie.wong_colors()[7],
    "DP DG/FD" => Makie.wong_colors()[2],
    "linearly stable DP DG/FD" => Makie.wong_colors()[1],
    "Flux Form" => Makie.wong_colors()[5]
)

function make_label(opts)
    Dict(
        "SkewSym (γ=0)" => "DGSEM/SBP FD",
        "SkewSym (γ>0)" => "DP DG/FD",
        "Lax-Fried." => "linearly stable DP DG/FD",
        "Flux Form" => "Flux Form",
    )[opts[:scheme]]
end

function make_model(opts)
    mesh = CartesianMesh(domain, (opts[:elems],))
    cell = local_dp_operator(opts[:deriv_type], opts[:order], (opts[:nodes],))
    xs, fdop = couple_operators(mesh, cell)
    return xs, fdop, cell
end

function solve_scheme(opts; runindex = 1)
    pdeinfo = schemes[opts[:scheme]]

    xs, fdop, _ = make_model(opts)
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
        unstable_check = (_, u, _, t) -> any(!isfinite, u) ?
                                         (crash_time[] = t; true) : false,
        callback = SavingCallback(qtys) do u, t, _
            to_primitive_vars!(primvars, pdeinfo, u)
            cons = map(qty -> ∫_Ω(qty.fn, primvars), cons_qtys)
            entr = map(qty -> ∫_Ω(qty.fn, primvars), entr_qtys)
            return (cons..., entr...)
        end,
        progress = true,
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

function compare_state(testopt, runopts, results)
    testhash = hash(testopt)
    testtag  = savename(testopt) * "_" * string(testhash; base = 60)
    testdir  = joinpath(outpath, testtag)

    names = [make_label(merge(runopt, testopt)) for runopt in runopts]

    for (n, time) in enumerate(saveat)
        analytic_xs = range(first(domain)..., 2048)
        analytic_sol = exact(time, analytic_xs)

        hmax = exact.hₗ
        humax = 2(exact.aₗ - exact.cₘ) * exact.cₘ^2 / g
        qtys = [
            ("h", [
                    ((0, 30), (0.0, 1.25hmax)),
                    ((10, 20), (0.0, 1.25hmax))
                ], ((h, u, b),) -> h),
            ("hu", [
                    ((0, 30), (-0.25humax, 1.5humax)),
                    ((10, 20), (-0.25humax, 1.5humax))
                ],
                ((h, u, b),) -> h .* u)
        ]
        for limᵢ in 1:2
            fig = Figure()
            axs = map(enumerate(qtys)) do (qty_i, (qty, limits, fn))
                limit = limits[limᵢ]
                ax = Axis(
                    fig[qty_i, 1],
                    xlabel = L"x",
                    ylabel = latexstring(qty),
                    limits = limit,
                    width = 220, height = 80
                )
                for (name, sol, runopt) in zip(names, results, runopts)
                    xs, _, cell = make_model(merge(runopt, testopt))
                    if checkbounds(Bool, sol.u, n)
                        lines!(ax, xs, fn(sol.u[n]);
                            label = name, color = label_colors[name])
                    end
                end
                lines!(ax, analytic_xs, fn(analytic_sol);
                    linestyle = :dash, color = :black, label = "exact")
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
            wsave(plotsdir(testdir, "$(limᵢ)_t=$time.pdf"), fig)
        end
    end
end

function compare_cons_globals(testopt, runopts, results)
    testhash = hash(testopt)
    testtag  = savename(testopt) * "_" * string(testhash; base = 60)
    testdir  = joinpath(outpath, testtag)

    names = [make_label(merge(runopt, testopt)) for runopt in runopts]

    fig = Figure()
    for (qty_i, cons_qty) in enumerate(cons_qtys)
        @unpack type, qtyname = cons_qty
        println("plotting total $qtyname")

        ax = Axis(fig[1, qty_i];
            xlabel = L"t",
            title = qtyname,
            width = 120, height = 100,
            yscale = pseudolog_tol(10.0^-14),
            yticks = [0.0; 10.0 .^ (-12:3:-3)],
            yminorticks = IntervalsBetween(5),
            yminorgridvisible = true,
            ytickformat = pows10_ytickformatter,
            limits = (nothing, (-1e-13, 1e-3))
        )
        if qty_i ≠ 1
            hideydecorations!(
                ax, minorgrid = false, grid = false, ticks = false, minorticks = false)
        end

        for i in eachindex(results)
            sol = results[i]
            label = names[i]
            ts = sol.qtys.t
            qty = getindex.(sol.qtys.saveval, qty_i)
            ys = type == :rel_diff ? (qty .- qty[begin]) ./ qty[begin] :
                 type == :abs_diff ? qty .- qty[begin] :
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

    analytic_ts = range(tspan..., 100)

    for (qty_i, entr_qty) in enumerate(entr_qtys)
        @unpack qtyname, ylimits, fn = entr_qty
        println("plotting total $qtyname")

        fig = Figure()
        ax = Axis(fig[1, qty_i];
            title = hasproperty(entr_qty, :title) ? entr_qty.title : qtyname,
            xlabel = L"t",
            width = 120, height = 100
        )

        for i in eachindex(results)
            sol = results[i]
            label = names[i]
            ts = sol.qtys.t
            ys = getindex.(sol.qtys.saveval, length(cons_qtys) + qty_i)
            lines!(ax, ts, @. (ys - ys[1]) / ys[1];
                label, color = label_colors[label])
        end
        let ys = [integrate_solution(fn, exact, t, domain[1]) for t in analytic_ts]
            lines!(ax, analytic_ts, @. (ys - ys[1]) / ys[1];
                label = "exact", linestyle = :dash, color = :black)
        end

        yrange = last(ylimits) - first(ylimits)
        if all(isfinite, ylimits)
            ylims!(ax, (@. ylimits + yrange * (-0.1, 0.1))...)
        end
        if qty_i == firstindex(cons_qtys)
            save_legend(plotsdir(testdir, "legend-entr-horizontal.pdf"), ax, :horizontal)
            save_legend(plotsdir(testdir, "legend-entr-vertical.pdf"), ax, :vertical)
        end
        resize_to_layout!(fig)
        wsave(plotsdir(testdir, "entr-$qtyname.pdf"), fig)
    end
end

const testopts = vcat(
    dict_list(Dict(
        :deriv_type => GlaubitzEtal2024(-0.2),
        :order      => Derived(:nodes, N -> N - 1),
        :elems      => 128,
        :nodes      => 6 .+ 1
    )),
    dict_list(Dict(
        :deriv_type => Mattsson2017,
        :order      => 6,
        :elems      => 24,
        :nodes      => 32 .+ 1
    )),
)

# const testopts = vcat(
#     dict_list(Dict(
#         :deriv_type => GlaubitzEtal2024(-0.2),
#         :order      => Derived(:nodes, N -> N - 1),
#         :elems      => [64, 128, 256],
#         :nodes      => collect(7:-1:4)
#     )),
#     dict_list(Dict(
#         :deriv_type => Mattsson2017,
#         :order      => collect(9:-1:5),
#         :elems      => [32],
#         :nodes      => [16, 32, 64] .+ 1
#     )),
# )

const runopts = dict_list(Dict(
    :scheme => ["SkewSym (γ=0)", "SkewSym (γ>0)", "Lax-Fried."],
))

for testopt in testopts
    println("Generating results for:")
    display_settings(testopt)

    results = generate_results(testopt, runopts)
    compare_state(testopt, runopts, results)
    compare_cons_globals(testopt, runopts, results)
    compare_entr_globals(testopt, runopts, results)
end
