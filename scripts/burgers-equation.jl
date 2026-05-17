using DrWatson
@quickactivate :HyperbolicPDEs
using Printf
using SummationByPartsOperators
using CairoMakie
using OrdinaryDiffEqSSPRK
using DiffEqCallbacks
using LaTeXStrings
using Base.Threads: @spawn
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include(srcdir("PDEs", "burgers.jl"))
include(srcdir("plotting-utils.jl"))
include(srcdir("makie-theme.jl"))

# problem setup {{{1
const tspan = (0.0, 1.0)
const Δt    = Δx -> 0.01Δx

const domain = (0.0, 1.0)

# const saveat = 0:0.02:1
const saveat = [0.1, 0.15, 0.75]

const initial_conditions = (
    x -> exp(-(x[1] - 0.3)^2 / 0.01),
)

const outpath = joinpath("burgers-1D", "gaussian")

const pdes = Dict(
    "Skew Sym." => Dict(
        :label => "DGSEM/SBP FD",
        :info => BurgersSkewSym1D(),
        :linestyle => (; color = Makie.wong_colors()[7])
    ),
    "Skew Sym. Upwind" => Dict(
        :label => "DP DG/FD",
        :info => BurgersSkewSym1D(FluxEntropyStable()),
        :linestyle => (; color = Makie.wong_colors()[2])
    ),
    "Lax-Fried." => Dict(
        :label => "linearly stable DP DG/FD",
        :info => BurgersFluxForm1D(FluxLaxFriedrichs()),
        :linestyle => (; color = Makie.wong_colors()[1])
    )
)

# driver functions {{{1
function solve_testcase(opts)
    @unpack deriv_type, order, nodes, elems = opts
    pde = pdes[opts[:scheme]]

    mesh = CartesianMesh((domain,), (elems,))
    cell = local_dp_operator(deriv_type, order, (nodes,))
    xs, fdop = couple_operators(mesh, cell)

    ∫_Ω = VolumeMeasure(xs, fdop)
    qtys = (
        ((u,),) -> ∫_Ω(u),
        ((u,),) -> ∫_Ω(u -> u^2 / 2, u)
    )

    u₀ = map(f -> f.(xs), initial_conditions)

    dudt! = semidiscretise(pde[:info], xs, fdop)
    prob = ODEProblem{true, SciMLBase.NoSpecialize}(
        dudt!, from_primitive_vars(pde[:info], u₀), tspan)

    primvars = similar.(u₀)
    global_qtys = SavedValues(Float64, NTuple{length(qtys), Float64})
    crash_time = Ref(NaN64)
    sol = solve(prob, SSPRK54();
        unstable_check = (_, u, _, t) -> any(!isfinite, u) ?
                                         (crash_time[] = t; true) : false,
        dt = Δt(step(xs)...),
        callback = SavingCallback(global_qtys) do u, t, _
            to_primitive_vars!(primvars, pde[:info], u)
            return ntuple(i -> qtys[i](primvars), Val(length(qtys)))
        end,
        saveat,
        progress = true,
        progress_id = Symbol(hash(opts)),
        progress_name = rpad(pde[:label], 25)
    )

    params_str = savename(opts)
    mkpath(datadir(outpath))

    # write crash times
    open(datadir(outpath, "crashtime-$params_str.txt"); write = true) do io
        println(io, crash_time[])
    end

    return (;
        t = sol.t,
        u = [to_primitive_vars(pde[:info], u) for u in sol.u],
        qtys = global_qtys,
        crash_time = crash_time[]
    )
end

function plot_qtys(odir, testopt, runopts, solutions)
    axargs = (;
        yminorticks = IntervalsBetween(5),
        yminorgridvisible = true,
        xlabel = L"t",
        height = 100
    )
    let
        println("plotting total u")
        fig = Figure()
        ax = Axis(fig[1, 1];
            axargs...,
            width = 80,
            title = "mass",
            yscale = pseudolog_tol(1e-14),
            yticks = [10.0 .^ (-3:-3:-12); 0],
            ytickformat = pows10_ytickformatter,
        )
        ylims!(ax, (-1e-13, 1e-3))
        for (sol, runopt) in zip(solutions, runopts)
            @unpack label, linestyle = pdes[merge(testopt, runopt)[:scheme]]
            qty = getindex.(sol.qtys.saveval, 1)
            lines!(ax, sol.qtys.t, abs.(qty .- qty[begin]) ./ qty[begin]; label, linestyle...)
        end
        resize_to_layout!(fig)
        wsave(joinpath(odir, "u.pdf"), fig)

        save_legend(joinpath(odir, "legend-horizontal.pdf"), ax, :horizontal)
        save_legend(joinpath(odir, "legend-vertical.pdf"), ax, :vertical)
    end
    let
        println("plotting total energy")
        fig = Figure()
        ax = Axis(fig[1, 1];
            axargs...,
            width = 120,
            title = "energy/entropy",
            yscale = pseudolog_tol(1e-3),
            yticks = [1e-2, 0, -1e-2, -1e-1, -1e-0]
        )
        ylims!(ax, (-2.0, 0.02))
        for (sol, runopt) in zip(solutions, runopts)
            @unpack label, linestyle = pdes[merge(testopt, runopt)[:scheme]]
            qty = getindex.(sol.qtys.saveval, 2)
            lines!(ax, sol.qtys.t, (qty .- qty[begin]) ./ qty[begin]; label, linestyle...)
        end
        resize_to_layout!(fig)
        wsave(joinpath(odir, "energy.pdf"), fig)
    end
end

function make_state_plots(odir, testopt, runopts, solutions; cols = 3)
    rows = 1 + (length(saveat) - 1) ÷ cols

    println("plotting state")
    fig = Figure()
    axs = map(enumerate(saveat)) do (n, time)
        row = 1 + (n - 1) ÷ cols
        col = 1 + (n - 1) % cols
        ax = Axis(fig[row, col],
            width = 160, height = 100,
            title = L"t = %$time",
            xlabel = L"x", ylabel = L"u"
        )
        col != 1 && hideydecorations!(ax; grid = false)
        row != rows && hidexdecorations!(ax; grid = false)
        ylims!(ax, -0.1, 1.4)
        ax
    end
    for (sol, runopt) in zip(solutions, runopts)
        opts = merge(testopt, runopt)

        @unpack elems, nodes, order, deriv_type = opts
        mesh = CartesianMesh((domain,), (elems,))
        cell = local_dp_operator(deriv_type, order, (nodes,))
        xs, fdop = couple_operators(mesh, cell)

        @unpack label, linestyle = pdes[opts[:scheme]]
        for (ax, u) in zip(axs, sol.u)
            lines!(ax, xs, only(u); label, linestyle...)
        end
    end
    resize_to_layout!(fig)
    wsave(joinpath(odir, "state.pdf"), fig)
end

function make_residual_plots(odir, opts, solution; cols = 3)
    rows = 1 + (length(saveat) - 1) ÷ cols

    println("plotting state")

    @unpack elems, nodes, order, deriv_type = opts
    mesh = CartesianMesh((domain,), (elems,))
    cell = local_dp_operator(deriv_type, order, (nodes,))
    xs, fdop = couple_operators(mesh, cell)

    @unpack label, linestyle = pdes[opts[:scheme]]

    fig = Figure()
    axs = map(enumerate(saveat)) do (n, time)
        row = 1 + (n - 1) ÷ cols
        col = 1 + (n - 1) % cols
        ax = Axis(fig[row, col];
            width = 160, height = 75,
            xlabel = L"x", ylabel = L"|γ(D_{+}-D_{-})u|",
        )
        col != 1 && hideydecorations!(ax; grid = false)
        # row != rows && hidexdecorations!(ax; grid = false)
        ax
    end
    linkyaxes!(axs...)

    for (ax, u) in zip(axs, solution.u)
        ϵ = fdop[1].Diᵥ * only(u)
        λs = abs.(only(u))
        foreachelement(xs) do I
            λ = @fastmath(maximum)(view(λs, I))
            ϵ[I] .= λ .* ϵ[I]
        end

        lines!(ax, xs, abs.(ϵ); linestyle..., color=:black)
    end

    resize_to_layout!(fig)
    wsave(joinpath(odir, "residual.pdf"), fig)
end

function make_state_anim(odir, testopt, runopts, solutions)
    println("plotting state frames")
    i::Int = 1

    while any(s -> checkbounds(Bool, s.u, i), solutions)
        fig = Figure()
        axs = map(runopts) do _
            Axis(fig[2, end + 1],
                width = 300, height = 300,
                xlabel = L"x", ylabel = L"u"
            )
        end
        hideydecorations!.(axs[(begin + 1):end]; grid = false)

        for (ax, sol, runopt) in zip(axs, solutions, runopts)
            opts = merge(testopt, runopt)
            @unpack deriv_type, order, nodes, elems = opts

            mesh = CartesianMesh((domain,), (elems,))
            cell = local_dp_operator(deriv_type, order, (nodes,))
            xs, fdop = couple_operators(mesh, cell)

            @unpack label, linestyle = pdes[opts[:scheme]]
            if checkbounds(Bool, sol.u, i)
                lines!(ax, cell, xs, only(sol.u[i]), 20; label, linestyle...)
            end
            ylims!(ax, -0.1, 1.1)
        end
        Label(fig[1, :], latexstring(@sprintf "t = %.3f" saveat[i]))

        resize_to_layout!(fig)
        wsave(plotsdir(odir, @sprintf("state-%05d.png", i)), fig)

        i += 1
    end
end

# compare snapshots {{{1
testopts = mapreduce(dict_list, vcat,
    [
        Dict(
            :order      => 8,
            :nodes      => [32] .+ 1,
            :elems      => [4],
            :deriv_type => Mattsson2017
        ),
        Dict(
            :order      => 8,
            :nodes      => Derived(:order, p -> p + 1),
            :elems      => [16],
            :deriv_type => GlaubitzEtal2024(-0.1)
        )
    ])

runopts = dict_list(Dict(
    :scheme => ["Skew Sym.", "Skew Sym. Upwind", "Lax-Fried."]
))

for testopt in testopts
    println("Running test case:")
    display_settings(testopt)
    solutions = fetch.(map(runopts) do runopt
        opts = merge(testopt, runopt)
        @spawn solve_testcase($opts)
    end)

    params_str = savename(testopt)

    mkpath(plotsdir(outpath, params_str))

    odir = plotsdir(outpath, params_str)

    plot_qtys(odir, testopt, runopts, solutions)
    make_state_plots(odir, testopt, runopts, solutions)
end

# compare residuals {{{1
testopts = mapreduce(dict_list, vcat,
    [
        Dict(
            :order      => 8,
            :nodes      => [32] .+ 1,
            :elems      => [4],
            :deriv_type => Mattsson2017,
            :scheme     => ["Skew Sym. Upwind"]
        ),
        Dict(
            :order      => 8,
            :nodes      => Derived(:order, p -> p + 1),
            :elems      => [16],
            :deriv_type => GlaubitzEtal2024(-0.1),
            :scheme     => ["Skew Sym. Upwind"]
        )
    ])

for testopt in testopts
    println("Running test case:")
    display_settings(testopt)
    solution = solve_testcase(testopt)

    params_str = savename(testopt)
    odir = plotsdir(outpath, params_str)
    mkpath(odir)
    make_residual_plots(odir, testopt, solution)
end

# compare crash times {{{1

testopts = mapreduce(dict_list, vcat,
    [
        Dict(
            :order      => collect(7:9),
            :nodes      => collect(16:16:64) .+ 1,
            :elems      => [4],
            :deriv_type => Mattsson2017
        ),
        Dict(
            :order      => collect(3:7),
            :nodes      => Derived(:order, p -> p + 1),
            :elems      => collect(4:4:16),
            :deriv_type => GlaubitzEtal2024(-0.1)
        )
    ])

runopts = dict_list(Dict(
    :scheme => ["Skew Sym.", "Skew Sym. Upwind", "Lax-Fried."]
))

for testopt in testopts
    println("Running test case:")
    display_settings(testopt)
    solutions = fetch.(map(runopts) do runopt
        opts = merge(testopt, runopt)
        @spawn solve_testcase($opts)
    end)
end
