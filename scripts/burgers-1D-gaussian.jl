using DrWatson
@quickactivate :HyperbolicPDEs
using Printf
using WriteVTK, ReadVTK
using SummationByPartsOperators
using CairoMakie
using OrdinaryDiffEqSSPRK
using LaTeXStrings
using Base.Threads: @spawn
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include(srcdir("PDEs", "burgers.jl"))
include(srcdir("plotting-utils.jl"))
include(srcdir("makie-theme.jl"))
set_theme!(hpdes_makie_theme)

const tspan = (0.0, 1.0)
const Δt    = Δx -> 0.1Δx

const N  = 256
const xs = bounded_range(0.0, 1.0, N)

const u₀ = let
    u = @. exp(-(xs - 0.25)^2 / 0.01)
    (u,)
end

const outpath = joinpath("burgers-1D", "gaussian")

# derived properties
const ts = tspan[1]:Δt(step(xs)):tspan[2]

D₋, D₊, H = dp_operator(Mattsson2017, PeriodicSAT(), 6, xs)

params_str = @savename Δt=step(ts) N

pdes = [
    (;
        label = "linearly stable",
        info = BurgersFluxForm1D(FluxLaxFriedrichs()),
        style = (; color = Makie.wong_colors()[1])
    ),
    (;
        label = "entropy conserving",
        info = BurgersSkewSym1D(),
        style = (; color = Makie.wong_colors()[2])
    ),
    (;
        label = "entropy stable",
        info = BurgersSkewSym1D(FluxEntropyStable()),
        style = (; color = Makie.wong_colors()[3])
    )
]

energy(u) = u[1]' * H * u[1]
totalu(u) = sum(H * u[1])

function solve_pdes(pdes)
    return map(enumerate(pdes)) do (idx, pde)
        return @spawn let pde = $pde, idx = $idx
            dudt! = semidiscretise(pde.info, xs, (D₊, D₋))
            prob = ODEProblem(dudt!, from_primitive_vars(pde.info, $u₀), tspan)
            sol = solve(prob, SSPRK54();
                dt = Δt(step(xs)),
                save_end = true,
                progress = true,
                progress_id = Symbol(idx),
                progress_name = rpad("$(pde.label)", 30)
            )
            (; t = sol.t, u = [to_primitive_vars(pde.info, u) for u in sol.u])
        end
    end .|> fetch
end

solutions = solve_pdes(pdes)
let
    println("plotting total u")
    fig = Figure()
    ax = Axis(fig[1, 1],
        ylabel = L"⟨1, \mathbf{u}⟩_H / ⟨1, \mathbf{u}⟩_H - 1",
        xlabel = L"t",
        width = 190, height = 130
    )
    for (sol, pde) in zip(solutions, pdes)
        qty = totalu.(sol.u)
        lines!(ax, sol.t, (qty .- qty[begin]) ./ qty[begin];
            label = pde.label, pde.style...)
    end
    # axislegend(position = :lb)
    resize_to_layout!(fig)
    wsave(plotsdir(outpath, "u-$params_str.pdf"), fig)

    fig = Figure()
    Legend(fig[1, 1], ax, halign = :center, orientation = :horizontal,
        tellwidth = true, tellheight = true)
    resize_to_layout!(fig)
    wsave(plotsdir(outpath, "legend-horizontal-$params_str.pdf"), fig)

    fig = Figure()
    Legend(fig[1, 1], ax, halign = :center, tellwidth = true, tellheight = true)
    resize_to_layout!(fig)
    wsave(plotsdir(outpath, "legend-vertical-$params_str.pdf"), fig)
end
let
    println("plotting total energy")
    fig = Figure()
    ax  = Axis(fig[1, 1], ylabel = L"‖\mathbf{u}‖_H^2 / ‖\mathbf{u}_0‖_H^2 - 1", xlabel = L"t", width = 190, height = 130)
    for (sol, pde) in zip(solutions, pdes)
        qty = energy.(sol.u)
        lines!(
            ax, sol.t, (qty .- qty[begin]) ./ qty[begin]; label = pde.label, pde.style...)
    end
    # axislegend(position = :lb)
    resize_to_layout!(fig)
    wsave(plotsdir(outpath, "energy-$params_str.pdf"), fig)
end

let save_times = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6]
    println("plotting state")
    fig = Figure()
    ax = ()
    for (n, time) in enumerate(save_times)
        ax = Axis(fig[(n - 1) ÷ 3, (n - 1) % 3],
            width = 150, height = 110,
            title = L"t = %$time",
            xlabel = L"x",
            ylabel = L"u"
        )
        for (sol, pde) in zip(solutions, pdes)
            i = findfirst(≈(time), sol.t)
            lines!(ax, xs, sol.u[i][1]; linewidth = 1, label = pde.label, pde.style...)
        end
        ylims!(ax, -0.1, 1.4)
    end
    # Legend(fig[end + 1, :], ax, halign = :center, orientation = :horizontal)
    resize_to_layout!(fig)
    wsave(plotsdir(outpath, "state-$params_str.pdf"), fig)
end
