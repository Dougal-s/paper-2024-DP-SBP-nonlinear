using DrWatson
@quickactivate :HyperbolicPDEs
using Printf
using WriteVTK, ReadVTK
using SummationByPartsOperators
using CairoMakie
using OrdinaryDiffEqSSPRK
using DiffEqCallbacks
using LaTeXStrings
using Base.Threads: @threads
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include(srcdir("PDEs", "swe.jl"))
include(srcdir("makie-theme.jl"))
include(srcdir("plotting-utils.jl"))
set_theme!(hpdes_makie_theme)

const f = 5.0 # coriolis
const g = 5.0 # gravity
const H = 8.0 # base height

# The collision and merging of two vortices
const testname   = "merging-vortex"
const tspan      = (0.0, 20.0)
const Δt         = (Δx, Δy) -> 0.1 / √40 * min(Δx, Δy) # Δx * Δy / (20Δx + 20Δy)
const save_every = 0.1
const domain     = ((-0, 2π), (-0, 2π))
const state0     = let
    ψᵢ(x, y, x₀, y₀, r)   = exp(-r * ((x - x₀)^2 + (y - y₀)^2))
    ∂xψᵢ(x, y, x₀, y₀, r) = -2r * (x - x₀) * ψᵢ(x, y, x₀, y₀, r)
    ∂yψᵢ(x, y, x₀, y₀, r) = -2r * (y - y₀) * ψᵢ(x, y, x₀, y₀, r)

    vortices  = ((2.6π / 3, π, 2.5), (3.5π / 3, π, 2.5))
    ψ(x, y)   = sum(ψᵢ(x, y, p...) for p in vortices)
    ∂xψ(x, y) = sum(∂xψᵢ(x, y, p...) for p in vortices)
    ∂yψ(x, y) = sum(∂yψᵢ(x, y, p...) for p in vortices)

    # geostrophic balance:
    # f u^⊥ + g ∇h = 0
    h = (x, y) -> H + f / g * ψ(x, y)
    u = (x, y) -> -∂yψ(x, y)
    v = (x, y) -> ∂xψ(x, y)
    b = (x, y) -> zero(x)

    (h, u, v, b)
end

const schemes = Dict(
    "Flux Form"         => ShallowWaterFluxForm2D(; f = f, g = g),
    "Lax-Fried."        => ShallowWaterFluxForm2D(FluxLaxFriedrichs(); f = f, g = g),
    "Skew Symm. (γ=0)"  => ShallowWaterSkewSym2D(; f = f, g = g),
    "Skew Symm. (γ>0)"  => ShallowWaterSkewSym2D(FluxEntropyStable(); f = f, g = g),
    "Vector Inv. (γ=0)" => ShallowWaterVectorInv2D(; f = f, g = g),
    "Vector Inv. (γ>0)" => ShallowWaterVectorInv2D(FluxEntropyStable(); f = f, g = g)
)

const scheme_colors = Dict(
    "Lax-Fried."       => Makie.wong_colors()[1],
    "Skew Symm. (γ=0)" => Makie.wong_colors()[2],
    "Skew Symm. (γ>0)" => Makie.wong_colors()[3],
    "Flux Form"        => Makie.wong_colors()[4]
)

const scheme_labels = Dict(
    "Lax-Fried."       => "linearly stable",
    "Skew Symm. (γ=0)" => "entropy conserving",
    "Skew Symm. (γ>0)" => "entropy stable",
    "Flux Form"        => "flux form"
)

function filetag(opts)
    pdeinfo = schemes[opts[:scheme]]
    optsstr = Dict(k => repr(v) for (k, v) in opts)
    params_hash = string(hash(pdeinfo)^hash(optsstr); base = 60)
    return savename("$params_hash", optsstr)
end

function make_output_dirname(opts)
    return joinpath("swe-2D", testname, filetag(opts))
end

allopts = dict_list(Dict(
    :scheme      => ["Skew Symm. (γ=0)", "Skew Symm. (γ>0)"],
    :gridsize    => [(128, 128)],
    :deriv_order => 7,
    :deriv_type  => Mattsson2017,
    :boundary    => PeriodicSAT(),
    :domain_type => bounded_range
))

@threads for opts in allopts
    pdeinfo = schemes[opts[:scheme]]

    xs = map((dom, n) -> opts[:domain_type](dom..., n), domain, opts[:gridsize])
    (Dx₋, Dx₊, Hx), (Dy₋, Dy₊, Hy) = dp_operator.(
        Ref(opts[:deriv_type]),
        Ref(opts[:boundary]),
        Ref(opts[:deriv_order]),
        xs
    )
    s₀ = meshgrid.(state0, Ref.(xs)...)

    ∂x(m) = ∂x2D(Dx₋ + Dx₊, m)
    ∂y(m) = ∂y2D(Dy₋ + Dy₊, m)
    vorticity((h, u, v)) = ∂x(v) .- ∂y(u) .+ f

    pde! = semidiscretise(pdeinfo, xs, ((Dx₊, Dx₋), (Dy₊, Dy₋)))
    prob = ODEProblem(pde!, from_primitive_vars(pdeinfo, s₀), tspan)

    output_dir = make_output_dirname(opts)
    mkpath(datadir(output_dir))

    open(datadir(output_dir, "settings.txt"); write = true) do io
        show(io, MIME("text/plain"), opts)
    end

    println(output_dir)
    crash_time = Ref{Union{Nothing, Float64}}(nothing)
    paraview_collection(datadir(output_dir, "$testname.pvd")) do pvd
        primitive_vars = map(similar, s₀)

        last_saved = Ref(-save_every - 1.0)

        solve(prob, SSPRK54();
            dt = Δt(step.(xs)...),
            save_on = false,
            unstable_check = (_, u, _, t) -> any(isnan, u) ? (crash_time[] = t; true) :
                                             false,
            callback = FunctionCallingCallback(
                (u, t, _) -> begin
                    approxless(last_saved[] + save_every, t) || return ()
                    last_saved[] = t
                    to_primitive_vars!(primitive_vars, pdeinfo, u)
                    vtk_grid(datadir(output_dir, "time_$t.vti"), xs...) do vtk
                        ω = vorticity(primitive_vars)

                        vtk["h"]  = primitive_vars[1]
                        vtk["u"]  = @views (primitive_vars[2], primitive_vars[3])
                        vtk["ω"]  = ω
                        vtk["PV"] = ω ./ primitive_vars[1]
                        pvd[t]    = vtk
                    end
                    return ()
                end;
                func_everystep = true),
            progress = true,
            progress_steps = 250,
            progress_id = Symbol(hash(opts)),
            progress_name = rpad(opts[:scheme], 20)
        )
    end

    open(datadir(output_dir, "crash_time.txt"); write = true) do io
        show(io, MIME("text/plain"), crash_time[])
    end
end

globals = Dict()

for opts in allopts
    output_dir = make_output_dirname(opts)
    mkpath(datadir(output_dir))

    pvdpath = datadir(output_dir, "$testname.pvd")
    pvd = PVDFile(pvdpath)
    nsteps = length(pvd.timesteps)

    xs, ys = map((dom, n) -> opts[:domain_type](dom..., n), domain, opts[:gridsize])
    (Dx₋, Dx₊, Hx), (Dy₋, Dy₊, Hy) = dp_operator.(
        Ref(opts[:deriv_type]),
        Ref(opts[:boundary]),
        Ref(opts[:deriv_order]),
        (xs, ys)
    )

    total_energy = Vector{Float64}(undef, length(pvd.timesteps))
    total_mass = Vector{Float64}(undef, length(pvd.timesteps))
    total_vorticity = Vector{Float64}(undef, length(pvd.timesteps))
    total_enstrophy = Vector{Float64}(undef, length(pvd.timesteps))

    # plot initial conditions
    let
        fig = Figure()
        xs, ys = map((dom, n) -> opts[:domain_type](dom..., n), domain, opts[:gridsize])
        vtk = VTKFile(datadir(dirname(pvdpath), pvd.vtk_filenames[begin]))
        pdata = get_point_data(vtk)
        pv = dropdims(get_data_reshaped(pdata["PV"]), dims = 3)
        ax = Axis(fig[1, 1],
            ylabel = L"$y$",
            xlabel = L"$x$",
            aspect = DataAspect(),
            width = 160,
            height = 160
        )
        hm = heatmap!(ax, xs, ys, pv,
            colormap = :seaborn_icefire_gradient,
            colorrange = (-1.2, 1.2),
            rasterize = 4
        )
        Colorbar(fig[1, 2], hm, label = L"\frac{ω}{h}")
        resize_to_layout!(fig)
        wsave(plotsdir(output_dir, "initial_conditions.pdf"), fig)
    end

    for (i, (timestep, vtk_file)) in enumerate(zip(pvd.timesteps, pvd.vtk_filenames))
        vtk = VTKFile(datadir(dirname(pvdpath), vtk_file))
        pdata = get_point_data(vtk)
        pv = dropdims(get_data_reshaped(pdata["PV"]), dims = 3)
        h = dropdims(get_data_reshaped(pdata["h"]), dims = 3)
        u = dropdims(selectdim(get_data_reshaped(pdata["u"]), 1, 1), dims = 3)
        v = dropdims(selectdim(get_data_reshaped(pdata["u"]), 1, 2), dims = 3)
        ω = dropdims(get_data_reshaped(pdata["ω"]), dims = 3)

        total_energy[i] = energy_norm((h, u, v), Hx, Hy) do (h, u, v)
            1 // 2 * h * (u^2 + v^2) + 1 // 2 * g * h^2
        end

        total_mass[i] = energy_norm((h, u, v), Hx, Hy) do (h, u, v)
            h
        end

        total_vorticity[i] = energy_norm((ω,), Hx, Hy) do (ω,)
            ω
        end

        total_enstrophy[i] = energy_norm((ω, h), Hx, Hy) do (ω, h)
            ω^2 / h
        end

        fig = Figure()
        ax = Axis(fig[1, 1],
            ylabel = L"$y$",
            xlabel = L"$x$",
            aspect = DataAspect(),
            title = latexstring(@sprintf "t = %0.3e" timestep),
            width = 200,
            height = 200
        )
        hm = heatmap!(ax, xs, ys, pv,
            colormap = :seaborn_icefire_gradient, #:seismic,
            colorrange = (-1.2, 1.2)
        )
        Colorbar(fig[:, end + 1], hm, label = L"\frac{ω}{h}")
        resize_to_layout!(fig)
        wsave(plotsdir(output_dir, @sprintf "%05d.png" i), fig)
    end

    push!(globals,
        opts[:scheme] => (;
            time = pvd.timesteps,
            energy = total_energy,
            mass = total_mass,
            vorticity = total_vorticity,
            enstrophy = total_enstrophy
        ))

    fig = Figure(size = (300, 200))
    ax = Axis(fig[1, 1])
    lines!(ax, pvd.timesteps, (total_energy .- total_energy[begin]) ./ total_energy[begin])
    wsave(plotsdir(output_dir, "energy.png"), fig)

    fig = Figure(size = (300, 200))
    ax = Axis(fig[1, 1])
    lines!(ax, pvd.timesteps, (total_mass .- total_mass[begin]) ./ total_mass[end])
    wsave(plotsdir(output_dir, "mass.png"), fig)

    fig = Figure(size = (300, 200))
    ax = Axis(fig[1, 1])
    lines!(ax, pvd.timesteps,
        (total_vorticity .- total_vorticity[begin]) ./ total_vorticity[end])
    wsave(plotsdir(output_dir, "vorticity.png"), fig)

    fig = Figure(size = (300, 200))
    ax = Axis(fig[1, 1])
    lines!(ax, pvd.timesteps,
        (total_enstrophy .- total_enstrophy[begin]) ./ total_enstrophy[end])
    wsave(plotsdir(output_dir, "enstrophy.png"), fig)
end

# compare relative dissipation of global constants
let
    odir = joinpath("swe-2D", "$testname")
    axwidth = 190
    axheight = 130

    function plot_relative_dissipation(qty, (qstr, q0str), yscale, yticks, ylims)
        println("Plotting ", qty)
        fig = Figure()
        ax = Axis(fig[1, 1],
            ylabel = L"%$qstr/%$q0str - 1",
            xlabel = L"t",
            yscale = pseudolog_tol(yscale),
            yticks = yticks,
            yminorticks = IntervalsBetween(5),
            ytickformat = pows10_ytickformatter,
            width = axwidth,
            height = axheight
        )
        for (name, g) in pairs(globals)
            q0 = g[qty][begin]
            qs = (g[qty] .- q0) ./ q0
            lines!(ax, g[:time], qs,
                label = scheme_labels[name],
                color = scheme_colors[name]
            )
        end

        isnothing(ylims) || ylims!(ax, ylims...)

        resize_to_layout!(fig)
        wsave(plotsdir(odir, "global-quantities-$qty.pdf"), fig)

        fig = Figure()
        Legend(fig[1, 1], ax, halign = :center, orientation = :horizontal,
            tellwidth = true, tellheight = true)
        resize_to_layout!(fig)
        wsave(plotsdir(odir, "global-quantities-legend-$qty.pdf"), fig)
    end

    plot_relative_dissipation(:energy, (L"E", L"E_0"),
        1e-8, [0, -1e-7, -1e-6, -1e-5], nothing)

    plot_relative_dissipation(:mass, (L"⟨1, h⟩_H", L"⟨1, h_0⟩_H"),
        1e-12, [-1e-10, -1e-11, 0.0, 1e-11, 1e-10], (-1e-12, 1e-10))

    plot_relative_dissipation(:vorticity, (L"⟨1, ω⟩_H", L"⟨1, ω_0⟩_H"),
        1e-16, [-1e-14, -1e-15, 0.0, 1e-15, 1e-14], (-1e-14, 1e-14))

    plot_relative_dissipation(:enstrophy, (L"⟨1, \mathcal{E}⟩_H", L"⟨1, \mathcal{E}_0⟩_H"),
        1e-3, [-1e-1, -1e-2, 0.0, 1e-2, 1e-1], (-1e-1, 1e-1))
end

# compare potential vorticity at chosen times
let comparison_ts = [
        1.0012813370251192,
        3.003844011075359,
        6.0076880221500035,
        12.015376044302721
    ]
    fig = Figure()
    for (row, t) in enumerate(comparison_ts)
        Label(fig[row, 1], latexstring(@sprintf "t = %.1f" t))
        hm = ()
        for (i, opts) in enumerate(allopts)
            xs, ys = map((dom, n) -> opts[:domain_type](dom..., n), domain, opts[:gridsize])
            output_dir = make_output_dirname(opts)
            pvdpath = datadir(output_dir, "$testname.pvd")
            vtk = VTKFile(joinpath(dirname(pvdpath), "time_$t.vti"))
            pdata = get_point_data(vtk)
            pv = dropdims(get_data_reshaped(pdata["PV"]), dims = 3)
            ax = Axis(fig[row, begin + i],
                ylabel = L"$y$",
                xlabel = L"$x$",
                aspect = DataAspect(),
                width = 140,
                height = 140
            )
            hm = heatmap!(ax, xs, ys, pv,
                colormap = :seaborn_icefire_gradient,
                colorrange = (-1.2, 1.2),
                rasterize = 4
            )
        end
        Colorbar(fig[row, 2 + length(allopts)], hm, label = L"\frac{ω}{h}")
    end
    resize_to_layout!(fig)
    wsave(plotsdir("swe-2D", "$testname", "comparison.pdf"), fig)
end
