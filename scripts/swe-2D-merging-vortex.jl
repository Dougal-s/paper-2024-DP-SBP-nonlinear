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

const f = 5.0 # coriolis
const g = 5.0 # gravity
const H = 8.0 # base height

const colormap = :seaborn_icefire_gradient
const pv_colorrange = (-0.3, 0.9)

# The collision and merging of two vortices
const testname   = "merging-vortex"
const tspan      = (0.0, 20.0)
const Δt         = Δx -> 1e-2minimum(Δx)
const save_every = 0.1
const domain     = ((0.0, 2π), (0.0, 2π))
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
    h = ((x, y),) -> H + f / g * ψ(x, y)
    u = ((x, y),) -> -∂yψ(x, y)
    v = ((x, y),) -> ∂xψ(x, y)
    b = ((x, y),) -> zero(x)

    (h, u, v, b)
end

const schemes = Dict(
    "Flux Form"         => ShallowWaterFluxForm2D(; f, g),
    "Lax-Fried."        => ShallowWaterFluxForm2D(FluxLaxFriedrichs(); f, g),
    "Skew Symm. (γ=0)"  => ShallowWaterSkewSym2D(; f, g),
    "Skew Symm. (γ>0)"  => ShallowWaterSkewSym2D(FluxEntropyStable(); f, g),
    "Vector Inv. (γ=0)" => ShallowWaterVectorInv2D(; f, g),
    "Vector Inv. (γ>0)" => ShallowWaterVectorInv2D(FluxEntropyStable(); f, g)
)

function colors(opts)
    Dict(
        "Lax-Fried."       => Makie.wong_colors()[1],
        "Skew Symm. (γ=0)" => Makie.wong_colors()[7],
        "Skew Symm. (γ>0)" => Makie.wong_colors()[2],
        "Flux Form"        => Makie.wong_colors()[4]
    )[opts[:scheme]]
end

function labels(opts)
    get(
        Dict(
            "Lax-Fried."       => "linearly stable DP DG/FD",
            "Skew Symm. (γ=0)" => "DGSEM/SBP FD",
            "Skew Symm. (γ>0)" => "DP DG/FD"
        ),
        opts[:scheme],
        opts[:scheme]
    )
end

function filetag(opts)
    pdeinfo = schemes[opts[:scheme]]
    params_hash = string(hash(pdeinfo)^hash(opts); base = 60)
    return savename(opts) * "_" * params_hash
end

make_output_dirname(opts) = joinpath("swe-2D", testname, filetag(opts))

allopts = merge.(
    Ref(Dict(
        :timestepper => (; solver = SSPRK54(), args = (;))
    )),
    mapreduce(dict_list, vcat,
        [
            Dict(
                :scheme     => ["Skew Symm. (γ=0)", "Skew Symm. (γ>0)"],
                :elems      => 32,
                :nodes      => Derived(:order, p -> p + 1),
                :order      => 6,
                :deriv_type => GlaubitzEtal2024(-0.1)
            ),
            Dict(
                :scheme     => ["Skew Symm. (γ=0)", "Skew Symm. (γ>0)"],
                :elems      => 6,
                :nodes      => 33,
                :order      => 6,
                :deriv_type => Mattsson2017
            )
        ]
    )
)

@threads for opts in allopts
    pdeinfo = schemes[opts[:scheme]]

    @unpack order, nodes, elems = opts

    mesh = CartesianMesh(domain, (elems, elems))
    cell = local_dp_operator(opts[:deriv_type], order, (nodes, nodes))
    xs, fdop = couple_operators(mesh, cell)

    s₀ = map(f -> f.(xs), state0)

    permute_cache = similar(s₀[1])
    ∂x(m) = axis_mul!(similar(m), Val(1), fdop[1].D + fdop[1].B, m, permute_cache)
    ∂y(m) = axis_mul!(similar(m), Val(2), fdop[2].D + fdop[2].B, m, permute_cache)

    vorticity((h, u, v)) = ∂x(v) .- ∂y(u) .+ f

    pde! = semidiscretise(pdeinfo, xs, fdop)
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(
        pde!, from_primitive_vars(pdeinfo, s₀), tspan)

    output_dir = make_output_dirname(opts)
    mkpath(datadir(output_dir))

    open(datadir(output_dir, "settings.txt"); write = true) do io
        show(io, MIME("text/plain"), opts)
    end

    println(output_dir)
    paraview_collection(datadir(output_dir, "$testname.pvd")) do pvd
        primitive_vars = map(similar, s₀)

        sol = solve(prob, opts[:timestepper].solver;
            opts[:timestepper].args...,
            dt = Δt(step(xs)),
            save_on = false,
            callback = FunctionCallingCallback(;
                funcat = first(tspan):save_every:last(tspan)
            ) do u, t, _
                to_primitive_vars!(primitive_vars, pdeinfo, u)
                vtk_grid(datadir(output_dir, "time_$t.vtr"), getaxis(xs)...) do vtk
                    ω = vorticity(primitive_vars)

                    vtk["h"] = primitive_vars[1]
                    vtk["u"] = @views (primitive_vars[2], primitive_vars[3])
                    vtk["ω"] = ω
                    pvd[t]   = vtk
                end
            end,
            progress = true,
            progress_steps = 250,
            progress_id = Symbol(hash(opts)),
            progress_name = rpad(opts[:scheme], 20)
        )

        open(datadir(output_dir, "results.txt"); write = true) do io
            println(io, "finaltime:\t", last(sol.t))
            println(io, "success:\t", SciMLBase.successful_retcode(sol.retcode))
            println(io, "reason:\t", sol.retcode)
            show(io, MIME("text/plain"), sol.stats)
        end
    end
end

for opts in allopts
    output_dir = make_output_dirname(opts)
    mkpath(datadir(output_dir))

    pvdpath = datadir(output_dir, "$testname.pvd")
    pvd = PVDFile(pvdpath)

    for (i, (t, vtk_file)) in enumerate(zip(pvd.timesteps, pvd.vtk_filenames))
        vtk   = VTKFile(joinpath(dirname(pvdpath), vtk_file))
        xs    = get_coordinates(vtk)[1:2]
        pdata = get_point_data(vtk)
        h     = dropdims(get_data_reshaped(pdata["h"]), dims = 3)
        ω     = dropdims(get_data_reshaped(pdata["ω"]), dims = 3)

        fig = Figure()
        ax = Axis(fig[1, 1],
            title = latexstring(@sprintf "t = %0.3e" t),
            xlabel = L"x", ylabel = L"y",
            aspect = DataAspect(),
            width = 200, height = 200,
            xticksmirrored = true, yticksmirrored = true
        )
        hm = heatmap!(ax, xs..., ω ./ h;
            colormap, colorrange = pv_colorrange
        )
        Colorbar(fig[:, end + 1], hm, label = L"\frac{ω}{h}")
        resize_to_layout!(fig)
        wsave(plotsdir(output_dir, @sprintf "%05d.png" i), fig)
    end
end

globals = map(allopts) do opts
    output_dir = make_output_dirname(opts)
    mkpath(datadir(output_dir))

    pvdpath = datadir(output_dir, "$testname.pvd")
    pvd = PVDFile(pvdpath)

    @unpack order, nodes, elems = opts

    mesh = CartesianMesh(domain, (elems, elems))
    cell = local_dp_operator(opts[:deriv_type], order, (nodes, nodes))
    xs, fdop = couple_operators(mesh, cell)
    ∫_Ω = VolumeMeasure(xs, fdop)

    total_energy = Vector{Float64}(undef, length(pvd.timesteps))
    total_enstrophy = Vector{Float64}(undef, length(pvd.timesteps))

    for (i, vtk_file) in enumerate(pvd.vtk_filenames)
        vtk = VTKFile(joinpath(dirname(pvdpath), vtk_file))
        pdata = get_point_data(vtk)
        h = dropdims(get_data_reshaped(pdata["h"]), dims = 3)
        u = dropdims(selectdim(get_data_reshaped(pdata["u"]), 1, 1), dims = 3)
        v = dropdims(selectdim(get_data_reshaped(pdata["u"]), 1, 2), dims = 3)
        ω = dropdims(get_data_reshaped(pdata["ω"]), dims = 3)

        total_energy[i] = ∫_Ω((h, u, v)) do (h, u, v)
            1 // 2 * h * (u^2 + v^2) + 1 // 2 * g * h^2
        end
        total_enstrophy[i] = ∫_Ω((ω, h)) do (ω, h)
            ω^2 / h
        end
    end

    (;
        time = pvd.timesteps,
        energy = total_energy,
        enstrophy = total_enstrophy
    )
end

# compare relative dissipation of global constants
let
    odir = joinpath("swe-2D", "$testname")

    function plot_relative_dissipation(qty, label, yscale, yticks, ylims)
        println("Plotting ", qty)
        fig = Figure()
        ax = Axis(fig[1, 1],
            title = label,
            xlabel = L"t",
            yscale = pseudolog_tol(yscale),
            yticks = sort(yticks; rev = true),
            yminorticks = IntervalsBetween(5),
            yminorgridvisible = true,
            ytickformat = pows10_ytickformatter,
            width = 120, height = 100
        )
        for (opts, g) in zip(allopts, globals)
            q0 = g[qty][begin]
            qs = (g[qty] .- q0) ./ q0
            ts = g.time
            lines!(ax, ts, qs; label = labels(opts), color = colors(opts))
        end

        isnothing(ylims) || ylims!(ax, ylims...)

        resize_to_layout!(fig)
        wsave(plotsdir(odir, "global-quantities-$qty.pdf"), fig)

        save_legend(plotsdir(odir, "legend-horizontal.pdf"), ax, :horizontal)
        save_legend(plotsdir(odir, "legend-vertical.pdf"), ax, :vertical)
    end

    plot_relative_dissipation(:energy, "energy/entropy",
        1e-7, [1e-5, 1e-6, 0, -1e-6, -1e-5, -1e-4], (-2e-5, 2e-5))

    plot_relative_dissipation(:enstrophy, "enstrophy",
        1e-2, [1e1, 1e0, 1e-1, 0.0, -1e-1], (-2e-1, 2e1))
end

let comparison_ts = [2.0, 16.0]
    fig = Figure()
    for (i, opts) in enumerate(allopts)
        Label(fig[i, 0], labels(opts); rotation = π / 2)
    end

    for (tᵢ, t) in enumerate(comparison_ts)
        Label(fig[0, tᵢ], latexstring(@sprintf "t = %.0f" t))
        for (optᵢ, opts) in enumerate(allopts)
            xs, pv = let
                pvdpath = datadir(make_output_dirname(opts), "$testname.pvd")
                pvd     = PVDFile(pvdpath)
                _, vtkᵢ = findmin(pvd_t -> abs(pvd_t - t), pvd.timesteps)
                vtk     = VTKFile(joinpath(dirname(pvdpath), pvd.vtk_filenames[vtkᵢ]))
                pdata   = get_point_data(vtk)

                ω = dropdims(get_data_reshaped(pdata["ω"]), dims = 3)
                h = dropdims(get_data_reshaped(pdata["h"]), dims = 3)

                (get_coordinates(vtk)[1:2], ω ./ h)
            end

            ax = Axis(fig[optᵢ, tᵢ],
                ylabel = L"$y$",
                xlabel = L"$x$",
                aspect = DataAspect(),
                width = 85, height = 85,
                xticksmirrored = true, yticksmirrored = true
            )
            optᵢ == lastindex(allopts) ||
                hidexdecorations!(ax; ticks = false, minorticks = false)
            tᵢ == firstindex(comparison_ts) ||
                hideydecorations!(ax; ticks = false, minorticks = false)
            heatmap!(ax, xs..., pv;
                colormap, colorrange = pv_colorrange,
                rasterize = 4
            )
            ax
        end
    end
    Colorbar(fig[1:end, end + 1]; label = L"\frac{ω}{h}",
        height = 130,
        colormap, colorrange = pv_colorrange)

    colgap!(fig.layout, 1, 10)
    rowgap!(fig.layout, 1, 10)

    resize_to_layout!(fig)
    wsave(plotsdir("swe-2D", "$testname", "comparison.pdf"), fig)
end
