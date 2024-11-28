using DrWatson
@quickactivate :HyperbolicPDEs
using Printf
using WriteVTK, ReadVTK
using SummationByPartsOperators
using Makie, CairoMakie
using OrdinaryDiffEqSSPRK
using DiffEqCallbacks
using LaTeXStrings
using Base.Threads: @threads
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include(srcdir("PDEs", "compressible-euler.jl"))
include(srcdir("makie-theme.jl"))
include(srcdir("plotting-utils.jl"))
set_theme!(hpdes_makie_theme)

# An instability caused by a velocity difference across the interface
# between two fluids
const testname = "kelvin-helmholtz"
const γ = 1.4

const tspan      = (0.0, 10.0)
const Δt         = (Δx, Δy) -> 0.1 * Δx * Δy / (Δx + Δy)
const domain     = [(-1, 1), (-1, 1)]
const save_every = 0.1
const state0     = let
    B(x, y) = tanh(15 * y + 7.5) - tanh(15 * y - 7.5)
    ϱ = (x, y) -> 0.5 + 0.75 * B(x, y)
    u = (x, y) -> 0.5 * (B(x, y) - 1.0)
    v = (x, y) -> sinpi(2x) / 10
    p = (x, y) -> 1.0

    (ϱ, u, v, p)
end

const schemes = Dict(
    "Nordstrom (γ=0)"  => NordstromCompEuler2D(γ = γ, flux_splitting = nothing),
    "Nordstrom (γ>0)"  => NordstromCompEuler2D(γ = γ, flux_splitting = FluxEntropyStable()),
    "Van Leer-Hänel"   => VanLeerHanelCompEuler2D(γ = γ),
    "Lax-Fried. (γ>0)" => CompEulerFluxForm2D(γ = γ, flux_splitting = FluxLaxFriedrichs()),
    "Flux Form"        => CompEulerFluxForm2D(γ = γ, flux_splitting = nothing)
)

const label_colors = Dict(
    "entropy conserving" => Makie.wong_colors()[2],
    "entropy stable"     => Makie.wong_colors()[3],
    "Van Leer-Hänel"     => Makie.wong_colors()[4],
    "Lax-Fried."         => Makie.wong_colors()[1],
    "Flux Form"          => Makie.wong_colors()[5]
)

function make_label(opts)
    Dict(
        "Nordstrom (γ=0)"  => "entropy conserving",
        "Nordstrom (γ>0)"  => "entropy stable",
        "Van Leer-Hänel"   => "Van Leer-Hänel",
        "Lax-Fried. (γ>0)" => "Lax-Fried.",
        "Flux Form"        => "Flux Form"
    )[opts[:scheme]]
end

function resume_from_time(pvdpath::String, time::Real, ts)
    pvd   = PVDFile(pvdpath)
    i     = searchsortedfirst(pvd.timesteps, time)
    vtk   = VTKFile(dirname(pvdpath) * "/" * pvd.vtk_filenames[i])
    pdata = get_point_data(vtk)

    ϱ = dropdims(get_data_reshaped(pdata["ϱ"]), dims = 3)
    u = dropdims(selectdim(get_data_reshaped(pdata["u"]), 1, 1), dims = 3)
    v = dropdims(selectdim(get_data_reshaped(pdata["u"]), 1, 2), dims = 3)
    p = dropdims(get_data_reshaped(pdata["p"]), dims = 3)
    return ts[searchsortedfirst(ts, time):end], (ϱ, u, v, p)
end

function solve_scheme(odir::String, opts; resume_from = nothing)
    pdeinfo = schemes[opts[:scheme]]
    gridsize = fill(opts[:gridsize], 2)

    xs = map((dom, n) -> opts[:domain_type](dom..., n), domain, gridsize)
    (Dx₋, Dx₊, Hx), (Dy₋, Dy₊, Hy) = dp_operator.(
        Ref(opts[:deriv_type]),
        Ref(opts[:boundary]),
        Ref(opts[:deriv_order]),
        xs
    )

    ∂y(m) = mapslices(col -> (Dy₋ + Dy₊) / 2 * col, m; dims = (2))
    ∂x(m) = mapslices(row -> (Dx₋ + Dx₊) / 2 * row, m; dims = (1))
    vorticity((ϱ, u, v, p)) = ∂x(v) - ∂y(u)

    ts = tspan[1]:Δt(step.(xs)...):tspan[2]
    ts, s₀ = if isnothing(resume_from)
        ts, meshgrid.(state0, Ref.(xs)...)
    else
        resume_from_time(joinpath(odir, "$testname.pvd"), resume_from, ts)
    end

    pde! = semidiscretise(pdeinfo, s₀, Dx₊, Dx₋, Dy₊, Dy₋)
    prob = ODEProblem(pde!, from_primitive_vars(pdeinfo, s₀), tspan)

    mkpath(odir)
    open(joinpath(odir, "settings.txt"); write = true) do io
        show(io, MIME("text/plain"), opts)
        println(io, "\nderived parameters:\n")
        print(io, "  ts = ")
        show(io, MIME("text/plain"), ts)
    end

    println(odir)
    crash_time = Ref(NaN64)
    paraview_collection(joinpath(odir, "$testname.pvd")) do pvd
        primitive_vars = map(similar, s₀)

        last_saved = Ref(-save_every)

        solve(prob, SSPRK54();
            dt = step(ts),
            save_on = false,
            progress = true,
            unstable_check = (_, u, _, t) -> any(x -> isinf(x) || isnan(x), u) ?
                                             (crash_time[] = t; true) : false,
            callback = FunctionCallingCallback(
                (u, t, _) -> begin
                    approxless(last_saved[] + save_every, t) || return ()
                    last_saved[] = last_saved[] + save_every
                    to_primitive_vars!(primitive_vars, pdeinfo, u)
                    vtk_grid(joinpath(odir, "time_$t.vti"), xs...) do vtk
                        vtk["ϱ"] = primitive_vars[1]
                        vtk["u"] = @views (primitive_vars[2], primitive_vars[3])
                        vtk["p"] = primitive_vars[4]

                        vtk["ω"] = vorticity(primitive_vars)

                        pvd[t] = vtk
                    end
                    return ()
                end;
                func_everystep = true),
            progress_steps = 250,
            progress_id = Symbol(odir),
            progress_name = rpad(make_label(opts), 30)
        )
    end
    open(joinpath(odir, "crash_time.txt"); write = true) do io
        print(io, crash_time[])
    end
    return ()
end

# Compare Schemes

const testopts = dict_list(Dict(
    :deriv_type  => Mattsson2017,
    :boundary    => PeriodicSAT(),
    :domain_type => bounded_range
))

const runopts = dict_list(Dict(
    :scheme      => ["Nordstrom (γ=0)", "Nordstrom (γ>0)", "Van Leer-Hänel", "Lax-Fried. (γ>0)"],
    :gridsize    => [512],
    :deriv_order => [6]
))

for testopt in testopts
    testhash = hash(testopt)
    testtag  = savename(testopt) * "_" * string(testhash; base = 60)
    testdir  = joinpath("comp-euler-2D", testname, testtag)

    names          = [make_label(merge(runopt, testopt)) for runopt in runopts]
    crash_times    = Vector{Float64}(undef, length(names))
    allglobal_qtys = Vector{NamedTuple}(undef, length(names))

    println("Testcase:")
    display(testopt)

    @threads for runidx in eachindex(runopts)
        runopt  = runopts[runidx]
        opts    = merge(runopt, testopt)
        runhash = hash(opts)
        runtag  = savename(opts) * "_" * string(runhash; base = 60)
        rundir  = joinpath("comp-euler-2D", testname, runtag)

        solve_scheme(datadir(rundir), opts)

        name    = names[runidx]
        pdeinfo = schemes[opts[:scheme]]

        gridsize = fill(opts[:gridsize], 2)
        xs = map(domain, gridsize) do dom, n
            opts[:domain_type](dom..., n)
        end

        (Dx₋, Dx₊, Hx), (Dy₋, Dy₊, Hy) = dp_operator.(
            Ref(opts[:deriv_type]),
            Ref(opts[:boundary]),
            Ref(opts[:deriv_order]),
            xs
        )

        crash_time = parse(Float64, datadir(rundir, "crash_time.txt") |> read |> String)
        ts, global_qtys = let
            pvdpath = datadir(rundir, "$testname.pvd")
            pvd = PVDFile(pvdpath)

            global_qtys = []

            for vtk_file in pvd.vtk_filenames
                vtk   = VTKFile(joinpath(dirname(pvdpath), vtk_file))
                pdata = get_point_data(vtk)

                ϱ = dropdims(get_data_reshaped(pdata["ϱ"]), dims = 3)
                u = dropdims(selectdim(get_data_reshaped(pdata["u"]), 1, 1), dims = 3)
                v = dropdims(selectdim(get_data_reshaped(pdata["u"]), 1, 2), dims = 3)
                p = dropdims(get_data_reshaped(pdata["p"]), dims = 3)
                push!(global_qtys, global_cons_qtys(pdeinfo, (ϱ, u, v, p), Hx, Hy))
            end

            pvd.timesteps, global_qtys
        end

        globals = (;
            time   = ts,
            energy = getindex.(global_qtys, 1),
            ϱ      = getindex.(global_qtys, 2),
            ϱu     = getindex.(global_qtys, 3),
            ϱv     = getindex.(global_qtys, 4),
            s      = getindex.(global_qtys, 5)
        )

        allglobal_qtys[runidx] = globals
        crash_times[runidx] = crash_time
    end

    globals = zip(crash_times, names, allglobal_qtys) |> collect |> sort |> reverse

    println("Plotting evolution of global variables to ", plotsdir(testdir))
    axwidth = 190
    axheight = 130

    function plot_relative_dissipation(qty, ylabel, yscale, yticks, ylims = nothing)
        println("Plotting ", qty)
        fig = Figure()
        ax = Axis(fig[1, 1],
            ylabel = ylabel,
            xlabel = L"t",
            yscale = pseudolog_tol(yscale),
            yticks = yticks,
            yminorticks = IntervalsBetween(5),
            ytickformat = pows10_ytickformatter,
            width = axwidth,
            height = axheight
        )
        for (_, name, g) in globals
            q0 = g[qty][begin]
            qs = (g[qty] .- q0) ./ q0
            lines!(ax, g[:time], qs, label = name, color = label_colors[name])
        end
        resize_to_layout!(fig)
        isnothing(ylims) || ylims!(ax, ylims...)
        wsave(plotsdir(testdir, "global-quantities-$qty.pdf"), fig)

        fig = Figure()
        Legend(fig[1, 1], ax, halign = :center, orientation = :horizontal,
            tellwidth = true, tellheight = true)
        resize_to_layout!(fig)
        wsave(plotsdir(testdir, "global-quantities-legend-$qty.pdf"), fig)
    end

    plot_relative_dissipation(:energy, L"E/E_0 - 1",
        1e-6, [0, -1e-5, -1e-4, -1e-3])

    plot_relative_dissipation(:ϱ, L"⟨1, ϱ⟩_H / ⟨1, ϱ_0⟩_H - 1",
        1e-12, [-1e-11, 0.0, 1e-11, 1e-10, 1e-9], (-1e-11, 1e-9))

    plot_relative_dissipation(:ϱu, L"⟨1, ϱu⟩_H / ⟨1, ϱu_0⟩_H - 1",
        1e-12, [-1e-11, 0.0, 1e-11, 1e-10, 1e-9], (-1e-11, 1e-9))

    plot_relative_dissipation(:s, L"⟨1, ϱs⟩_H / ⟨1, ϱs_0⟩_H - 1",
        1e-4, [0, -1e-3, -1e-2, -1e-1])

    let qty = :ϱv
        println("Plotting ", qty)
        fig = Figure()
        ax = Axis(fig[1, 1],
            ylabel = L"⟨1, ϱv⟩_H - ⟨1, ϱv_0⟩_H",
            xlabel = L"t",
            yscale = pseudolog_tol(1e-12),
            yticks = [-1e-10, -1e-11, 0, 1e-11, 1e-10],
            ytickformat = pows10_ytickformatter,
            yminorticks = IntervalsBetween(5),
            width = axwidth,
            height = axheight
        )
        for (_, name, g) in globals
            q0 = g[qty][1]
            qs = g[qty] .- q0
            lines!(ax, g[:time], qs, label = name, color = label_colors[name])
        end
        ylims!(-1e-10, 1e-10)
        resize_to_layout!(fig)
        wsave(plotsdir(testdir, "global-quantities-$qty.pdf"), fig)
    end

    display(Dict(zip(names, crash_times)))
end
