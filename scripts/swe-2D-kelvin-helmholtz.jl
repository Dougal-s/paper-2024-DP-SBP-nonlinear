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
set_theme!(hpdes_makie_theme)

const f = 7.292e-5
const g = 9.80616
const ω_colorrange = (-3.0e-5 + f, 3.0e-5 + f)

# An instability caused by a velocity difference across the interface
# between two fluids
const testname   = "kelvin-helmholtz"
const tspan      = (0.0, 80 * 24 * 3600.0)
const Δt         = (Δx, Δy) -> Δx * Δy / (400Δx + 400Δy)
const save_every = 12 * 3600.0
const Lx         = 2π * 6371.22e3
const Ly         = 2π * 6371.22e3
const domain     = ((0.0, Lx), (0.0, Ly))
const state0     = let
    d((x₁, y₁), (x₂, y₂)) = ((x₁ - x₂) / Lx)^2 + ((y₁ - y₂) / Ly)^2

    H    = 10e3
    u₀   = 50
    kᵥ   = 1.0e-6
    k    = 1000
    jets = [(u₀, 0.25Ly), (-u₀, 0.75Ly)]
    ps   = [(0.85Lx, 0.75Ly), (0.15Lx, 0.25Ly)]

    uᵢ(x, y, (uᵢ, yᵢ))  = uᵢ * sech(kᵥ * (y - yᵢ))
    ∫uᵢ(x, y, (uᵢ, yᵢ)) = uᵢ / kᵥ * 2atan(exp(kᵥ * (y - yᵢ)))

    h̃(x, y) = 0.01H * sum(exp(-k * d((x, y), p)) for p in ps)

    h = (x, y) -> H - f / g * sum(∫uᵢ(x, y, j) - ∫uᵢ(x, 0, j) for j in jets) + h̃(x, y)
    u = (x, y) -> sum(uᵢ(x, y, j) for j in jets)
    v = (x, y) -> zero(x)
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
    :gridsize    => [(256, 256)],
    :deriv_order => 6,
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

    Dx = (Dx₋ + Dx₊) / 2
    Dy = (Dy₋ + Dy₊) / 2
    ∂x(m) = ∂x2D(Dx, m)
    ∂y(m) = ∂y2D(Dy, m)
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

                        vtk["h"] = primitive_vars[1]
                        vtk["u"] = @views (primitive_vars[2], primitive_vars[3])
                        vtk["ω"] = ω
                        pvd[t]   = vtk
                    end
                    return ()
                end;
                func_everystep = true),
            progress = true,
            progress_steps = 250,
            progress_id = Symbol(hash(opts)),
            progress_name = rpad(repr(opts[:scheme]), 20)
        )
    end

    open(datadir(output_dir, "crash_time.txt"); write = true) do io
        show(io, MIME("text/plain"), crash_time[])
    end
end

for opts in allopts
    output_dir = make_output_dirname(opts)
    mkpath(datadir(output_dir))

    pvdpath = datadir(output_dir, "$testname.pvd")
    pvd = PVDFile(pvdpath)
    nsteps = length(pvd.timesteps)

    xs, ys = map((dom, n) -> opts[:domain_type](dom..., n), domain, opts[:gridsize])

    # plot initial conditions
    let
        fig = Figure()
        xs, ys = map((dom, n) -> opts[:domain_type](dom..., n), domain, opts[:gridsize])
        vtk = VTKFile(datadir(dirname(pvdpath), pvd.vtk_filenames[begin]))
        pdata = get_point_data(vtk)
        ω = dropdims(get_data_reshaped(pdata["ω"]), dims = 3)
        ax = Axis(fig[1, 1],
            ylabel = L"$y$ (1000 km)",
            xlabel = L"$x$ (1000 km)",
            aspect = DataAspect(),
            width = 160,
            height = 160
        )
        hm = heatmap!(ax, xs ./ 1e6, ys ./ 1e6, ω,
            colormap = :seismic,
            colorrange = ω_colorrange,
            rasterize = 4
        )
        Colorbar(fig[1, 2], hm, label = L"ω")
        resize_to_layout!(fig)
        wsave(plotsdir(output_dir, "initial_conditions.pdf"), fig)
    end

    for (frameno, (timestep, vtk_file)) in enumerate(zip(pvd.timesteps, pvd.vtk_filenames))
        vtk = VTKFile(joinpath(dirname(pvdpath), vtk_file))
        pdata = get_point_data(vtk)
        ω = dropdims(get_data_reshaped(pdata["ω"]), dims = 3)

        fig = Figure()
        ax = Axis(fig[1, 1],
            ylabel = L"$y$ (1000 km)",
            xlabel = L"$x$ (1000 km)",
            title = latexstring(@sprintf "t = %0.3e" timestep),
            aspect = DataAspect(),
            width = 200,
            height = 200
        )
        hm = heatmap!(ax, xs ./ 1e6, ys ./ 1e6, ω,
            colormap = :seismic,
            colorrange = ω_colorrange
        )
        Colorbar(fig[:, end + 1], hm, label = L"ω")
        resize_to_layout!(fig)
        wsave(plotsdir(output_dir, @sprintf "%05d.png" frameno), fig)
    end
end

let comparison_ts = [
        1.5178631609354524e6,
        2.0382733875416827e6,
        3.035726321871404e6,
        4.0331792562011955e6
    ]
    fig = Figure()
    for (row, t) in enumerate(comparison_ts)
        if t == 0
            Label(fig[row, 1], L"t = 0×10^0")
        else
            texp = floor(Int64, log10(t))
            tfrac = t / exp10(texp)
            Label(fig[row, 1], latexstring(@sprintf "t = %.1f×10^{%d}s" tfrac texp))
        end
        hm = ()
        for (i, opts) in enumerate(allopts)
            xs, ys = map((dom, n) -> opts[:domain_type](dom..., n), domain, opts[:gridsize])
            output_dir = make_output_dirname(opts)
            pvdpath = datadir(output_dir, "$testname.pvd")
            vtk = VTKFile(joinpath(dirname(pvdpath), "time_$t.vti"))
            pdata = get_point_data(vtk)
            ω = dropdims(get_data_reshaped(pdata["ω"]), dims = 3)
            ax = Axis(fig[row, begin + i],
                ylabel = L"$y$ (1000 km)",
                xlabel = L"$x$ (1000 km)",
                aspect = DataAspect(),
                width = 140,
                height = 140
            )
            hm = heatmap!(ax, xs ./ 1e6, ys ./ 1e6, ω,
                colormap = :seismic,
                colorrange = ω_colorrange,
                rasterize = 4
            )
        end
        Colorbar(fig[row, 2 + length(allopts)], hm, label = L"ω")
    end
    resize_to_layout!(fig)
    wsave(plotsdir("swe-2D", "$testname", "comparison.pdf"), fig)
end
