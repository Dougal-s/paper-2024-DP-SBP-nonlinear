using DrWatson
@quickactivate :HyperbolicPDEs
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

# problem setup {{{1
# An instability caused by a velocity difference across the interface
# between two fluids parameterised by the Atwood number A ∈ [0, 1) defined by
#     A = (ϱₕᵢ - ϱₗₒ) / (ϱₕᵢ + ϱₗₒ)
#     ϱₕᵢ = (1 + A) / (1 - A) ϱₗₒ
@kwdef struct KelvinHelmholtz
    A::Float64   = 0.6
    ϱₗₒ::Float64 = 0.5
end

function initial_conditions(params::KelvinHelmholtz)
    (; A, ϱₗₒ) = params
    ϱₕᵢ = (1 + A) / (1 - A) * ϱₗₒ

    # smoothed approximation of the rectangle function
    B(x, y) = (tanh(15y + 7.5) - tanh(15y - 7.5)) / 2

    ϱ((x, y)) = ϱₗₒ + (ϱₕᵢ - ϱₗₒ) * B(x, y)
    u((x, y)) = B(x, y) - 0.5
    v((x, y)) = sinpi(2x) / 10
    p((x, y)) = 1.0

    (ϱ, u, v, p)
end

const γ = 1.4

global tspan::NTuple{2, Float64}
const Δt         = (Δx, Δy) -> 0.01min(Δx, Δy)
const domain     = ((-1, 1), (-1, 1))
const volΩ       = abs(prod(splat(-), domain))
const save_every = 0.1

const cons_qtys_metadata = (
    (; type = :rel_diff, qtyname = "mass",
        fn = ((ϱ, v..., p),) -> begin
            ϱ
        end),
    (; type = :rel_diff, qtyname = "x momentum",
        fn = ((ϱ, v..., p),) -> begin
            ϱ * v[1]
        end),
    (; type = :rel_diff, qtyname = "y momentum",
        fn = ((ϱ, v..., p),) -> begin
            ϱ * v[2]
        end),
    (; type = :rel_diff, qtyname = "energy",
        fn = ((ϱ, v..., p),) -> begin
            p / (γ - 1) + ϱ * (v ⋅ v) / 2
        end)
)

const entr_qtys_metadata = (
    (; qtyname = "entropy", ylimits = nothing,
        fn = Base.Fix1(thermodynamic_entropy, γ)
    ),
)

const schemes = Dict(
    "Nordstrom (γ=0)"  => NordstromCompEuler2D(; γ),
    "Nordstrom (γ>0)"  => NordstromCompEuler2D(FluxDSL2025(); γ),
    "Nordstrom (γ>0) [1e-4]"  => NordstromCompEuler2D(FluxDSL2025(; scaling=1e-4); γ),
    "Nordstrom (γ>0) [1e-3]"  => NordstromCompEuler2D(FluxDSL2025(; scaling=1e-3); γ),
    "Nordstrom (γ>0) [1e-2]"  => NordstromCompEuler2D(FluxDSL2025(; scaling=1e-2); γ),
    "Nordstrom (γ>0) [1e-1]"  => NordstromCompEuler2D(FluxDSL2025(; scaling=1e-1); γ),
    "Nordstrom (γ>0) [1e-0]"  => NordstromCompEuler2D(FluxDSL2025(; scaling=1e-0); γ),
    "Van Leer-Hänel"   => CompEulerFluxForm2D(FluxVanLeerHanel(); γ),
    "Lax-Fried. (γ>0)" => CompEulerFluxForm2D(FluxLaxFriedrichs(); γ),
    "Flux Form"        => CompEulerFluxForm2D(; γ)
)

const label_colors = Dict(
    "DGSEM/SBP FD"   => Makie.wong_colors()[7],
    "DP DG/FD"       => Makie.wong_colors()[2],
    "Van Leer-Hänel" => Makie.wong_colors()[4],
    "Lax-Fried."     => Makie.wong_colors()[1],
    "Flux Form"      => Makie.wong_colors()[5]
)

function make_label(opts)
    Dict(
        "Nordstrom (γ=0)"  => "DGSEM/SBP FD",
        "Nordstrom (γ>0)"  => "DP DG/FD",
        "Nordstrom (γ>0) [1e-4]"  => L"10^{-4}",
        "Nordstrom (γ>0) [1e-3]"  => L"10^{-3}",
        "Nordstrom (γ>0) [1e-2]"  => L"10^{-2}",
        "Nordstrom (γ>0) [1e-1]"  => L"10^{-1}",
        "Nordstrom (γ>0) [1e-0]"  => L"1",
        "Van Leer-Hänel"   => "Van Leer-Hänel",
        "Lax-Fried. (γ>0)" => "Lax-Fried.",
        "Flux Form"        => "Flux Form"
    )[opts[:scheme]]
end

make_tag(opts) = savename(opts) * "_" * string(hash(opts); base = 60)

# driver functions {{{1

function resume_from_time(pvdpath::String, time::Real)
    pvd   = PVDFile(pvdpath)
    i     = argmin(i -> abs(time - pvd.timesteps[i]), eachindex(pvd.timesteps))
    vtk   = VTKFile(joinpath(dirname(pvdpath), pvd.vtk_filenames[i]))
    pdata = get_point_data(vtk)

    ϱ = dropdims(get_data_reshaped(pdata["ϱ"]), dims = 3)
    u = dropdims(selectdim(get_data_reshaped(pdata["u"]), 1, 1), dims = 3)
    v = dropdims(selectdim(get_data_reshaped(pdata["u"]), 1, 2), dims = 3)
    p = dropdims(get_data_reshaped(pdata["p"]), dims = 3)
    return pvd.timesteps[i], (ϱ, u, v, p)
end

function solve_scheme(odir::String, opts, testparams::KelvinHelmholtz; runindex, resume_from = nothing)
    pdeinfo = schemes[opts[:scheme]]

    @unpack elems, nodes = opts
    mesh = CartesianMesh(domain, (elems, elems))
    cell = local_dp_operator(opts[:deriv_type], opts[:order], (nodes, nodes))
    xs, fdop = couple_operators(mesh, cell)

    tstart, s₀ = if isnothing(resume_from)
        first(tspan), map(f -> f.(xs), initial_conditions(testparams))
    else
        resume_from_time(joinpath(odir, "$testname.pvd"), resume_from)
    end

    pde! = semidiscretise(pdeinfo, xs, fdop)
    prob = ODEProblem{true}(pde!, from_primitive_vars(pdeinfo, s₀), (tstart, last(tspan)))

    mkpath(odir)
    open(joinpath(odir, "settings.txt"); write = true) do io
        show(io, MIME("text/plain"), opts)
    end

    primitive_vars = map(similar, s₀)
    function write_state(pathname, u, t; pvd = nothing)
        to_primitive_vars!(primitive_vars, pdeinfo, u)
        vtk_grid(pathname, getaxis(xs)...) do vtk
            vtk["ϱ"] = primitive_vars[1]
            vtk["u"] = (primitive_vars[2], primitive_vars[3])
            vtk["p"] = primitive_vars[4]

            isnothing(pvd) || (pvd[t] = vtk)
        end
    end

    println(odir)
    paraview_collection(joinpath(odir, "$testname.pvd")) do pvd
        dt = Δt(step(xs)...)
        sol = solve(prob, opts[:timestepper].solver;
            opts[:timestepper].args..., dt, dtmin = 1e-4dt,
            save_on = false,
            callback = FunctionCallingCallback(;
                funcat = first(tspan):save_every:last(tspan)
            ) do u, t, _
                write_state(joinpath(odir, "time_$t.vtr"), u, t; pvd)
            end,
            progress = true,
            progress_steps = 200,
            progress_id = Symbol(runindex),
            progress_name = rpad(make_label(opts), 30)
        )

        write_state(
            joinpath(odir, "final_state_t=$(sol.t[end]).vtr"), sol.u[end], sol.t[end])

        open(joinpath(odir, "results.txt"); write = true) do io
            println(io, "finaltime:\t", last(sol.t))
            println(io, "success:\t", SciMLBase.successful_retcode(sol.retcode))
            println(io, "reason:\t", sol.retcode)
            show(io, MIME("text/plain"), sol.stats)
        end
    end
    return
end

function gather_quantities(dir, opts)
    @unpack elems, nodes = opts
    mesh = CartesianMesh(domain, (elems, elems))
    cell = local_dp_operator(opts[:deriv_type], opts[:order], (nodes, nodes))
    xs, fdop = couple_operators(mesh, cell)
    ∫_Ω = VolumeMeasure(xs, fdop)

    pvdpath = datadir(dir, "$testname.pvd")
    pvd = PVDFile(pvdpath)

    cons_L1   = Matrix{Float64}(undef, length(pvd.timesteps), length(cons_qtys_metadata))
    cons_qtys = Matrix{Float64}(undef, length(pvd.timesteps), length(cons_qtys_metadata))
    entr_qtys = Matrix{Float64}(undef, length(pvd.timesteps), length(entr_qtys_metadata))

    for (t, vtk_file) in enumerate(pvd.vtk_filenames)
        vtk   = VTKFile(joinpath(dirname(pvdpath), vtk_file))
        pdata = get_point_data(vtk)

        ϱ = dropdims(get_data_reshaped(pdata["ϱ"]), dims = 3)
        u = dropdims(selectdim(get_data_reshaped(pdata["u"]), 1, 1), dims = 3)
        v = dropdims(selectdim(get_data_reshaped(pdata["u"]), 1, 2), dims = 3)
        p = dropdims(get_data_reshaped(pdata["p"]), dims = 3)

        for (i, qty) in enumerate(cons_qtys_metadata)
            cons_L1[t, i]   = ∫_Ω(abs ∘ qty.fn, (ϱ, u, v, p))
            cons_qtys[t, i] = ∫_Ω(qty.fn, (ϱ, u, v, p))
        end
        for (i, qty) in enumerate(entr_qtys_metadata)
            entr_qtys[t, i] = ∫_Ω(qty.fn, (ϱ, u, v, p))
        end
    end

    return pvd.timesteps, cons_L1, cons_qtys, entr_qtys
end

# varying upwind scaling {{{1
# options {{{2
tspan = (0.0, 15.0)
testparams = KelvinHelmholtz(; A = 0.6, ϱₗₒ = 0.5)
testname = savename("kelvin-helmholtz", testparams)

testopts = merge.(
    Ref(Dict(
        :timestepper => (; solver = SSPRK43(), args = (; abstol = 1e-7, reltol = 1e-6))
    )),
    vcat(
        dict_list(Dict(
            :deriv_type => GlaubitzEtal2024(-0.1),
            :elems      => 64,
            :order      => 6,
            :nodes      => Derived(:order, p -> p + 1)
        )),
        dict_list(Dict(
            :deriv_type => Mattsson2017,
            :elems      => 8,
            :nodes      => 64 + 1,
            :order      => 6
        )),
    )
)

runopts = dict_list(Dict(
    :scheme => [
        "Nordstrom (γ>0) [1e-0]",
        "Nordstrom (γ>0) [1e-1]",
        "Nordstrom (γ>0) [1e-2]",
        "Nordstrom (γ>0) [1e-3]",
        "Nordstrom (γ>0) [1e-4]",
        "Nordstrom (γ=0)",
    ]
))

# driver {{{2
for testopt in testopts
    parse_scaling(label) = parse(Float64, replace(label, "DGSEM/SBP FD" => "0", "\$" => "", r"10\^{-(\d)}" => s"1e-\1", "0" => "0"))
    println("Testcase:")
    display_settings(testopt)

    names     = [make_label(merge(runopt, testopt)) for runopt in runopts]

    atwood_numbers = [0.6, 0.8, 0.9]
    crash_times = Matrix{Float64}(undef, length(runopts), length(atwood_numbers))

    for (atwood_index, A) in enumerate(atwood_numbers)
        global testparams = KelvinHelmholtz(; A, ϱₗₒ = 0.5)
        global testname = savename("kelvin-helmholtz", testparams)
        testtag = make_tag(testopt)
        testdir = joinpath("comp-euler-2D", testname, testtag)

        @threads for runidx in eachindex(runopts)
            opts   = merge(runopts[runidx], testopt)
            rundir = joinpath("comp-euler-2D", testname, make_tag(opts))

            solve_scheme(datadir(rundir), opts, testparams; runindex = runidx)
        end

        times     = Vector{Vector{Float64}}(undef, length(names))
        entropies = Vector{Matrix{Float64}}(undef, length(names))

        @threads for runidx in eachindex(runopts)
            opts   = merge(runopts[runidx], testopt)
            rundir = joinpath("comp-euler-2D", testname, make_tag(opts))

            ts, cons_L1, cons, entr = gather_quantities(rundir, opts)

            resultsfile = String(read(datadir(rundir, "results.txt")))
            finaltime = only(match(r"finaltime:\s*([0-9.]*)", resultsfile).captures)
            crash_times[runidx, atwood_index] = parse(Float64, finaltime)
            times[runidx]        = ts
            entropies[runidx]    = entr
        end

        println("Plotting evolution of global variables to ", plotsdir(testdir))
        fig = Figure()
        for (qty_i, entr_qty) in enumerate(entr_qtys_metadata)
            @unpack qtyname, ylimits = entr_qty
            println("plotting total $qtyname")

            fig = Figure()
            ax = Axis(fig[1, 1];
                xlabel = L"t",
                title = qtyname,
                width = 120, height = 100)

            for i in eachindex(times)
                label = names[i] == "DGSEM/SBP FD" ? L"0" : names[i]
                ts = times[i]
                qty = view(entropies[i], :, qty_i)
                scaling = parse_scaling(names[i])
                color = round(Int, scaling == 0 ? 1e-5 : log10(scaling))
                colorrange = [-5, 1]
                lines!(ax, ts, qty; label, color, colorrange, colormap=:inferno)
            end
            resize_to_layout!(fig)
            wsave(plotsdir(testdir, "$qtyname.pdf"), fig)

            save_legend(plotsdir(testdir, "legend-entr-horizontal.pdf"), ax, :horizontal)
            save_legend(plotsdir(testdir, "legend-entr-vertical.pdf"), ax, :vertical)
        end
    end

    testtag = make_tag(testopt)
    testdir = joinpath("comp-euler-2D", testtag)
    println("plotting final times")
    update_theme!(;
        ScatterLines = (; cycle = Cycle([:marker, :color, :linestyle], covary=true))
    )
    let
        fig = Figure()
        ax = Axis(fig[1,1];
                  xlabel = "upwind scaling",
                  xminorgridvisible=true,
                  xscale = pseudolog_tol(1e-5),
                  xtickformat = pows10_ytickformatter,
                  xticks = [10.0 .^ (0:-1:-4); 0.0],
                  xminorticks = IntervalsBetween(4),
                  ylabel = "final time",
                  yscale = log10,
                  yticks = [1, 2.5, 5, 10, 15],
                  yminorgridvisible=true,
                  yminorticks = IntervalsBetween(4),
                  limits = (nothing, (2, 16)),
                  width = 210, height = 130,
                 )
        upwindscaling = map(parse_scaling, names)
        for (Aᵢ, A) in enumerate(atwood_numbers)
            scatterlines!(ax, upwindscaling, crash_times[:,Aᵢ]; label=L"A=%$A")
        end

        lines!(ax, [0, 1], [15, 15]; label=L"T_{\mathrm{final}}", color=:black, linestyle=:dash)
        resize_to_layout!(fig)
        wsave(plotsdir(testdir, "finaltimes.pdf"), fig)

        save_legend(plotsdir(testdir, "legend-finaltimes-horizontal.pdf"), ax, :horizontal)
        save_legend(plotsdir(testdir, "legend-finaltimes-vertical.pdf"), ax, :vertical)
    end
end

# high Atwood number crash times {{{1
# options {{{2
tspan = (0.0, 15.0)

testopts = merge.(
    Ref(Dict(
        :timestepper => (; solver = SSPRK43(), args = (; abstol = 1e-7, reltol = 1e-6))
    )),
    vcat(
        dict_list(Dict(
            :deriv_type => WilliamsDuru2024,
            :elems      => 8,
            :nodes      => 32 .+ 1,
            :order      => 6
        )),
        dict_list(Dict(
            :deriv_type => Mattsson2017,
            :elems      => 8,
            :nodes      => 32 .+ 1,
            :order      => 6
        ))
    )
)

runopts = dict_list(Dict(
    :scheme => ["Nordstrom (γ=0)", "Nordstrom (γ>0)", "Lax-Fried. (γ>0)", "Van Leer-Hänel"]
))

# driver {{{2
for A in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99]
    global testparams = KelvinHelmholtz(; A, ϱₗₒ = 0.5)
    global testname = savename("kelvin-helmholtz-high-atwood", testparams)

    for testopt in testopts
        println("Testcase:")
        display_settings(testopt)

        @threads for runidx in eachindex(runopts)
            opts   = merge(runopts[runidx], testopt)
            rundir = joinpath("comp-euler-2D", testname, make_tag(opts))

            solve_scheme(datadir(rundir), opts, testparams; runindex = runidx)
        end
    end
end

# normal Atwood number crash times {{{1
# options {{{2
tspan = (0.0, 15.0)
testparams = KelvinHelmholtz(; A = 0.6, ϱₗₒ = 0.5)
testname = savename("kelvin-helmholtz", testparams)

testopts = merge.(
    Ref(Dict(
        :timestepper => (; solver = SSPRK43(), args = (; abstol = 1e-7, reltol = 1e-6))
    )),
    vcat(
        dict_list(Dict(
            :deriv_type => GlaubitzEtal2024(-0.1),
            :elems      => [64, 32, 16],
            :nodes      => 7:-1:3 |> collect,
            :order      => Derived(:nodes, N -> N - 1)
        )),
        dict_list(Dict(
            :deriv_type => Mattsson2017,
            :elems      => [8],
            :nodes      => [16, 32, 64] .+ 1,
            :order      => 4:9 |> collect
        )),
        dict_list(Dict(
            :deriv_type => WilliamsDuru2024,
            :elems      => 8,
            :nodes      => 64 .+ 1,
            :order      => 6
        ))
    )
)

runopts = dict_list(Dict(
    :scheme => ["Nordstrom (γ=0)", "Nordstrom (γ>0)", "Lax-Fried. (γ>0)", "Van Leer-Hänel"]
))

# driver {{{2
for testopt in testopts
    testtag = make_tag(testopt)
    testdir = joinpath("comp-euler-2D", testname, testtag)

    println("Testcase:")
    display_settings(testopt)

    @threads for runidx in eachindex(runopts)
        opts   = merge(runopts[runidx], testopt)
        rundir = joinpath("comp-euler-2D", testname, make_tag(opts))

        solve_scheme(datadir(rundir), opts, testparams; runindex = runidx)
    end

    names        = [make_label(merge(runopt, testopt)) for runopt in runopts]
    times        = Vector{Vector{Float64}}(undef, length(names))
    conserved_L1 = Vector{Matrix{Float64}}(undef, length(names))
    conserved    = Vector{Matrix{Float64}}(undef, length(names))
    entropies    = Vector{Matrix{Float64}}(undef, length(names))

    @threads for runidx in eachindex(runopts)
        opts   = merge(runopts[runidx], testopt)
        rundir = joinpath("comp-euler-2D", testname, make_tag(opts))

        ts, cons_L1, cons, entr = gather_quantities(rundir, opts)

        times[runidx]        = ts
        conserved_L1[runidx] = cons_L1
        conserved[runidx]    = cons
        entropies[runidx]    = entr
    end

    println("Plotting evolution of global variables to ", plotsdir(testdir))
    fig = Figure()
    for (qty_i, cons_qty) in enumerate(cons_qtys_metadata)
        @unpack type, qtyname = cons_qty
        println("plotting total $qtyname")

        ax = Axis(fig[1, qty_i];
            xlabel = L"t",
            title = qtyname,
            yminorgridvisible = true,
            yscale = pseudolog_tol(1e-8),
            ytickformat = pows10_ytickformatter,
            yticks = [0.0; 10.0 .^ (-7:2:-1)],
            yminorticks = IntervalsBetween(4),
            limits = (nothing, (-1e-7, 1e-1)),
            width = 85, height = 100)
        if qty_i ≠ 1
            hideydecorations!(ax;
                minorgrid = false, grid = false, ticks = false, minorticks = false)
        end

        let ts = range(tspan..., 100)
            band!(ax, ts, zero(ts), 1e-6ts; alpha = 0.2, color = :black)
        end

        for i in eachindex(times)
            label = names[i]
            ts = times[i]
            qty = view(conserved[i], :, qty_i)
            qty_L1 = view(conserved_L1[i], :, qty_i)
            ys = type == :rel_diff ? (qty .- qty[begin]) ./ qty_L1[begin] :
                 type == :abs_diff ? (qty .- qty[begin]) ./ volΩ :
                 qty
            lines!(ax, ts, abs.(ys); label, color = label_colors[label])
        end
        if qty_i == 1
            save_legend(plotsdir(testdir, "legend-cons-horizontal.pdf"), ax, :horizontal)
            save_legend(plotsdir(testdir, "legend-cons-vertical.pdf"), ax, :vertical)
        end
    end
    resize_to_layout!(fig)
    wsave(plotsdir(testdir, "cons-qtys.pdf"), fig)

    for (qty_i, entr_qty) in enumerate(entr_qtys_metadata)
        @unpack qtyname, ylimits = entr_qty
        println("plotting total $qtyname")

        fig = Figure()
        ax = Axis(fig[1, 1];
            xlabel = L"t",
            title = qtyname,
            width = 120, height = 100)

        for i in eachindex(times)
            label = names[i]
            ts = times[i]
            qty = view(entropies[i], :, qty_i)
            lines!(ax, ts, qty; label, color = label_colors[label])
        end
        resize_to_layout!(fig)
        wsave(plotsdir(testdir, "$qtyname.pdf"), fig)
    end
end

# compare conservation error under timestep refinement {{{1
# options {{{2
tspan = (0.0, 10.0)
testparams = KelvinHelmholtz(; A = 0.6, ϱₗₒ = 0.5)
testname = savename("kelvin-helmholtz-conservation", testparams)

testopts = merge.(
    Ref(Dict(
        :scheme => "Nordstrom (γ>0)"
    )),
    vcat(
        dict_list(Dict(
            :deriv_type => Mattsson2017,
            :elems      => 8,
            :nodes      => 64 .+ 1,
            :order      => 6
        )),
        dict_list(Dict(
            :deriv_type => GlaubitzEtal2024(-0.1),
            :elems      => 64,
            :nodes      => Derived(:order, p -> p + 1),
            :order      => 6
        ))
    )
)

runopts = dict_list(Dict(
    :timestepper => [
        (; solver = SSPRK43(), args = (; abstol = 1e-7, reltol = 1e-6)),
        (; solver = SSPRK43(), args = (; abstol = 1e-8, reltol = 1e-7)),
        (; solver = SSPRK43(), args = (; abstol = 1e-9, reltol = 1e-8)),
        (; solver = SSPRK43(), args = (; abstol = 1e-10, reltol = 1e-9)),
    ]
))

# driver {{{2
for testopt in testopts
    testtag = make_tag(testopt)
    testdir = joinpath("comp-euler-2D", testname, testtag)

    println("Testcase:")
    display_settings(testopt)

    @threads for runidx in eachindex(runopts)
        opts   = merge(runopts[runidx], testopt)
        rundir = joinpath("comp-euler-2D", testname, make_tag(opts))

        solve_scheme(datadir(rundir), opts, testparams; runindex = runidx)
    end

    names        = [latexstring(@sprintf "\\text{tol}_\\text{rel} = 10^{%d}" log10(runopt[:timestepper].args.reltol)) for runopt in runopts]
    times        = Vector{Vector{Float64}}(undef, length(names))
    conserved_L1 = Vector{Matrix{Float64}}(undef, length(names))
    conserved    = Vector{Matrix{Float64}}(undef, length(names))

    @threads for runidx in eachindex(runopts)
        opts   = merge(runopts[runidx], testopt)
        rundir = joinpath("comp-euler-2D", testname, make_tag(opts))

        ts, cons_L1, cons, entr = gather_quantities(rundir, opts)

        times[runidx]        = ts
        conserved_L1[runidx] = cons_L1
        conserved[runidx]    = cons
    end

    update_theme!(;
        ScatterLines = (; cycle = Cycle([:marker, :color, :linestyle], covary=true))
    )
    println("Plotting convergence of global variables to ", plotsdir(testdir))
    let
        fig = Figure()
        ax = Axis(fig[1, 1];
            xlabel = "rel. tolerance",
            ylabel = "rel. change",
            yminorgridvisible = true,
            xminorgridvisible = true,
            xscale = log10,
            yscale = log10,
            yminorticks = IntervalsBetween(4),
            xminorticks = IntervalsBetween(4),
            limits = (nothing, (1e-16, 1e-4)),
            width = 210, height = 130)

        tolerances = map(o -> o[:timestepper].args.reltol, runopts)
        for (qty_i, cons_qty) in enumerate(cons_qtys_metadata)
            @unpack type, qtyname = cons_qty
            println("plotting total $qtyname")

            final_Δqtys = Float64[]
            for i in eachindex(times)
                qty = view(conserved[i], :, qty_i)
                qty_L1 = conserved_L1[i][begin, qty_i]
                Δqty = (qty[end] - qty[begin]) / qty_L1
                push!(final_Δqtys, abs(Δqty))
            end
            scatterlines!(ax, tolerances, final_Δqtys; label=qtyname)
        end
        let
            type = first(cons_qtys_metadata).type
            qty = @view last(conserved)[:, end]
            qty_L1 = last(conserved_L1)[begin, 1]
            Δqty = (qty[end] - qty[begin]) / qty_L1
            tols = tolerances[end-1:end]
            lines!(ax, tols, 30abs(Δqty) / tols[end] * tols; color=:black, label=L"O(\mathrm{tol}_{\mathrm{rel}})")
        end
        resize_to_layout!(fig)
        wsave(plotsdir(testdir, "cons-qtys-final.pdf"), fig)

        save_legend(plotsdir(testdir, "legend-cons-final-horizontal.pdf"), ax, :horizontal)
        save_legend(plotsdir(testdir, "legend-cons-final-vertical.pdf"), ax, :vertical)
    end

    println("Plotting evolution of global variables to ", plotsdir(testdir))
    let
        update_theme!(;
            Lines = (; cycle = Cycle([:linestyle], covary=true), colormap = :inferno)
        )
        fig = Figure()
        for (qty_i, cons_qty) in enumerate(cons_qtys_metadata)
            @unpack type, qtyname = cons_qty
            println("plotting total $qtyname")

            ax = Axis(fig[1, qty_i];
                xlabel = L"t",
                title = qtyname,
                yminorgridvisible = true,
                yscale = pseudolog_tol(1e-10),
                ytickformat = pows10_ytickformatter,
                yticks = [10.0 .^ (-3:-2:-9); 0.0],
                yminorticks = IntervalsBetween(4),
                limits = (nothing, (-1e-9, 1e-3)),
                width = 85, height = 100)
            if qty_i ≠ 1
                hideydecorations!(ax;
                    minorgrid = false, grid = false, ticks = false, minorticks = false)
            end

            for i in eachindex(times)
                label = names[i]
                ts = times[i]
                qty = view(conserved[i], :, qty_i)
                qty_L1 = view(conserved_L1[i], :, qty_i)
                ys = type == :rel_diff ? (qty .- qty[begin]) ./ qty_L1[begin] :
                     type == :abs_diff ? (qty .- qty[begin]) ./ volΩ :
                     qty
                lines!(ax, ts, abs.(ys); label,
                    color = log10(runopts[i][:timestepper].args.reltol),
                    colorrange = (-10, -5)
                )
            end
            if qty_i == 1
                save_legend(plotsdir(testdir, "legend-cons-horizontal.pdf"), ax, :horizontal)
                save_legend(plotsdir(testdir, "legend-cons-vertical.pdf"), ax, :vertical)
            end
        end
        resize_to_layout!(fig)
        wsave(plotsdir(testdir, "cons-qtys.pdf"), fig)
    end
end
