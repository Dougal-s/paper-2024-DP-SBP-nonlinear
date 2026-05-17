using DrWatson
@quickactivate :HyperbolicPDEs
using SummationByPartsOperators
using OrdinaryDiffEqSSPRK
using LinearAlgebra
using Base.Threads: @threads
using Base.Iterators: product
using UnPack: @unpack
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include(srcdir("PDEs", "swe.jl"))

const savedir = joinpath("swe-1D", "MMS")

const tspan = (0.0, 2.0)
const xspan = (-1.0, 1.0)
const Δt    = Δx -> 0.01Δx

const g = 1.0

# wavespeed = |u| + √(gh)
const exact = (
    (t, (x,)) -> 2 + 0.3 * sinpi(2 * (x - t)), # h
    (t, (x,)) -> 0.3 * sinpi(2 * (x + t)),     # u
    (t, (x,)) -> 0.0                           # b
)

const schemes = [
    # (; label = "Flux Form",
    #     pde = ShallowWaterFluxForm1D(SourceMMS(exact); g = g),
    # ),
    (; label = "linearly stable",
        pdeinfo = ShallowWaterFluxForm1D(SourceMMS(exact), FluxLaxFriedrichs(); g)
    ),
    (; label = "entropy conserving",
        pdeinfo = ShallowWaterSkewSym1D(SourceMMS(exact); g)
    ),
    (; label = "entropy stable",
        pdeinfo = ShallowWaterSkewSym1D(SourceMMS(exact), FluxEntropyStable(); g)
    )
]

const opt_list = dict_list(Dict(
    :order       => [2, 4, 6, 8],
    :nodes       => [[16, 32, 64, 128] .+ 1],
    :elems       => [[4]],
    :deriv_type  => Mattsson2017
))

for opts in opt_list
    println()
    display_settings(opts)

    @unpack nodes, elems = opts

    xgrids = Iterators.product(elems, nodes) |> collect
    errors = [fill(NaN64, length(xgrids)) for _ in eachindex(schemes)]
    runs = product(eachindex(schemes), eachindex(xgrids)) |> collect
    @threads for (scheme_idx, xgrid_idx) in runs
        # initialize
        @unpack pdeinfo, label = schemes[scheme_idx]
        nb, N = xgrids[xgrid_idx]

        mesh = CartesianMesh((xspan,), (nb,))
        cell = local_dp_operator(opts[:deriv_type], opts[:order], (N,))
        xs, fdop = couple_operators(mesh, cell)

        prob = let
            s0   = map(f -> f.(tspan[begin], xs), exact)
            pde! = semidiscretise(pdeinfo, xs, fdop)
            ODEProblem{true, SciMLBase.NoSpecialize}(
                pde!, from_primitive_vars(pdeinfo, s0), tspan)
        end

        # solve
        sol = solve(prob, SSPRK54();
            dt = Δt(step(xs)...),
            save_end = true,
            save_everystep = false,
            progress = true,
            progress_steps = 250,
            progress_id = Symbol(scheme_idx * length(xgrids) + xgrid_idx),
            progress_name = rpad(label, 20) * lpad(join(size(xs), "×"), 3)
        )

        # compute error
        final_exact     = map(f -> f.(tspan[end], xs), exact)
        final_numerical = to_primitive_vars(pdeinfo, sol.u[end])
        final_err       = final_exact .- final_numerical

        ∫_Ω = VolumeMeasure(xs, fdop)
        errors[scheme_idx][xgrid_idx] = √∫_Ω(δ -> δ ⋅ δ, final_err)
    end

    tag = savename(opts) * "-" * string(hash(opts); base = 60)
    tablepath = datadir(savedir, "convergence-$tag.tex")
    mkpath(dirname(tablepath))
    open(tablepath; write = true) do io
        print_convergence_table_latex(io, schemes, xgrids, errors)
    end
    print_convergence_table_ascii(schemes, xgrids, errors)
end
