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

const savedir = joinpath("swe-2D", "MMS")

const tspan = (0.0, 1.0)
const xspan = (-1.0, 1.0)
const Δt    = (Δx, Δy) -> 0.05Δx

const f = 0.0
const g = 1.0

# wavespeed = |u| + √(gh), |v| + √(gh)
const exact = (
    (t, (x, y)) -> 2 + 0.2 * sinpi(2 * (x - t)) * sinpi(2 * (y - t)), # h
    (t, (x, y)) -> 2 + 0.2 * sinpi(2 * (x + t)) * sinpi(2 * (y + t)), # u
    (t, (x, y)) -> 2 + 0.2 * sinpi(2 * (x + t)) * sinpi(2 * (y + t)), # v
    (t, (x, y)) -> 0.0                                                # b
)

const schemes = [
    # (; label = "Flux Form",
    #     pde = ShallowWaterFluxForm1D(SourceMMS(exact); f = f, g = g),
    # ),
    (; label = "linearly stable",
        pdeinfo = ShallowWaterFluxForm2D(SourceMMS(exact), FluxLaxFriedrichs(); f, g)
    ),
    (; label = "entropy conserving",
        pdeinfo = ShallowWaterSkewSym2D(SourceMMS(exact); f, g)
    ),
    (; label = "entropy stable",
        pdeinfo = ShallowWaterSkewSym2D(SourceMMS(exact), FluxEntropyStable(); f, g)
    )
]

const opt_list = dict_list(Dict(
    :order      => [4, 5, 6, 7],
    :nodes      => [[16, 32, 64, 128] .+ 1],
    :elems      => [[2]],
    :deriv_type => Mattsson2017
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

        mesh = CartesianMesh((xspan, xspan), (nb, nb))
        cell = local_dp_operator(opts[:deriv_type], opts[:order], (N, N))
        xs, fdop = couple_operators(mesh, cell)

        prob = let
            s0   = map(f -> f.(tspan[begin], xs), exact)
            pde! = semidiscretise(pdeinfo, xs, fdop)
            ODEProblem{true}(pde!, from_primitive_vars(pdeinfo, s0), tspan)
        end

        # solve
        sol = solve(prob, SSPRK54();
            dt = Δt(step(xs)...),
            save_end = true,
            save_everystep = false,
            progress = true,
            progress_steps = 200,
            progress_id = Symbol(scheme_idx * length(xgrids) + xgrid_idx),
            progress_name = rpad(label, 20) * lpad(join(size(xs), "×"), 7)
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
