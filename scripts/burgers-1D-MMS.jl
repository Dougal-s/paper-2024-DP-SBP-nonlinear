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

include(srcdir("PDEs", "burgers.jl"))

const savedir = joinpath("burgers-1D", "MMS")

const tspan = (0.0, 2.0)
const xspan = (-1.0, 1.0)
const Δt    = Δx -> 0.01Δx

# max wavespeed = u
const exact = (
    (t, (x,)) -> 2 + 0.3 * sinpi(2 * (x - t)), # u
)

const schemes = [
    # (; label = "Flux Form",
    #     pde = BurgersFluxForm1D(g, nothing),
    #     marker = :cross
    # ),
    (; label = "linearly stable",
        pdeinfo = BurgersFluxForm1D(SourceMMS(exact), FluxLaxFriedrichs())
    ),
    (; label = "entropy conserving",
        pdeinfo = BurgersSkewSym1D(SourceMMS(exact))
    ),
    (; label = "entropy stable",
        pdeinfo = BurgersSkewSym1D(SourceMMS(exact), FluxEntropyStable())
    )
]

const opt_list = vcat(
    dict_list(Dict(
        :order      => [2, 4, 6, 8],
        :nodes      => [[16, 32, 64, 128] .+ 1],
        :elems      => [4],
        :deriv_type => Mattsson2017
    )),
    dict_list(Dict(
        :order      => Derived(:nodes, N -> only(N) - 1),
        :nodes      => [4, 5, 6, 7],
        :elems      => [[8, 16, 32, 64]],
        :deriv_type => GlaubitzEtal2024(-0.1)
    ))
)

for opts in opt_list
    println()
    display_settings(opts)

    @unpack nodes, elems = opts

    xgrids = Iterators.product(elems, nodes) |> collect
    errors = [fill(NaN64, length(xgrids)) for _ in eachindex(schemes)]
    runs   = product(eachindex(schemes), eachindex(xgrids)) |> collect
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
