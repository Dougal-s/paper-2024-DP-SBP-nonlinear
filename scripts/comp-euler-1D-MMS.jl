using DrWatson
@quickactivate :HyperbolicPDEs
using SummationByPartsOperators
using OrdinaryDiffEqSSPRK
using Base.Threads: @threads
using Base.Iterators: product
using UnPack: @unpack
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include(srcdir("PDEs", "compressible-euler.jl"))

const savedir = joinpath("comp-euler-1D", "MMS")

const tspan = (0.0, 2.0)
const xspan = (-1.0, 1.0)
const Δt    = Δx -> 0.05Δx

const γ = 1.4

const exact = (
    (t, (x,)) -> 2.0 + 0.3 * sinpi(2 * (x - t)), # ϱ
    (t, (x,)) -> 1.0,                            # u
    (t, (x,)) -> 2.0 + 0.3 * sinpi(2 * (x + t))  # p
)

const schemes = [
    # (; label = "Flux Form",
    #     pdeinfo = CompEulerFluxForm1D(SourceMMS(exact); γ)
    # ),
    (; label = "linearly stable DP DG/FD",
        pdeinfo = CompEulerFluxForm1D(SourceMMS(exact), FluxLaxFriedrichs(); γ)
    ),
    (; label = "DGSEM/SBP FD",
        pdeinfo = NordstromCompEuler1D(SourceMMS(exact); γ)
    ),
    (; label = "DP DG/FD",
        pdeinfo = NordstromCompEuler1D(SourceMMS(exact), FluxDSL2025(); γ)
    )
]

const opt_list = mapreduce(dict_list, vcat,
    [
        Dict(
            :order      => [2, 4, 6, 8],
            :nodes      => [[16, 32, 64, 128] .+ 1],
            :elems      => [[4]],
            :deriv_type => Mattsson2017
        ),
        Dict(
            :order      => Derived(:nodes, N -> only(N) - 1),
            :nodes      => [4, 5, 6, 7],
            :elems      => [[8, 16, 32, 64]],
            :deriv_type => GlaubitzEtal2024(-0.1)
        )
    ])

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
