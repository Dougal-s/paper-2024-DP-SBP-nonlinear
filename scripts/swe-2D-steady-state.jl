using DrWatson
@quickactivate :HyperbolicPDEs
using SummationByPartsOperators
using CairoMakie
using OrdinaryDiffEqSSPRK
using LaTeXStrings
using Base.Threads: @threads
using Base.Iterators: product
using UnPack: @unpack
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include(srcdir("PDEs", "swe.jl"))
include(srcdir("plotting-utils.jl"))
include(srcdir("makie-theme.jl"))

const tspan  = (0.0, 1.0)
const domain = ((0.0, 25.0), (0.0, 25.0))
const Δt     = (Δx, Δy) -> 1e-2Δx

const f = 0.0
const g = 9.81

b(x, y) = (x - 10)^2 + (y - 10)^2 < 4 ?
          0.2 - 0.05 * ((x - 10)^2 + (y - 10)^2) :
          0.0

const exact = (
    (t, (x, y)) -> 0.5 - b(x, y), # h
    (t, (x, y)) -> 0.0,           # u
    (t, (x, y)) -> 0.0,           # v
    (t, (x, y)) -> b(x, y)        # b
)

const schemes = [
    # (; label   = "Flux Form",
    #     pdeinfo = ShallowWaterFluxForm2D(g, nothing),
    # ),
    (; label   = "lin. stable DP DG/FD",
        pdeinfo = ShallowWaterFluxForm2D(FluxLaxFriedrichs(); f, g),
    ),
    (; label   = "DGSEM/SBP FD",
        pdeinfo = ShallowWaterSkewSym2D(; f, g),
    ),
    (; label   = "DP DG/FD",
        pdeinfo = ShallowWaterSkewSym2D(FluxEntropyStable(); f, g),
    )
]

const opt_list = vcat(
    dict_list(Dict(
        :order       => Derived(:nodes, Ns -> only(Ns) - 1),
        :nodes       => [[N] for N in 5:7],
        :elems       => [[16, 32, 64]],
        :deriv_type  => GlaubitzEtal2024(-0.1)
    )),
    dict_list(Dict(
        :order       => 4:6 |> collect,
        :nodes       => [[16, 32, 64] .+ 1],
        :elems       => [[4]],
        :deriv_type  => Mattsson2017
    ))
)

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

        mesh = CartesianMesh(domain, (nb, nb))
        cell = local_dp_operator(opts[:deriv_type], opts[:order], (N, N))
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
            progress_name = rpad(label, 25) * lpad(join(size(xs), "×"), 7)
        )

        # compute error
        final_exact = map(f -> f.(tspan[end], xs), exact)
        final_numerical = to_primitive_vars(pdeinfo, sol.u[end])
        errors[scheme_idx][xgrid_idx] = mapreduce(
            (hₙᵤₘ, h) -> abs((h - hₙᵤₘ) / h),
            max,
            final_numerical[1],
            final_exact[1]
        )
    end

    tag = savename(opts) * "-" * string(hash(opts); base = 60)
    tablepath = datadir("swe-2D", "steady-state", "error-$tag.tex")
    mkpath(dirname(tablepath))
    open(tablepath; write = true) do io
        print_error_table_latex(io, schemes, xgrids, errors)
    end
    print_error_table_ascii(schemes, xgrids, errors)
end
