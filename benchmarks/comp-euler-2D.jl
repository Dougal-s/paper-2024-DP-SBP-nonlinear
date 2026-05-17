using DrWatson
using JET: @report_opt
using BenchmarkTools: @benchmark
@quickactivate :HyperbolicPDEs
using SummationByPartsOperators
using Printf

include(srcdir("PDEs", "compressible-euler.jl"))

const γ = 1.4

const domain = ((0.0, 1.0), (0.0, 1.0))
const state0 = let
    h = (x, y) -> 1.0 + sinpi(2x + 0 // 4) * sinpi(2y + 0 // 4)
    u = (x, y) -> 1.0 + sinpi(2x + 1 // 4) * sinpi(2y + 1 // 4)
    v = (x, y) -> 1.0 + sinpi(2x + 2 // 4) * sinpi(2y + 2 // 4)
    p = (x, y) -> 1.0 + sinpi(2x + 3 // 4) * sinpi(2y + 3 // 4)
    (h, u, v, p)
end

const schemes = Dict(
    "Flux Form"        => CompEulerFluxForm2D(; γ),
    "Lax-Fried."       => CompEulerFluxForm2D(FluxLaxFriedrichs(); γ),
    "van Leer-Hänel"   => CompEulerFluxForm2D(FluxVanLeerHanel(); γ),
    "Skew Symm. (γ=0)" => NordstromCompEuler2D(; γ),
    "Skew Symm. (γ>0)" => NordstromCompEuler2D(FluxDSL2025(); γ),
    "Reiss-Sesterhenn" => ReissSesterhennCompEuler2D(; γ)
)

io = IOContext(stdout, :histmin => 0.5e6, :histmax => 10e6, :logbins => true)

function run_benchmarks()
    allopts = dict_list(Dict(
        :scheme      => keys(schemes) |> collect,
        :deriv_order => 6,
        :deriv_type  => Mattsson2017
    ))

    for opts in allopts
        pdeinfo = schemes[opts[:scheme]]
        display(opts)
        println()

        for (K, N) in [(4, 65), (2, 129), (1, 257)]
            println("grid\033[90m:\033[0m $(K)×$(K) elements, $(N)×$(N) nodes per element")

            mesh     = CartesianMesh(domain, (K, K))
            cell     = local_dp_operator(opts[:deriv_type], opts[:deriv_order], (N, N))
            xs, fdop = couple_operators(mesh, cell)
            pde!     = semidiscretise(pdeinfo, xs, fdop)

            s₀ = map(f -> splat(f).(xs), state0)
            s  = from_primitive_vars(pdeinfo, s₀)
            ds = similar(s)

            pde!(ds, s, (), 0.0)
            show(io, MIME("text/plain"), @benchmark $pde!($ds, $s, (), 0.0))
            println()
        end
    end
end

function run_static_analysis()
    allopts = dict_list(Dict(
        :scheme      => keys(schemes) |> collect,
        :deriv_order => 6,
        :deriv_type  => Mattsson2017
    ))
    for opts in allopts
        pdeinfo = schemes[opts[:scheme]]

        mesh     = CartesianMesh(domain, (1, 1))
        cell     = local_dp_operator(opts[:deriv_type], opts[:deriv_order], (128, 128))
        xs, fdop = couple_operators(mesh, cell)

        pde! = semidiscretise(pdeinfo, xs, fdop)

        s₀ = map(f -> splat(f).(xs), state0)
        s = from_primitive_vars(pdeinfo, s₀)
        ds = similar(s)

        display(opts)
        pde!(ds, s, (), 0.0)
        display(@report_opt pde!(ds, s, (), 0.0))
    end
end

println("\033[1mBenchmarks\033[0m")
run_benchmarks()

println("\033[1mStatic Analysis\033[0m")
run_static_analysis()
