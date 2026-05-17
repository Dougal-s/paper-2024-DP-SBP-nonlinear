using DrWatson
using JET: @report_opt
using BenchmarkTools: @benchmark
@quickactivate :HyperbolicPDEs
using SummationByPartsOperators
using Printf

include(srcdir("PDEs", "swe.jl"))

const f = 7.292e-5
const g = 9.80616

const domain = ((0.0, 1.0), (0.0, 1.0))
const state0 = let
    h = (x, y) -> 1.0 + sinpi(2x) * sinpi(2y)
    u = (x, y) -> 1.0 + cospi(2x) * sinpi(2y)
    v = (x, y) -> 1.0 + sinpi(2x) * cospi(2y)
    b = (x, y) -> zero(x)
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

io = IOContext(stdout, :histmin => 0.1e6, :histmax => 10e6, :logbins => true)

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

printstyled("Benchmarks\n"; bold = true)
run_benchmarks()

printstyled("Static Analysis\n"; bold = true)
run_static_analysis()
