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
    "Flux Form"              => CompEulerFluxForm2D(γ = γ),
    "Lax-Fried."             => CompEulerFluxForm2D(γ = γ, flux_splitting = FluxLaxFriedrichs()),
    "Skew Symm. (γ=0)"       => NordstromCompEuler2D(γ = γ),
    "Skew Symm. (γ>0)"       => NordstromCompEuler2D(γ = γ, flux_splitting = FluxEntropyStable()),
    "Riess-Sesterhenn (γ=0)" => ReissSesterhennCompEuler2D(γ = γ),
)

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

        for gridsize in [(128, 128), (256, 256), (512, 512)]
            println("grid\033[90m:\033[0m ", gridsize[begin], "×", gridsize[end])

            xs = map((dom, n) -> bounded_range(dom..., n), domain, gridsize)
            (Dx₋, Dx₊, Hx), (Dy₋, Dy₊, Hy) = dp_operator.(
                Ref(opts[:deriv_type]),
                Ref(PeriodicSAT()),
                Ref(opts[:deriv_order]),
                xs
            )
            s₀ = meshgrid.(state0, Ref.(xs)...)
            pde! = semidiscretise(pdeinfo, s₀, Dx₊, Dx₋, Dy₊, Dy₋)

            s = from_primitive_vars(pdeinfo, s₀)
            ds = similar(s)

            pde!(ds, s, (), 0.0)
            display(@benchmark $pde!($ds, $s, (), 0.0))
            println()
        end
    end
end

function run_static_analysis()
    allopts = dict_list(Dict(
        :scheme      => keys(schemes) |> collect,
        :gridsize    => [(128, 128)],
        :deriv_order => 6,
        :deriv_type  => Mattsson2017
    ))
    for opts in allopts
        pdeinfo = schemes[opts[:scheme]]

        xs = map((dom, n) -> bounded_range(dom..., n), domain, opts[:gridsize])
        (Dx₋, Dx₊, Hx), (Dy₋, Dy₊, Hy) = dp_operator.(
            Ref(opts[:deriv_type]),
            Ref(PeriodicSAT()),
            Ref(opts[:deriv_order]),
            xs
        )
        s₀ = meshgrid.(state0, Ref.(xs)...)
        pde! = semidiscretise(pdeinfo, s₀, Dx₊, Dx₋, Dy₊, Dy₋)

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
