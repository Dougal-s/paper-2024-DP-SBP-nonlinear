using DrWatson
using JET: @report_opt
using BenchmarkTools: @benchmark
@quickactivate :HyperbolicPDEs
using SummationByPartsOperators
using Printf

include(srcdir("PDEs", "compressible-euler.jl"))

const γ = 1.4

const domain = (0.0, 1.0)
const state0 = let
    h = (x) -> 1.0 + sinpi(2x + 0 // 3)
    u = (x) -> 1.0 + sinpi(2x + 1 // 3)
    p = (x) -> 1.0 + sinpi(2x + 2 // 3)
    (h, u, p)
end

const schemes = Dict(
    "Flux Form"              => CompEulerFluxForm1D(γ = γ),
    "Lax-Fried."             => CompEulerFluxForm1D(γ = γ, flux_splitting = FluxLaxFriedrichs()),
    "Skew Symm. (γ=0)"       => NordstromCompEuler1D(γ = γ),
    "Skew Symm. (γ>0)"       => NordstromCompEuler1D(γ = γ, flux_splitting = FluxEntropyStable()),
    "Riess-Sesterhenn (γ=0)" => ReissSesterhennCompEuler1D(γ = γ),
)

function run_benchmarks()
    allopts = dict_list(Dict(
        :scheme => keys(schemes) |> collect,
        :deriv_order => 6,
        :deriv_type  => Mattsson2017
    ))

    for opts in allopts
        pdeinfo = schemes[opts[:scheme]]
        display(opts)
        println()

        for gridsize in [128, 256, 512]
            println("grid\033[90m:\033[0m ", gridsize[begin], "×", gridsize[end])

            xs = bounded_range(domain..., gridsize)
            (Dx₋, Dx₊, Hx) = dp_operator(
                opts[:deriv_type],
                PeriodicSAT(),
                opts[:deriv_order],
                xs
            )
            s₀ = meshgrid.(state0, Ref(xs))
            pde! = semidiscretise(pdeinfo, xs, (Dx₊, Dx₋))

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
        :gridsize    => [128],
        :deriv_order => 6,
        :deriv_type  => Mattsson2017
    ))
    for opts in allopts
        pdeinfo = schemes[opts[:scheme]]

        xs = bounded_range(domain..., opts[:gridsize])
        (Dx₋, Dx₊, Hx) = dp_operator(
            opts[:deriv_type],
            PeriodicSAT(),
            opts[:deriv_order],
            xs
        )
        s₀ = meshgrid.(state0, Ref(xs))
        pde! = semidiscretise(pdeinfo, xs, (Dx₊, Dx₋))

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
