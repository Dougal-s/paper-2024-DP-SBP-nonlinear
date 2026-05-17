using DrWatson
using JET: @report_opt
using BenchmarkTools: @benchmark
@quickactivate :HyperbolicPDEs
using SummationByPartsOperators
using Printf
using UnPack
using LinearAlgebra

const γ = 1.4

const domain = ((0.0, 1.0), (0.0, 1.0))

io = IOContext(stdout, :histmin => 0.05e6, :histmax => 1e6, :logbins => true)

function run_benchmarks()
    dofs = 3 * 128
    allopts = mapreduce(dict_list, vcat,
        [
            Dict(
                :order      => [3, 5],
                :deriv_type => GlaubitzEtal2024(-0.1),
                :nodes      => Derived(:order, p -> p + 1),
                :elems      => Derived(:nodes, N -> dofs ÷ N)
            ),
            Dict(
                :order      => 6,
                :deriv_type => Mattsson2017,
                :nodes      => Derived(:elems, K -> dofs ÷ K),
                :elems      => [4, 2, 1]
            )
        ])

    for opts in allopts
        display(opts)
        println()

        @unpack order, elems, nodes = opts

        mesh     = CartesianMesh(domain, (elems, elems))
        cell     = local_dp_operator(opts[:deriv_type], order, (nodes, nodes))
        xs, fdop = couple_operators(mesh, cell)
        ∫_Ω      = VolumeMeasure(xs, fdop)

        Dx = fdop[1].D + fdop[1].B
        Dy = fdop[2].D + fdop[2].B

        s     = xs .|> ((x, y),) -> rand()
        ds    = similar(s)
        cache = similar(s)

        println("∂/∂x")
        show(io, MIME("text/plain"),
            @benchmark axis_mul!($ds, Val(1), $Dx, $s, $cache))
        println()
        println("∂/∂y")
        show(io, MIME("text/plain"),
            @benchmark axis_mul!($ds, Val(2), $Dy, $s, $cache))
        println()
        println("∫")
        show(io, MIME("text/plain"),
            @benchmark $∫_Ω(x -> x ⋅ x, ($s,)))
        println()

        println()
    end
end

function run_static_analysis()
    allopts = [
        Dict(
            :order      => 4,
            :deriv_type => GlaubitzEtal2024(-0.1),
            :nodes      => 5,
            :elems      => 96
        ),
        Dict(
            :order      => 6,
            :deriv_type => Mattsson2017,
            :nodes      => 96 + 1,
            :elems      => 4
        )
    ]
    for opts in allopts
        @unpack order, elems, nodes = opts
        mesh = CartesianMesh(domain, (elems,elems))
        cell = local_dp_operator(opts[:deriv_type], order, (nodes,nodes))
        xs, fdop = couple_operators(mesh, cell)
        ∫_Ω = VolumeMeasure(xs, fdop)

        Dx = fdop[1].D + fdop[1].B
        Dy = fdop[2].D + fdop[2].B

        s     = xs .|> ((x, y),) -> rand()
        ds    = similar(s)
        cache = similar(s)

        display(opts)
        display(@report_opt axis_mul!(ds, Val(1), Dx, s, cache))
        display(@report_opt axis_mul!(ds, Val(2), Dy, s, cache))
        display(@report_opt ∫_Ω(only, (s,)))
        println()
    end
end

println("\033[1mBenchmarks\033[0m")
run_benchmarks()

println("\033[1mStatic Analysis\033[0m")
run_static_analysis()
