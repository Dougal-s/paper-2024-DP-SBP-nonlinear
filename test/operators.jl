using LinearAlgebra
using SummationByPartsOperators

@testset "Operators" verbose=true begin
    @testset "NullOperator" begin
        op = NullOperator()

        B = rand(24)
        C = rand(24)
        C_original = copy(C)
        α = 0.7
        β = 0.8
        mul!(C, op, B, α, β)

        @test B - 5 * op / 3 == B
        @test op * B * α + C_original * β == C
    end

    @testset "PeriodicMultiblockOperator" begin
        K = 20
        N = 21
        local_op = upwind_operators(Mattsson2017;
            derivative_order=1,
            accuracy_order=3,
            xmin=0.0, xmax=1.0, N)
        Dm = HyperbolicPDEs.PeriodicMultiblockOperator(local_op.minus,
            1/left_boundary_weight(local_op), 0.0,
            K, N)
        Dp = HyperbolicPDEs.PeriodicMultiblockOperator(local_op.plus,
            0.0, 1/right_boundary_weight(local_op),
            K, N)
        D = (Dm + Dp) / 2
        H = kron(Diagonal(I, K), mass_matrix(local_op))

        B = rand(K * N)
        C = rand(K * N)
        C_original = copy(C)
        α = 0.7
        β = 0.8
        mul!(C, D, B, α, β)

        @test norm((H * Matrix(Dp)) + (H * Matrix(Dm))') ≤ 1e-9
        @test norm((H * Matrix(D)) + (H * Matrix(D))') ≤ 1e-9
        @test D * B * α + C_original * β ≈ C
        @test norm(Dm * ones(K * N)) ≤ 1e-9
        @test norm(Dp * ones(K * N)) ≤ 1e-9
    end

    @testset "Dissipation Operator" begin
        domain = ((-1.0, 1.0),)
        K = 20
        @testset "Mattsson2017" begin
            N = 21
            mesh = CartesianMesh(domain, (K,))
            cell = local_dp_operator(Mattsson2017, 3, (N,))
            xs, fdop = couple_operators(mesh, cell)

            @test Matrix(fdop[1].Diᵥ) ≈ Matrix((fdop[1].D₊ - fdop[1].D₋) / 2)
            @test Matrix(fdop[1].Diₛ) ≈ Matrix((fdop[1].B₊ - fdop[1].B₋) / 2)
        end
        @testset "GlaubitzEtal2024" begin
            N = 5
            mesh = CartesianMesh(domain, (K,))
            cell = local_dp_operator(GlaubitzEtal2024(-0.1), N-1, (N,))
            xs, fdop = couple_operators(mesh, cell)

            @test Matrix(fdop[1].Diᵥ) ≈ Matrix((fdop[1].D₊ - fdop[1].D₋) / 2)
            @test Matrix(fdop[1].Diₛ) ≈ Matrix((fdop[1].B₊ - fdop[1].B₋) / 2)
        end
    end

    @testset "Measures" begin
        domain = ((0.0, 1.0), (0.0, 2.0))
        K = 10
        @testset "GlaubitzEtal2024" begin
            N = 4
            mesh = CartesianMesh(domain, (K,K))
            cell = local_dp_operator(GlaubitzEtal2024(-0.1), N-1, (N,N))
            xs, fdop = couple_operators(mesh, cell)
            ∫_Ω = VolumeMeasure(xs, fdop)

            f = first.(xs)
            X = (first.(xs), last.(xs))

            @test ∫_Ω(_ -> 1.0, (first.(xs),last.(xs))) ≈ 2
            for px in 1:(2N-1), py in 1:(2N-1)
                p = (px, py)
                exact = 2 / (px+1) + 2^(py+1) / (py+1)
                @test ∫_Ω(x -> sum(x.^p), X) ≈ exact
            end

            @inferred ∫_Ω(f)
            @inferred ∫_Ω(x -> sum(abs2, x), X)
        end
    end
end
