# DP SBP operators on LGL nodes from:
#     Generalized upwind summation-by-parts operators and their application to
#       nodal discontinuous Galerkin methods
#     by Jan Glaubitz , Hendrik Ranocha, Andrew R. Winters, Michael
#       Schlottke-Lakemper, Philipp Offner, and Gregor Gassner
#     https://arxiv.org/pdf/2406.14557
#     https://github.com/trixi-framework/paper-2024-generalized-upwind-sbp/

using LinearAlgebra
using StaticArrays
using SummationByPartsOperators
using FastGaussQuadrature: gausslobatto

export GlaubitzEtal2024

@kwdef struct GlaubitzEtal2024
    σ::Float64 = -0.1
end

function local_dp_operator(
        T::Type,
        coefficient_source::GlaubitzEtal2024,
        accuracy_order::Int,
        Ns::NTuple{N, Int}
) where {N}
    return ntuple(Val(N)) do i
        make_dp_sbp_LGL_operators(Ns[i], accuracy_order, coefficient_source.σ)
    end
end

"""
    compute_dop_vandermonde_matrix(nodes::Vector{<:Real})::Matrix

Construct the Vandermonde matrix for the discrete orthonormal polynomial basis
on `nodes`.
"""
function dop_vandermonde(nodes::Vector{<:Real})::Matrix
    dofs = length(nodes)
    V = [x^p for x in nodes, p in 0:(dofs - 1)]
    for i in axes(V, 2)
        for j in 1:(i - 1)
            V[:, i] -= (V[:, i] ⋅ V[:, j]) * V[:, j]
        end
        normalize!(view(V, :, i))
    end
    return V
end

function make_dp_sbp_LGL_operators(n::Int, p::Int, λₙ::Float64)
    xs, hs = gausslobatto(n)
    # remap from [-1,1] to [0,1]
    @. xs = (xs + 1) / 2
    @. hs /= 2

    V = dop_vandermonde(xs)
    Λ = Diagonal([zeros(n-1); λₙ])
    S = V * Λ * V'

    H⁻¹ = Diagonal(inv.(hs))
    op = legendre_derivative_operator(0.0, 1.0, n)
    D  = SMatrix{n,n}(Matrix(op))
    D₋ = SMatrix{n,n}(D - H⁻¹ * S / 2)
    D₊ = SMatrix{n,n}(D + H⁻¹ * S / 2)

    source = GlaubitzEtal2024(λₙ)
    return UpwindOperators(
        MatrixDerivativeOperator(0.0, 1.0, xs, hs, D₋, n - 1, source),
        MatrixDerivativeOperator(0.0, 1.0, xs, hs, D, n - 1, source),
        MatrixDerivativeOperator(0.0, 1.0, xs, hs, D₊, n - 1, source)
    )
end

function interpolate_onto(::GlaubitzEtal2024, new_xs, xs, us)
    x_m = (first(xs) + last(xs)) / 2
    x_r = last(xs) - x_m
    V = [((x - x_m) / x_r)^p for x in xs, p in 0:(length(xs) - 1)]
    coefs = V \ us
    return evalpoly.((new_xs .- x_m) ./ x_r, Ref(coefs))
end
