module SbpOperators

using SummationByPartsOperators
using LinearAlgebra
import Base: +, -, *, /, eltype
import LinearAlgebra: mul!, Diagonal

export dp_operator, make_periodic, PeriodicUpwindSAT, PeriodicSAT, PeriodicDomain

"""
Represents a bounded SBP finite difference operator with periodic boundaries
added with SAT.

Equivalent the the expression `op + σ₁e₁e₁ᵀ - σₙeₙeₙᵀ`.
"""
struct PeriodicSATOperator{Op, T}
    op::Op
    σ₁::T
    σₙ::T
end

Base.eltype(op::PeriodicSATOperator) = eltype(op.op)

function Base.:+(lhs::PeriodicSATOperator, rhs::PeriodicSATOperator)
    PeriodicSATOperator(lhs.op + rhs.op, lhs.σ₁ + rhs.σ₁, lhs.σₙ + rhs.σₙ)
end

function Base.:-(lhs::PeriodicSATOperator, rhs::PeriodicSATOperator)
    PeriodicSATOperator(lhs.op - rhs.op, lhs.σ₁ - rhs.σ₁, lhs.σₙ - rhs.σₙ)
end

function Base.:*(s::Real, op::PeriodicSATOperator)
    PeriodicSATOperator(s * op.op, s * op.σ₁, s * op.σₙ)
end

function Base.:*(op::PeriodicSATOperator, s::Real)
    PeriodicSATOperator(s * op.op, s * op.σ₁, s * op.σₙ)
end

function Base.:/(op::PeriodicSATOperator, s::Real)
    PeriodicSATOperator(op.op / s, op.σ₁ / s, op.σₙ / s)
end

function *(op::PeriodicSATOperator, src::AbstractArray)
    result_type = promote_type(eltype(op), eltype(src))
    dst = similar(src, result_type)
    mul!(dst, op, src)
end

function mul!(dst, op::PeriodicSATOperator, src)
    mul!(dst, op.op, src)
    diff = src[begin] - src[end]
    dst[begin] += op.σ₁ * diff
    dst[end] += op.σₙ * diff
    return dst
end

"""
Impose periodic boundary conditions using an upwind SBP Simultaneous
Approximation Term (SAT)
"""
struct PeriodicUpwindSAT end

"""
Impose periodic boundary conditions using a SBP Simultaneous Approximation Term
(SAT)
"""
struct PeriodicSAT end

"""
Impose periodic boundary conditions by making the domain periodic
"""
struct PeriodicDomain end

make_periodic(::PeriodicSAT, H::Diagonal) = H
make_periodic(::PeriodicUpwindSAT, H::Diagonal) = H

function make_periodic(::PeriodicUpwindSAT, op::DerivativeOperator, upwind)
    H = mass_matrix(op)
    return if upwind == :+
        PeriodicSATOperator(op, zero(eltype(H)), 1 / H[end, end])
    else
        PeriodicSATOperator(op, 1 / H[begin, begin], zero(eltype(H)))
    end
end

function make_periodic(::PeriodicSAT, op::DerivativeOperator, _)
    H = mass_matrix(op)
    return PeriodicSATOperator(op, 1 // 2 / H[begin, begin], 1 // 2 / H[end, end])
end

make_periodic(::PeriodicDomain, H::Diagonal) = Diagonal(I, size(H, 1))

function make_periodic(::PeriodicDomain, op::DerivativeOperator, _)
    coefs = op.coefficients
    periodic_coefs = SummationByPartsOperators.PeriodicDerivativeCoefficients(
        coefs.lower_coef, coefs.central_coef, coefs.upper_coef,
        coefs.mode,
        coefs.derivative_order,
        coefs.accuracy_order,
        coefs.source_of_coefficients
    )
    xs = grid(op)
    return PeriodicDerivativeOperator(
        periodic_coefs,
        range(first(xs), last(xs) + step(xs), 1 + length(xs))
    )
end

function dp_operator(
        coefficient_source,
        boundary_type,
        accuracy_order::Int,
        xs::AbstractRange
)
    op = upwind_operators(coefficient_source;
        accuracy_order = accuracy_order,
        xmin = first(xs),
        xmax = last(xs),
        N = length(xs)
    )
    return (
        make_periodic(boundary_type, op.minus, :-),
        make_periodic(boundary_type, op.plus, :+),
        make_periodic(boundary_type, mass_matrix(op)),
    )
end

end
