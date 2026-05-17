using LinearAlgebra

export NullOperator

struct NullOperator end

Base.eltype(::NullOperator) = Bool

Base.:+(op, ::NullOperator) = op
Base.:+(::NullOperator, op) = op
Base.:+(::NullOperator, ::NullOperator) = NullOperator()

Base.:-(op, ::NullOperator) = op
Base.:-(::NullOperator, op) = -op
Base.:-(::NullOperator, ::NullOperator) = NullOperator()

Base.:*(::Real, op::NullOperator) = op
Base.:*(op::NullOperator, ::Real) = op
Base.:*(::NullOperator, u::AbstractArray) = zero(u)

Base.:/(op::NullOperator, ::Real) = op

LinearAlgebra.mul!(Y::AbstractVector{T}, ::NullOperator, ::AbstractVector) where {T} = fill!(Y, zero(T))
LinearAlgebra.mul!(C::AbstractVector, ::NullOperator, ::AbstractVector, _, β) = rmul!(C, β)

"""
Represents a bounded SBP finite difference operator with periodic boundaries
added with SAT.

Equivalent the the expression `op + σ₁e₁e₁ᵀ - σₙeₙeₙᵀ`.
"""
struct PeriodicMultiblockOperator{Op, T <: Real}
    op::Op
    σ₁::T
    σₙ::T
    numblocks::Int64
    blocklen::Int64
end

function LinearAlgebra.Matrix(op::PeriodicMultiblockOperator{Op, T}) where {Op, T}
    N = op.blocklen * op.numblocks
    return stack(i -> op * .==(i, 1:N), 1:N)
end

function Base.:+(lhs::PeriodicMultiblockOperator, rhs::PeriodicMultiblockOperator)
    @assert lhs.numblocks == rhs.numblocks
    @assert lhs.blocklen == rhs.blocklen
    PeriodicMultiblockOperator(lhs.op + rhs.op, lhs.σ₁ + rhs.σ₁, lhs.σₙ + rhs.σₙ,
        lhs.numblocks,
        lhs.blocklen
    )
end

function Base.:-(lhs::PeriodicMultiblockOperator, rhs::PeriodicMultiblockOperator)
    @assert lhs.numblocks == rhs.numblocks
    @assert lhs.blocklen == rhs.blocklen
    PeriodicMultiblockOperator(lhs.op - rhs.op, lhs.σ₁ - rhs.σ₁, lhs.σₙ - rhs.σₙ,
        lhs.numblocks,
        lhs.blocklen
    )
end

function Base.:*(s::Real, op::PeriodicMultiblockOperator)
    PeriodicMultiblockOperator(s * op.op, s * op.σ₁, s * op.σₙ,
        op.numblocks,
        op.blocklen
    )
end

function Base.:*(op::PeriodicMultiblockOperator, s::Real)
    PeriodicMultiblockOperator(s * op.op, s * op.σ₁, s * op.σₙ,
        op.numblocks,
        op.blocklen
    )
end

function Base.:/(op::PeriodicMultiblockOperator, s::Real)
    PeriodicMultiblockOperator(op.op / s, op.σ₁ / s, op.σₙ / s,
        op.numblocks,
        op.blocklen
    )
end

function Base.:*(op::PeriodicMultiblockOperator{Op, T}, src::AbstractArray) where {Op, T}
    result_type = promote_type(T, eltype(op.op), eltype(src))
    dst = similar(src, result_type)
    mul!(dst, op, src)
end

@inbounds @muladd function add_coupling!(
        dst::AbstractVector, src::AbstractVector,
        numblocks::Int, blocklen::Int,
        σ₁, σₙ)
    diff = src[begin] - src[end]

    dst[begin] = dst[begin] + σ₁ * diff
    dst[end]   = dst[end] + σₙ * diff

    for i in 1:(numblocks - 1)
        iₗ = i * blocklen
        iᵣ = i * blocklen - 1
        diff = src[begin + iₗ] - src[begin + iᵣ]

        dst[begin + iₗ] = dst[begin + iₗ] + σ₁ * diff
        dst[begin + iᵣ] = dst[begin + iᵣ] + σₙ * diff
    end
end

@muladd function LinearAlgebra.mul!(
        dst::AbstractVector, op::PeriodicMultiblockOperator, src::AbstractVector)
    @unpack numblocks, blocklen = op
    len = numblocks * blocklen

    @boundscheck @assert len == length(src)
    @boundscheck @assert len == length(dst)

    @inbounds for i in 0:blocklen:(len - 1)
        @views mul!(
            dst[(begin + i):(begin + i + blocklen - 1)],
            op.op,
            src[(begin + i):(begin + i + blocklen - 1)]
        )
    end

    add_coupling!(dst, src, numblocks, blocklen, op.σ₁, op.σₙ)
    return dst
end
@muladd function LinearAlgebra.mul!(
        C::AbstractVector,
        A::PeriodicMultiblockOperator,
        B::AbstractVector,
        α, β)
    @unpack numblocks, blocklen = A
    len = numblocks * blocklen

    @boundscheck @assert len == length(C)
    @boundscheck @assert len == length(B)

    @inbounds for i in 0:blocklen:(len - 1)
        @views mul!(
            C[(begin + i):(begin + i + blocklen - 1)],
            A.op,
            B[(begin + i):(begin + i + blocklen - 1)],
            α,
            β
        )
    end

    add_coupling!(C, B, numblocks, blocklen, α * A.σ₁, α * A.σₙ)
    return C
end

function make_surface_operator(op, numblocks::Int64, blocksize, upwind::Symbol)
    h₁, hₙ = blocksize .* (left_boundary_weight(op), right_boundary_weight(op))
    blocklen = length(grid(op))
    σ₁, σₙ = upwind == :+ ? (zero(h₁), 1 / hₙ) :
             upwind == :- ? (1 / h₁, zero(h₁)) :
             upwind == :c ? (1 / (2h₁), 1 / (2hₙ)) :
             @error "unknown upwinding type. Expected one of ':+', ':-', or ':c'" upwind
    return PeriodicMultiblockOperator(NullOperator(), σ₁, σₙ, numblocks, blocklen)
end

function make_volume_operator(op, numblocks::Int64, blocksize::T) where {T}
    blocklen = length(grid(op))
    PeriodicMultiblockOperator(
        inv(blocksize) * op, zero(eltype(op)), zero(eltype(op)), numblocks, blocklen)
end

