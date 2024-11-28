using Symbolics
using Base.Threads
using LinearAlgebra

import Base: *

mutable struct MemoryPool{T, Dims}
    blocks::Vector{T}
    blocks_in_use::Int64
    blockdims::NTuple{Dims, Int64}
end

function MemoryPool(T, dims)
    MemoryPool{Array{T, length(dims)}, length(dims)}(
        Array{T, length(dims)}[],
        0,
        dims
    )
end

function MemoryPool(s₀::Array)
    MemoryPool{typeof(s₀), ndims(s₀)}(
        typeof(s₀)[],
        0,
        size(s₀)
    )
end

function returnblocks(mempool::MemoryPool{T, Dims}) where {T, Dims}
    mempool.blocks_in_use = 0
end

function getblock(mempool::MemoryPool{T, Dims}) where {T, Dims}
    mempool.blocks_in_use += 1
    if mempool.blocks_in_use > length(mempool.blocks)
        push!(mempool.blocks, similar(T, mempool.blockdims...))
    end
    mempool.blocks[mempool.blocks_in_use]
end

# derivatives
# Extends 1D derivative operators to 2D dimensions

function ∂y2D!(dst, D, m; cache = similar(m))
    permutedims!(dst, m, (2, 1))
    permutedims!(dst, ∂x2D!(cache, D, dst), (2, 1))
    dst
end

function ∂x2D!(dst, D, m)
    @inbounds for i in axes(m, 2)
        @views mul!(dst[:, i], D, m[:, i])
    end
    dst
end

function ∂y2D(mempool::MemoryPool, D, m; cache = getblock(mempool))
    ∂y2D!(getblock(mempool), D, m; cache = cache)
end
∂x2D(mempool::MemoryPool, D, m) = ∂x2D!(getblock(mempool), D, m)

∂y2D(D, m) = ∂y2D!(similar(m), D, m)
∂x2D(D, m) = ∂x2D!(similar(m), D, m)

#
finitemaximum(f, it) = maximum(Iterators.filter(isfinite, Iterators.map(f, it)))

# Symbolic Utilities

# Used to emulate ∇
struct VectorOperator
    scale::Number
    ops::Vector{Any}
end

VectorOperator(ops::AbstractVector) = VectorOperator(1, ops)

Base.getindex(op::VectorOperator, i::Int64) = op.ops[i]

(op::VectorOperator)(x::Number) = op.scale * map(f -> f(x), op.ops)

*(x::Number, op::VectorOperator) = VectorOperator(x * op.scale, op.ops)

function LinearAlgebra.dot(op::VectorOperator, v::AbstractVector)
    op.scale * sum(((f, vᵢ),) -> f(vᵢ), zip(op.ops, v))
end

LinearAlgebra.dot(op::VectorOperator, m::AbstractMatrix) = [op ⋅ v for v in eachrow(m)]

function make_source_modifier_from_syms(
        statevars::Tuple,
        source_terms::Dict,
        grid
)
    if !(grid isa Tuple)
        grid = tuple(grid)
    end

    source_terms_expr = Dict(zip(
        keys(source_terms),
        map(Symbolics.toexpr ∘ simplify ∘ expand_derivatives, values(source_terms))
    ))

    add_source_terms! = eval(quote
        (ds, t::Real) -> begin
            for (i, x) in enumerate(Iterators.product($(grid...)))
                $(Expr(:block,
                    map(statevars) do var
                        :(ds.$var[i] += $(source_terms_expr[var]))
                    end...))
            end
            return nothing
        end
    end)

    return (ds, s, t::Real, _...) -> begin
        @invokelatest add_source_terms!(ds, t)
        return nothing
    end
end
