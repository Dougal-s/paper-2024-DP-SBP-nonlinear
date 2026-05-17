using LinearAlgebra
using LoopVectorization: @turbo
using MuladdMacro: @muladd

export MemoryPool
export returnblocks, getblock

export axis_mul!
export add_splitting!

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

"""
    axis_mul!(dst::AbstractArray, ::Val{N}, D, m::AbstractArray, cache)

Applies the operator `D` along the `N`-th dimension of the array `m` and stores
the result in dst.
"""
function axis_mul!(
        dst::AbstractArray{T, M},
        ::Val{1},
        D,
        m::AbstractArray{T, M},
        _...
) where {T, M}
    @inbounds for i in CartesianIndices(size(m)[2:M])
        @views mul!(dst[:, i], D, m[:, i])
    end
    dst
end,
function axis_mul!(
        dst::AbstractArray{T, M},
        ::Val{N},
        D,
        m::AbstractArray{T, M},
        cache
) where {N, T, M}
    @assert 2 ≤ N ≤ M
    σ₁ᴺ = (N, (2:(N - 1))..., 1, (N + 1):M...)
    @turbo cache .= PermutedDimsArray(m, σ₁ᴺ)
    axis_mul!(PermutedDimsArray(dst, σ₁ᴺ), Val(1), D, cache)
    dst
end

"""
    axis_mul!(mempool::MemoryPool, args...)

Allocates a temporary array using `dst` and forwards the argument to `axis_mul!`.
"""
axis_mul!(mempool::MemoryPool, args...) = axis_mul!(getblock(mempool), args...)


add_splitting!(dst::AbstractArray, dim::Val{1}, grid, Diᵥ, Diₛ, λs, m, tmp) =
    add_splitting!(dst, dim, grid, Diᵥ, Diₛ, λs, m, tmp, nothing)

# Global upwinding
# function add_splitting!(
#         dst::AbstractArray,
#         dim::Val,
#         grid,
#         Diᵥ, Diₛ,
#         λs,
#         m,
#         tmp, permute_cache
#     )
#     axis_mul!(tmp, dim, Diᵥ + Diₛ, m, permute_cache)
#     λ = @fastmath(maximum)(λs)
#     @. dst = muladd(λ, tmp, dst)
# end

# Local upwinding
function add_splitting!(
        dst::AbstractArray,
        dim::Val,
        grid,
        Diᵥ, Diₛ,
        λs,
        m,
        tmp,
        permute_cache
    )
    axis_mul!(tmp, dim, Diᵥ, m, permute_cache)
    @inbounds foreachelement(grid) do I
        λ = @fastmath(maximum)(view(λs, I))
        @views @. dst[I] = muladd(λ, tmp[I], dst[I])
    end
    axis_mul!(tmp, dim, Diₛ, m, permute_cache)
    @inbounds foreachinterface(grid, dim) do I₋, I₊
        for (i₋, i₊) in zip(I₋, I₊)
            λ = @fastmath(max)(λs[i₋], λs[i₊])
            dst[i₋] = muladd(λ, tmp[i₋], dst[i₋])
            dst[i₊] = muladd(λ, tmp[i₊], dst[i₊])
        end
    end
end

