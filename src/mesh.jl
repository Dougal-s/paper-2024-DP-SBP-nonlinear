export CartesianMesh
export getaxis, foreachelement, foreachinterface
export local_dp_operator
export couple_operators
export interpolate_onto

using SummationByPartsOperators
using LinearAlgebra
using MuladdMacro: @muladd
using UnPack: @unpack

import Base: +, -, *, /
import LinearAlgebra: ×, mul!

struct CartesianMesh{T <: Real, N}
    domain::NTuple{N, Tuple{T, T}}
    blocks::NTuple{N, Int64}
end

Base.ndims(::CartesianMesh{T, N}) where {T, N} = N

function ×(lhs::CartesianMesh{T, N}, rhs::CartesianMesh{T, M}) where {T, N, M}
    return CartesianMesh(
        (lhs.domain..., rhs.domain...),
        (lhs.blocks..., rhs.blocks...)
    )
end

struct TensorProdGrid{T <: Real, N, BlockNodes} <: AbstractArray{NTuple{N, T}, N}
    domain::NTuple{N, Tuple{T, T}}
    blocks::NTuple{N, Int64}
    nodes::NTuple{N, BlockNodes}
end

Base.ndims(::TensorProdGrid{T, N}) where {T, N} = N

function Base.step(mesh::TensorProdGrid{T, N}) where {T, N}
    Δx_local = ntuple(Val(N)) do d
        mapreduce(-, min, Iterators.drop(mesh.nodes[d], 1), mesh.nodes[d])
    end
    mesh_range = @. last(mesh.domain) - first(mesh.domain)
    Δx = @. Δx_local * mesh_range / mesh.blocks
    return Δx
end

Base.size(mesh::TensorProdGrid{T, N}) where {T, N} = mesh.blocks .* length.(mesh.nodes)

function Base.getindex(mesh::TensorProdGrid{T, N}, I::Vararg{Int, N}) where {T, N}
    @unpack domain, blocks, nodes = mesh
    blocksize  = @. length(nodes)
    block      = @. (I - 1) ÷ blocksize + 1
    x_ref      = @. getindex(nodes, (I - 1) % blocksize + 1)
    boundaries = @. range(first(domain), last(domain), blocks + 1)
    l = @. getindex(boundaries, block)
    r = @. getindex(boundaries, block + 1)
    ϵ = @. eps(max(abs(l), abs(r)))
    return @. (l + ϵ) * (1 - x_ref) + (r - ϵ) * x_ref
end

function getaxis(mesh::TensorProdGrid{T, N}, dim::Int) where {T, N}
    dₗ = first(mesh.domain[dim])
    dᵣ = last(mesh.domain[dim])
    blocks = mesh.blocks[dim]
    b_width = (dᵣ - dₗ) / blocks
    boundaries = range(dₗ, dᵣ, blocks + 1)

    ((@. boundaries[i] + b_width * mesh.nodes[dim]) for i in 1:blocks) |>
    Iterators.flatten |> collect
end

function getaxis(mesh::TensorProdGrid{T, N}) where {T, N}
    return ntuple(η -> getaxis(mesh, η), Val(N))
end

"""
    foreachelement(fn::Function, nodes::TensorProdGrid)
Calls `fn(I)` for each element `K` where `I` are the indices of all the nodes in `K`.
"""
function foreachelement(fn, space::TensorProdGrid{T, N}) where {T, N}
    refindices = CartesianIndices(length.(space.nodes))
    for cell in CartesianIndices(space.blocks)
        offset = @. ($Tuple(cell) - 1) * $size(refindices)
        fn(CartesianIndex(offset) .+ refindices)
    end
end

"""
    foreachinterface(fn::Function, nodes::TensorProdGrid)
Calls `fn(I₋, I₊)` for each interface `K₋ ∩ K₊` where `I₋` and `I₊` are the indices of the
nodes in elements `K₋` and `K₊` that lie on the interface.
"""
function foreachinterface(fn, mesh::TensorProdGrid{T, N}, ::Val{D}) where {T, N, D}
    I = CartesianIndices(mesh)
    blocks = mesh.blocks[D]
    blocklen = length(mesh.nodes[D])

    fn(selectdim(I, D, blocks * blocklen), selectdim(I, D, 1))
    for i in 1:(blocks - 1)
        fn(selectdim(I, D, i * blocklen), selectdim(I, D, i * blocklen + 1))
    end
end

function local_dp_operator(
        coefficient_source, accuracy_order::Int, Ns::NTuple{N, Int}) where {N}
    local_dp_operator(Float64, coefficient_source, accuracy_order, Ns)
end,
function local_dp_operator(
        T::Type,
        coefficient_source::Type,
        accuracy_order::Int,
        Ns::NTuple{N, Int}
) where {N}
    return ntuple(Val(N)) do i
        upwind_operators(coefficient_source;
            accuracy_order = accuracy_order,
            xmin = zero(T),
            xmax = one(T),
            N = Ns[i]
        )
    end
end

function couple_operators(
        mesh::CartesianMesh{T, N},
        local_op::NTuple{N, Op}
) where {T, N, Op}
    xs = TensorProdGrid(mesh.domain, mesh.blocks, grid.(local_op))
    blocksize = @. (last(mesh.domain) - first(mesh.domain)) / mesh.blocks
    fdop = ntuple(Val(N)) do i
        D₊ = make_volume_operator(local_op[i].plus, mesh.blocks[i], blocksize[i])
        D₋ = make_volume_operator(local_op[i].minus, mesh.blocks[i], blocksize[i])
        D  = make_volume_operator(local_op[i].central, mesh.blocks[i], blocksize[i])
        B₊ = make_surface_operator(local_op[i], mesh.blocks[i], blocksize[i], :+)
        B₋ = make_surface_operator(local_op[i], mesh.blocks[i], blocksize[i], :-)
        B  = make_surface_operator(local_op[i], mesh.blocks[i], blocksize[i], :c)

        Diᵥ = make_volume_operator(dissipation_operator(local_op[i]),
            mesh.blocks[i], blocksize[i])

        return (;
            D₊, D₋, D, Diᵥ,
            B₊, B₋, B, Diₛ = (B₊ - B₋) / 2
        )
    end
    return xs, fdop
end

"""
    interpolate_onto([op, ]new_xs, xs, us)

Falls back to linear interpolation when no specialized interpolation exists.
"""
function interpolate_onto(new_xs, xs, us)
    interpolate_onto(nothing, new_xs, xs, us)
end,
function interpolate_onto(op::MatrixDerivativeOperator, new_xs, xs, us)
    interpolate_onto(source_of_coefficients(op), new_xs, xs, us)
end,
function interpolate_onto(op::UpwindOperators, new_xs, xs, us)
    interpolate_onto(op.central, new_xs, xs, us)
end,
function interpolate_onto(_, new_xs, xs, us)
    srcᵢ = 1
    new_us = similar(us, size(new_xs))
    for dstᵢ in eachindex(new_xs)
        while new_xs[dstᵢ] > xs[srcᵢ + 1]
            srcᵢ += 1
        end

        t = (new_xs[dstᵢ] - xs[srcᵢ]) / (xs[srcᵢ + 1] - xs[srcᵢ])
        new_us[dstᵢ] = t * us[srcᵢ + 1] + (1 - t) * us[srcᵢ]
    end
    return new_us
end
