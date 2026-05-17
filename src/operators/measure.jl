using SummationByPartsOperators: mass_matrix

export VolumeMeasure

abstract type AbstractMeasure end

(μ::AbstractMeasure)(ϕ::AbstractArray)     = μ(identity, ϕ)
(μ::AbstractMeasure)(fn, ϕ::AbstractArray) = _integrate(fn ∘ only, eltype(ϕ), (ϕ,), μ)
function (μ::AbstractMeasure)(fn, args::Tuple)
    T = promote_type(eltype.(args)...)
    _integrate(fn, T, args, μ)
end

abstract type AbstractCellMeasure{Dims, T} <: AbstractMeasure end

struct TensorProductMeasure{Dims, T} <: AbstractCellMeasure{Dims, T}
    weights::NTuple{Dims, Vector{T}}
end

function _integrate(fn::Fn, ::Type{T}, args::Tuple, μ::TensorProductMeasure) where {T, Fn}
    ws = μ.weights

    Is = CartesianIndices(length.(ws))
    @boundscheck checkbounds.(args, (Is,))

    acc::T = zero(T)
    @inbounds @simd ivdep for I in Is
        w = prod(getindex.(ws, Tuple(I)))
        f = fn(getindex.(args, I))
        acc = muladd(w, f, acc)
    end
    return acc
end

struct VolumeMeasure{Grid, CellMeasure <: AbstractCellMeasure} <: AbstractMeasure
    grid::Grid
    μ̂::CellMeasure
end

function VolumeMeasure(grid, fdop::Tuple)
    get_weights(ops) = (parent ∘ mass_matrix ∘ only)(ops.D.op.operators)
    μ̂ = TensorProductMeasure(get_weights.(fdop))
    return VolumeMeasure(grid, μ̂)
end

function _integrate(fn::Fn, ::Type{T}, args::Tuple, μ::VolumeMeasure) where {T, Fn}
    μ̂ = μ.μ̂
    acc::T = zero(T)
    foreachelement(μ.grid) do I
        acc += _integrate(fn, T, view.(args, (I,)), μ̂)
    end

    domain = μ.grid.domain
    blocks = μ.grid.blocks
    detΦ = prod(@. last(domain) - first(domain)) / prod(blocks)

    return detΦ * acc
end
