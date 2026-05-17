using HyperbolicPDEs: NullOperator
using LinearAlgebra
using NaNMath
using RecursiveArrayTools: NamedArrayPartition
using StaticArrays
using UnPack

"""
    ShallowWaterScheme{N, T <: Real, StateVars}

Abstract supertype for all N-dimensional Shallow Water equation
semidiscretisations with prognostic variables `StateVars`.
"""
abstract type ShallowWaterScheme{N, T <: Real, StateVars} end;

function to_primitive_vars(eqs::ShallowWaterScheme{N, T}, state) where {N, T}
    dst = ntuple(_ -> Array{T}(undef, size(state.h)), Val(2 + N))
    to_primitive_vars!(dst, eqs, state)
    return dst
end

@kwdef struct FluxLaxFriedrichs{T}
    volume::Bool  = true
    surface::Bool = true
    scaling::T    = 1.0
end

@kwdef struct FluxEntropyStable{T}
    volume::Bool  = true
    surface::Bool = true
    scaling::T    = 1.0
end

struct SourceMMS{T}
    exact::T
end

####################
# 1D Shallow Water #
####################

"""
"""
@kwdef struct ShallowWaterFluxForm1D{T <: Real, Modifiers} <:
              ShallowWaterScheme{1, T, (:h, :hu, :b)}
    g::T = 1.0
    modifiers::Modifiers = ()
end

function ShallowWaterFluxForm1D(mods...; kwargs...)
    ShallowWaterFluxForm1D(; modifiers = Tuple(mods), kwargs...)
end

function from_primitive_vars(::ShallowWaterScheme{1, T, (:h, :hu, :b)}, (h, u, b)) where {T}
    return NamedArrayPartition((; h = h, hu = h .* u, b = b))
end

function to_primitive_vars!(dst, ::ShallowWaterScheme{1, T, (:h, :hu, :b)}, state) where {T}
    @unpack h, hu, b = state
    dst[1] .= @. h
    dst[2] .= @. hu / h
    dst[3] .= @. b
    return ()
end

function semidiscretise(info::ShallowWaterFluxForm1D{T}, grid, fdop;
        alloc = () -> Array{T}(undef, size(grid))
) where {T}
    @unpack g = info

    D = (@unpack D, B = fdop[1]; D + B)
    ∂x!(dst, m) = mul!(dst, D, m)

    apply_modifier!s = map(
        mod -> make_modifier(info, mod, grid, fdop, alloc), info.modifiers)

    fluxₕᵤ = alloc()
    b² = alloc()
    ∂ₓhu = alloc()
    ∂ₓfluxₕᵤ = alloc()

    ∂ₓb = alloc()
    ∂ₓb² = alloc()

    (ds, s, _, t) -> begin
        @unpack h, hu, b = s

        @. fluxₕᵤ = hu^2 / h + 1 // 2 * g * h^2
        @. b² = b^2

        ∂x!(∂ₓhu, hu)
        ∂x!(∂ₓfluxₕᵤ, fluxₕᵤ)
        ∂x!(∂ₓb, b)
        ∂x!(∂ₓb², b²)

        @. ds.h  = -∂ₓhu
        @. ds.hu = -∂ₓfluxₕᵤ - g * (h + b) * ∂ₓb + 1 // 2 * g * ∂ₓb²
        @. ds.b  = $zero(T)

        for apply_mod! in apply_modifier!s
            apply_mod!(ds, s, t)
        end

        return nothing
    end
end

"""
"""
@kwdef struct ShallowWaterSkewSym1D{T <: Real, Modifiers} <:
              ShallowWaterScheme{1, T, (:h, :hu, :b)}
    g::T = 1.0
    modifiers::Modifiers = ()
end

function ShallowWaterSkewSym1D(mods...; kwargs...)
    ShallowWaterSkewSym1D(; modifiers = Tuple(mods), kwargs...)
end

function semidiscretise(info::ShallowWaterSkewSym1D{T}, grid, fdop;
        alloc = () -> Array{T}(undef, size(grid))
) where {T}
    @unpack g = info

    D = (@unpack D, B = fdop[1]; D + B)
    ∂x!(dst, m) = mul!(dst, D, m)

    apply_modifier!s = map(
        mod -> make_modifier(info, mod, grid, fdop, alloc), info.modifiers)

    H   = alloc()
    u   = alloc()
    huu = alloc()

    ∂ₓH = alloc()
    ∂ₓu = alloc()
    ∂ₓhu = alloc()
    ∂ₓhuu = alloc()

    (ds, s, _, t) -> begin
        @unpack h, hu, b = s

        @. H = h + b
        @. u = hu / h
        @. huu = hu * u

        ∂x!(∂ₓH, H)
        ∂x!(∂ₓu, u)
        ∂x!(∂ₓhu, hu)
        ∂x!(∂ₓhuu, huu)

        ds.h  .= @. -∂ₓhu
        ds.hu .= @. -1 // 2 * (∂ₓhuu + u * ∂ₓhu + hu * ∂ₓu) - g * h * ∂ₓH
        ds.b  .= zero(T)

        for apply_mod! in apply_modifier!s
            apply_mod!(ds, s, t)
        end

        return nothing
    end
end

###################
# Flux Splittings #
###################

function make_modifier(
        eqs::ShallowWaterScheme{1, T, (:h, :hu, :b)},
        splitting::FluxLaxFriedrichs,
        grid,
        fdop,
        allocator
) where {T}
    @unpack g = eqs

    Diᵥ = splitting.volume ? splitting.scaling * fdop[1].Diᵥ : NullOperator()
    Diₛ = splitting.surface ? splitting.scaling * fdop[1].Diₛ : NullOperator()

    H = allocator()
    λ = allocator()
    tmp = allocator()

    (ds, s, _) -> begin
        @unpack h, hu, b = s

        @. λ = abs(hu / h) + NaNMath.sqrt(g * h)
        @. H = h + b

        add_splitting!(ds.h, Val(1), grid, Diᵥ, Diₛ, λ, H, tmp)
        add_splitting!(ds.hu, Val(1), grid, Diᵥ, Diₛ, λ, hu, tmp)
    end
end

function make_modifier(
        eqs::ShallowWaterScheme{1, T, (:h, :hu, :b)},
        splitting::FluxEntropyStable,
        grid,
        fdop,
        allocator
) where {T}
    @unpack g = eqs

    Diᵥ = splitting.volume ? splitting.scaling * fdop[1].Diᵥ : NullOperator()
    Diₛ = splitting.surface ? splitting.scaling * fdop[1].Diₛ : NullOperator()

    u  = allocator()
    gₕ = allocator()

    λh  = allocator()
    λhu = allocator()

    tmp = allocator()

    (ds, s, _) -> begin
        @unpack h, hu, b = s

        @inbounds @simd ivdep for i in eachindex(h)
            hᵢ, huᵢ, bᵢ = h[i], hu[i], b[i]

            uᵢ  = huᵢ / hᵢ
            c = NaNMath.sqrt(g * hᵢ)
            a = abs(uᵢ) + c

            gₕ[i] = g * (hᵢ + bᵢ / 2) - uᵢ^2 / 2
            u[i] = uᵢ

            # λh[i]  = 6(c / g)
            λh[i]  = 8a * hᵢ / (hᵢ * g + uᵢ^2)
            λhu[i] = 8a * hᵢ
        end

        add_splitting!(ds.h, Val(1), grid, Diᵥ, Diₛ, λh, gₕ, tmp)
        add_splitting!(ds.hu, Val(1), grid, Diᵥ, Diₛ, λhu, u, tmp)
    end
end

####################
# 2D Shallow Water #
####################

"""
    struct ShallowWaterFluxForm2D{T <: Real, Modifiers} <: ShallowWaterScheme

# Fields
- `f::T` - Coriolis frequency
- `g::T` - gravitational acceleration
"""
@kwdef struct ShallowWaterFluxForm2D{T <: Real, Modifiers} <:
              ShallowWaterScheme{2, T, (:h, :hu, :hv, :b)}
    f::T = 0.0
    g::T = 1.0

    modifiers::Modifiers = ()
end

function ShallowWaterFluxForm2D(mods...; kwargs...)
    ShallowWaterFluxForm2D(; modifiers = Tuple(mods), kwargs...)
end

function from_primitive_vars(
        ::ShallowWaterScheme{2, T, (:h, :hu, :hv, :b)},
        (h, u, v, b)
) where {T}
    return NamedArrayPartition((; h = h, hu = h .* u, hv = h .* v, b = b))
end

function to_primitive_vars!(
        dst,
        ::ShallowWaterScheme{2, T, (:h, :hu, :hv, :b)},
        state
) where {T}
    @unpack h, hu, hv, b = state
    dst[1] .= @. h
    dst[2] .= @. hu / h
    dst[3] .= @. hv / h
    dst[4] .= @. b
    return ()
end

function semidiscretise(info::ShallowWaterFluxForm2D{T}, grid, fdop;
        alloc = () -> Array{T}(undef, size(grid))
) where {T}
    @unpack modifiers, g, f = info

    mempool = MemoryPool(alloc())
    permute_cache = alloc()

    Dx = (@unpack D, B = fdop[1]; D + B)
    ∂x(m) = axis_mul!(mempool, Val(1), Dx, m, permute_cache)

    Dy = (@unpack D, B = fdop[2]; D + B)
    ∂y(m) = axis_mul!(mempool, Val(2), Dy, m, permute_cache)

    apply_modifier!s = map(
        mod -> make_modifier(info, mod, grid, fdop, alloc, mempool), info.modifiers)

    fluxˣₕᵤ = alloc()
    huv     = alloc()
    fluxʸₕᵥ = alloc()
    b²      = alloc()

    (ds, s, _, t) -> begin
        returnblocks(mempool)
        @unpack h, hu, hv, b = s

        @. fluxˣₕᵤ = hu * hu / h + 1 // 2 * g * h^2
        @. fluxʸₕᵥ = hv * hv / h + 1 // 2 * g * h^2
        @. huv = hu * hv / h
        @. b²  = b^2

        @. ds.h  = -$∂x(hu) - $∂y(hv)
        @. ds.hu = (-$∂x(fluxˣₕᵤ) - $∂y(huv)
            + f * hv
            - g * (h + b) * $∂x(b) + 1 // 2 * g * $∂x(b²)
        )
        @. ds.hv = (-$∂x(huv) - $∂y(fluxʸₕᵥ)
            - f * hu
            - g * (h + b) * $∂y(b) + 1 // 2 * g * $∂y(b²)
        )
        @. ds.b  = $zero(T)

        for apply_mod! in apply_modifier!s
            apply_mod!(ds, s, t)
        end

        return
    end
end

"""
    struct ShallowWaterVectorInv2D{T <: Real, Modifiers} <: ShallowWaterScheme

# Fields
- `f::T = 0.0` - Coriolis frequency
- `g::T = 1.0` - gravitational acceleration
"""
@kwdef struct ShallowWaterVectorInv2D{T <: Real, Modifiers} <:
              ShallowWaterScheme{2, T, (:h, :u, :v, :b)}
    f::T = 0.0
    g::T = 1.0

    modifiers::Modifiers = ()
end

function ShallowWaterVectorInv2D(mods...; kwargs...)
    ShallowWaterVectorInv2D(; modifiers = Tuple(mods), kwargs...)
end

function from_primitive_vars(::ShallowWaterVectorInv2D, (h, u, v, b))
    return NamedArrayPartition((; h = h, u = u, v = v, b = b))
end

function to_primitive_vars!(dst, ::ShallowWaterVectorInv2D, state)
    @unpack h, u, v, b = state
    dst[1] .= @. h
    dst[2] .= @. u
    dst[3] .= @. v
    dst[4] .= @. b
    return ()
end

function semidiscretise(info::ShallowWaterVectorInv2D{T}, grid, fdop;
        alloc = () -> Array{T}(undef, size(grid))
) where {T}
    @unpack modifiers, g, f = info

    mempool = MemoryPool(alloc())
    permute_cache = alloc()

    Dx = (@unpack D, B = fdop[1]; D + B)
    ∂x(m) = axis_mul!(mempool, Val(1), Dx, m, permute_cache)

    Dy = (@unpack D, B = fdop[2]; D + B)
    ∂y(m) = axis_mul!(mempool, Val(2), Dy, m, permute_cache)

    apply_modifier!s = map(
        mod -> make_modifier(info, mod, grid, fdop, alloc, mempool), info.modifiers)

    hu = alloc()
    hv = alloc()
    ω  = alloc()
    G  = alloc()

    (ds, s, _, t) -> begin
        returnblocks(mempool)

        @unpack h, u, v, b = s

        @. hu = u * h
        @. hv = v * h
        @. ω = $∂x(v) - $∂y(u) + f
        @. G = (u^2 + v^2) / 2 + g * h

        @. ds.h = -$∂x(hu) - $∂y(hv)
        @. ds.u = ω * v - $∂x(G)
        @. ds.v = -ω * u - $∂y(G)
        @. ds.b = $zero(T)

        for apply_mod! in apply_modifier!s
            apply_mod!(ds, s, t)
        end

        return nothing
    end
end

"""
    struct ShallowWaterSkewSym2D{T <: Real, Modifiers} <: ShallowWaterScheme

# Fields
- `f::T` - Coriolis frequency
- `g::T` - gravitational acceleration
"""
@kwdef struct ShallowWaterSkewSym2D{T <: Real, Modifiers} <:
              ShallowWaterScheme{2, T, (:h, :hu, :hv, :b)}
    f::T = 0.0
    g::T = 1.0

    modifiers::Modifiers = ()
end

function ShallowWaterSkewSym2D(mods...; kwargs...)
    ShallowWaterSkewSym2D(; modifiers = Tuple(mods), kwargs...)
end

function semidiscretise(info::ShallowWaterSkewSym2D{T}, grid, fdop;
        alloc = () -> Array{T}(undef, size(grid))
) where {T}
    @unpack modifiers, g, f = info

    mempool = MemoryPool(alloc())
    permute_cache = alloc()

    Dx = (@unpack D, B = fdop[1]; D + B)
    ∂x(m) = axis_mul!(mempool, Val(1), Dx, m, permute_cache)

    Dy = (@unpack D, B = fdop[2]; D + B)
    ∂y(m) = axis_mul!(mempool, Val(2), Dy, m, permute_cache)

    apply_modifier!s = map(
        mod -> make_modifier(info, mod, grid, fdop, alloc, mempool), info.modifiers)

    H   = alloc()
    huv = alloc()
    huu = alloc()
    hvv = alloc()
    u   = alloc()
    v   = alloc()

    ∂x_hu = alloc()
    ∂y_hv = alloc()

    (ds, s, _, t) -> begin
        returnblocks(mempool)

        @unpack h, hu, hv, b = s

        @. H = h + b
        @. u = hu / h
        @. v = hv / h
        @. huu = hu * u
        @. huv = hu * v
        @. hvv = hv * v

        @. ∂x_hu = $∂x(hu)
        @. ∂y_hv = $∂y(hv)

        @. ds.h = -∂x_hu - ∂y_hv

        @. ds.hu = -1 // 2 * ($∂x(huu) + u * ∂x_hu + hu * $∂x(u)) -
                   1 // 2 * ($∂y(huv) + u * ∂y_hv + hv * $∂y(u)) -
                   g * h * $∂x(H) +
                   f * hv

        @. ds.hv = -1 // 2 * ($∂x(huv) + v * ∂x_hu + hu * $∂x(v)) -
                   1 // 2 * ($∂y(hvv) + v * ∂y_hv + hv * $∂y(v)) -
                   g * h * $∂y(H) -
                   f * hu

        @. ds.b = $zero(T)

        for apply_mod! in apply_modifier!s
            apply_mod!(ds, s, t)
        end

        return nothing
    end
end

#######
# MMS #
#######

⊥(u::NTuple{2}) = (-u[2], u[1])
⊥(::NTuple{1}) = (false,)

function make_modifier(
        info::ShallowWaterScheme{N, T, StateVars},
        mms::SourceMMS,
        grid,
        _...
) where {N, T, StateVars}
    ∂t = SymUtils.∂t
    ∇  = ntuple(i -> SymUtils.SpatialDerivative{i}(), Val(N))
    ∀  = Base.Fix{2}(ntuple, Val(N))

    h, v⃗..., b = SymUtils.LazyField.(mms.exact)

    g = info.g
    f = N == 2 ? info.f : false

    hv⃗ = h .* v⃗
    ω = ∇ ⋅ ⊥(v⃗) + f
    G = v⃗ ⋅ v⃗ / 2 + g * h

    hv⃗_sym = [:hu, :hv][1:N]
    v⃗_sym = [:u, :v][1:N]

    eqs_hv⃗ = ∀(i -> ∂t(hv⃗[i]) + ∇ ⋅ (hv⃗[i] .* v⃗) + g * h * ∇[i](h + b) + f * ⊥(hv⃗)[i])
    eqs_v⃗  = ∀(i -> ∂t(v⃗[i]) + ω * ⊥(v⃗)[i] + ∇[i]G)

    eqs = (;
        h = ∂t(h) + ∇ ⋅ hv⃗,
        b = ∂t(b),
        (@. hv⃗_sym => eqs_hv⃗)...,
        (@. v⃗_sym => eqs_v⃗)...
    )

    SymUtils.make_source_modifier_from_syms(StateVars, eqs, grid)
end

###################
# Flux Splittings #
###################

function make_modifier(
        eqs::ShallowWaterScheme{2, T, (:h, :hu, :hv, :b)},
        splitting::FluxLaxFriedrichs,
        grid,
        fdop,
        alloc,
        mempool
) where {T}
    @unpack g = eqs

    tmp = alloc()
    permute_cache = alloc()

    Dixᵥ = (splitting.volume ? splitting.scaling * fdop[1].Diᵥ : NullOperator())
    Dixₛ = (splitting.surface ? splitting.scaling * fdop[1].Diₛ : NullOperator())
    Diyᵥ = (splitting.volume ? splitting.scaling * fdop[2].Diᵥ : NullOperator())
    Diyₛ = (splitting.surface ? splitting.scaling * fdop[2].Diₛ : NullOperator())

    H = alloc()
    λ = ntuple(_ -> alloc(), Val(2))

    (ds, s, t) -> begin
        @unpack h, hu, hv, b = s

        @. H = h + b

        @inbounds for i in eachindex(h)
            p = NaNMath.sqrt(g * h[i])
            λ[1][i] = abs(hu[i] / h[i]) + p
            λ[2][i] = abs(hv[i] / h[i]) + p
        end

        for (dqdt, q) in ((ds.h, H), (ds.hu, hu), (ds.hv, hv))
            add_splitting!(dqdt, Val(1), grid, Dixᵥ, Dixₛ, λ[1], q, tmp, permute_cache)
            add_splitting!(dqdt, Val(2), grid, Diyᵥ, Diyₛ, λ[2], q, tmp, permute_cache)
        end
    end
end

function make_modifier(
        eqs::ShallowWaterScheme{2, T, (:h, :hu, :hv, :b)},
        splitting::FluxEntropyStable,
        grid,
        fdop,
        alloc,
        mempool
) where {T}
    @unpack g = eqs

    tmp = alloc()
    permute_cache = alloc()

    Dixᵥ = (splitting.volume ? splitting.scaling * fdop[1].Diᵥ : NullOperator())
    Dixₛ = (splitting.surface ? splitting.scaling * fdop[1].Diₛ : NullOperator())
    Diyᵥ = (splitting.volume ? splitting.scaling * fdop[2].Diᵥ : NullOperator())
    Diyₛ = (splitting.surface ? splitting.scaling * fdop[2].Diₛ : NullOperator())

    u  = alloc()
    v  = alloc()
    gh = alloc()

    λhs  = ntuple(_ -> alloc(), 2)
    λhus = ntuple(_ -> alloc(), 2)
    λhvs = ntuple(_ -> alloc(), 2)

    (ds, s, t) -> begin
        @unpack h, hu, hv, b = s

        @inbounds @simd ivdep for i in eachindex(h)
            hᵢ, hv⃗ᵢ, bᵢ = h[i], @SVector[hu[i], hv[i]], b[i]

            v⃗ᵢ = hv⃗ᵢ / hᵢ
            aᵢ  = abs.(v⃗ᵢ) .+ NaNMath.sqrt(g * hᵢ)

            λhsᵢ = aᵢ * hᵢ / (hᵢ * g + v⃗ᵢ ⋅ v⃗ᵢ)
            λhusᵢ = aᵢ * hᵢ
            λhvsᵢ = aᵢ * hᵢ

            u[i], v[i]  = hv⃗ᵢ / hᵢ
            gh[i] = g * (hᵢ + bᵢ) - 1 // 2 * v⃗ᵢ ⋅ v⃗ᵢ

            for d in 1:2
                λhs[d][i]  = λhsᵢ[d]
                λhus[d][i] = λhusᵢ[d]
                λhvs[d][i] = λhvsᵢ[d]
            end
        end

        add_splitting!(ds.h, Val(1), grid, Dixᵥ, Dixₛ, λhs[1], gh, tmp, permute_cache)
        add_splitting!(ds.h, Val(2), grid, Diyᵥ, Diyₛ, λhs[2], gh, tmp, permute_cache)

        add_splitting!(ds.hu, Val(1), grid, Dixᵥ, Dixₛ, λhus[1], u, tmp, permute_cache)
        add_splitting!(ds.hu, Val(2), grid, Diyᵥ, Diyₛ, λhus[2], u, tmp, permute_cache)

        add_splitting!(ds.hv, Val(1), grid, Dixᵥ, Dixₛ, λhvs[1], v, tmp, permute_cache)
        add_splitting!(ds.hv, Val(2), grid, Diyᵥ, Diyₛ, λhvs[2], v, tmp, permute_cache)
    end
end

function make_modifier(
        eqs::ShallowWaterVectorInv2D{T},
        splitting::FluxEntropyStable,
        grid,
        fdop,
        alloc,
        mempool
) where {T}
    @unpack g = eqs

    permute_cache = alloc()
    Dxₛ = splitting.scaling * ((splitting.volume ? fdop[1].Diᵥ : NullOperator()) +
                               (splitting.surface ? fdop[1].Diₛ : NullOperator()))
    ∂xs(m) = axis_mul!(mempool, Val(1), Dxₛ, m, permute_cache)

    Dyₛ = splitting.scaling * ((splitting.volume ? fdop[2].Diᵥ : NullOperator()) +
                               (splitting.surface ? fdop[2].Diₛ : NullOperator()))
    ∂ys(m) = axis_mul!(mempool, Val(2), Dyₛ, m, permute_cache)

    hu = alloc()
    hv = alloc()
    G  = alloc()

    (ds, s, t) -> begin
        @unpack h, u, v, b = s

        @. hu = u * h
        @. hv = v * h
        @. G = (u^2 + v^2) / 2 + g * h

        α = zero(T)
        β = zero(T)
        @inbounds for i in eachindex(h)
            c = hypot(u[i], v[i]) + sqrt(g * h[i])
            α = max(α, c / (2h[i]))
            β = max(β, g / (2c))
        end

        @. ds.h += α * ($∂xs(G) + $∂ys(G))
        @. ds.u += β * ($∂xs(hu) + $∂ys(hu))
        @. ds.v += β * ($∂xs(hv) + $∂ys(hv))
    end
end
