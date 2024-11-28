using Symbolics
using StaticArrays
using LinearAlgebra
using RecursiveArrayTools: NamedArrayPartition
using UnPack
using NaNMath

include(srcdir("pde-utils.jl"))

"""
    ShallowWaterScheme{N, T <: Real, StateVars}

Abstract supertype for all N-dimensional Shallow Water equation
semidiscretisations with prognostic variables `StateVars`.
"""
abstract type ShallowWaterScheme{N, T <: Real, StateVars} end;

"""
    global_cons_qtys(eq, (h, u[, v], b), Hx[, Hy])

Get the following conserved global quantities from the primitive variables:
    the total energy
    the total mass
    the total x momentum
    [the total y momentum]
"""
function global_cons_qtys(eqs::ShallowWaterScheme{N}, state, H...) where {N}
    @unpack g = eqs

    total_energy(s) = energy_norm(s, H...) do (h, u..., b)
        h * (1 // 2 * h + b) * g + 1 // 2 * h * (u ⋅ u)
    end
    total_mass(s)   = energy_norm(s, H...) do (h, u..., b)
        h
    end
    total_ϱu(s)     = (energy_norm(s, H...) do (h, u..., b)
        h * u[η]
    end
    for η in 1:N)

    return (total_energy(state), total_mass(state), total_ϱu(state)...)
end

function to_primitive_vars(eqs::ShallowWaterScheme{N, T}, state) where {N, T}
    dst = ntuple(_ -> Array{T}(undef, size(state.h)), 2 + N)
    to_primitive_vars!(dst, eqs, state)
    return dst
end

@kwdef struct FluxLaxFriedrichs{T}
    scaling::T = 1.0
end

@kwdef struct FluxEntropyStable{T}
    scaling::T = 1.0
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
        alloc = () -> Array{T}(undef, length(grid))
) where {T}
    @unpack g = info

    D₊, D₋ = fdop
    D      = (D₋ + D₊) / 2
    ∂x(m)  = D * m

    apply_modifier!s = map(
        mod -> make_modifier(info, mod, grid, fdop, alloc), info.modifiers)

    fluxₕᵤ = alloc()

    (ds, s, _, t) -> begin
        @unpack h, hu, b = s

        @. fluxₕᵤ = hu^2 / h + 1 // 2 * g * h^2

        @. ds.h  = -$∂x(hu)
        @. ds.hu = -$∂x(fluxₕᵤ) - g * (h + b) * $∂x(b) + 1 // 2 * g * $∂x(b^2)
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
        alloc = () -> Array{T}(undef, length(grid))
) where {T}
    @unpack g = info

    D₊, D₋ = fdop
    D      = (D₋ + D₊) / 2
    ∂x(m)  = D * m

    apply_modifier!s = map(
        mod -> make_modifier(info, mod, grid, fdop, alloc), info.modifiers)

    u   = alloc()
    huu = alloc()

    (ds, s, _, t) -> begin
        @unpack h, hu, b = s

        @. u = hu / h
        @. huu = hu * u

        ds.h  .= @. -$∂x(hu)
        ds.hu .= @. -1 // 2 * ($∂x(huu) + u * $∂x(hu) + hu * $∂x(u)) - g * h * $∂x(h) - g * h * $∂x(b)
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

    D₊, D₋ = fdop
    Dₛ     = (D₊ - D₋) / 2
    ∂xs(f) = Dₛ * f

    total_h = allocator()

    (ds, s, _) -> begin
        @unpack h, hu, b = s

        λ = splitting.scaling * finitemaximum(zip(h, hu)) do (h, hu)
            abs(hu / h) + NaNMath.sqrt(g * h)
        end

        @. total_h = h + b

        @. ds.h  += λ * $∂xs(total_h)
        @. ds.hu += λ * $∂xs(hu)

        return nothing
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

    D₊, D₋ = fdop
    Dₛ     = (D₊ - D₋) / 2
    ∂xs(f) = Dₛ * f

    u  = allocator()
    gₕ = allocator()

    (ds, s, _) -> begin
        @unpack h, hu, b = s

        λh = 2 * finitemaximum(zip(h, hu)) do (h, hu)
            h / (abs(hu / h) + NaNMath.sqrt(g * h))
        end
        λhu = 4 * finitemaximum(zip(h, hu)) do (h, hu)
            abs(hu) + 1 // 2 * abs(h) * NaNMath.sqrt(g * h)
        end
        λh, λhu = splitting.scaling .* (λh, λhu)

        @. u  = hu / h
        @. gₕ = g * (h + b) - 1 // 2 * u^2

        @. ds.h  += λh * $∂xs(gₕ)
        @. ds.hu += λhu * $∂xs(u)

        return nothing
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
        alloc = () -> Array{T}(undef, length.(grid)...)
) where {T}
    @unpack modifiers, g, f = info

    (Dx₊, Dx₋), (Dy₊, Dy₋) = fdop
    Dx = (Dx₋ + Dx₊) / 2
    Dy = (Dy₋ + Dy₊) / 2

    mempool = MemoryPool(alloc())
    ∂x(m)   = ∂x2D(mempool, Dx, m)
    ∂y(m)   = ∂y2D(mempool, Dy, m)

    apply_modifier!s = map(
        mod -> make_modifier(info, mod, grid, fdop, alloc, mempool), info.modifiers)

    fluxˣₕᵤ = alloc()
    huv     = alloc()
    fluxʸₕᵥ = alloc()

    (ds, s, _, t) -> begin
        returnblocks(mempool)
        @unpack h, hu, hv, b = s

        @. fluxˣₕᵤ = hu * hu / h + 1 // 2 * g * h^2
        @. fluxʸₕᵥ = hv * hv / h + 1 // 2 * g * h^2
        @. huv = hu * hv / h

        @. ds.h  = -$∂x(hu) - $∂y(hv)
        @. ds.hu = -$∂x(fluxˣₕᵤ) - $∂y(huv) + f * hv - g * h * $∂x(b)
        @. ds.hv = -$∂x(huv) - $∂y(fluxʸₕᵥ) - f * hu - g * h * $∂y(b)
        @. ds.b  = $zero(T)

        for apply_mod! in apply_modifier!s
            apply_mod!(ds, s, t)
        end

        return nothing
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
        alloc = () -> Array{T}(undef, length.(grid)...)
) where {T}
    @unpack modifiers, g, f = info

    (Dx₊, Dx₋), (Dy₊, Dy₋) = fdop
    Dx = (Dx₋ + Dx₊) / 2
    Dy = (Dy₋ + Dy₊) / 2

    mempool = MemoryPool(alloc())
    ∂x(m)   = ∂x2D(mempool, Dx, m)
    ∂y(m)   = ∂y2D(mempool, Dy, m)

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
        alloc = () -> Array{T}(undef, length.(grid)...)
) where {T}
    @unpack modifiers, g, f = info

    (Dx₊, Dx₋), (Dy₊, Dy₋) = fdop
    Dx = (Dx₋ + Dx₊) / 2
    Dy = (Dy₋ + Dy₊) / 2

    mempool = MemoryPool(alloc())
    ∂x(m)   = ∂x2D(mempool, Dx, m)
    ∂y(m)   = ∂y2D(mempool, Dy, m)

    apply_modifier!s = map(
        mod -> make_modifier(info, mod, grid, fdop, alloc, mempool), info.modifiers)

    H   = alloc()
    huv = alloc()
    huu = alloc()
    hvv = alloc()
    u   = alloc()
    v   = alloc()

    (ds, s, _, t) -> begin
        returnblocks(mempool)

        @unpack h, hu, hv, b = s

        @. H = h + b
        @. u = hu / h
        @. v = hv / h
        @. huu = hu * u
        @. huv = hu * v
        @. hvv = hv * v

        @. ds.h = -$∂x(hu) - $∂y(hv)

        @. ds.hu = -1 // 2 * ($∂x(huu) + u * $∂x(hu) + hu * $∂x(u)) -
                   1 // 2 * ($∂y(huv) + u * $∂y(hv) + hv * $∂y(u)) -
                   g * h * $∂x(H) +
                   f * hv

        @. ds.hv = -1 // 2 * ($∂x(huv) + v * $∂x(hu) + hu * $∂x(v)) -
                   1 // 2 * ($∂y(hvv) + v * $∂y(hv) + hv * $∂y(v)) -
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

⊥(u::SVector{2}) = SVector{2}(-u[2], u[1])
⊥(u::SVector{1}) = SVector{1}(zero(u))

function make_modifier(
        info::ShallowWaterScheme{N, T, StateVars},
        mms::SourceMMS,
        grid,
        _...
) where {N, T, StateVars}
    @variables t x[1:N]
    h, 𝐮..., b = Ref((t, x...)) .|> Base.splat.(mms.exact)
    𝐮 = SVector{N}(𝐮)

    g = info.g
    f = N == 2 ? info.f : 0

    h𝐮 = h * 𝐮
    hu_sym = [:hu, :hv][1:N]
    u_sym = [:u, :v][1:N]

    ∂x = [Differential(xᵢ) for xᵢ in x]
    ∇  = VectorOperator(∂x)
    ∂t = Differential(t)

    ω = ∇ ⋅ ⊥(𝐮) + f
    G = 𝐮 ⋅ 𝐮 / 2 + g * h

    #! format: off
    eqs = Dict(
        :h       => ∂t(h)   + ∇ ⋅ h𝐮,
        (hu_sym .=> ∂t.(h𝐮) + ∇ ⋅ (h𝐮 * 𝐮') + g * h * ∇(h + b) + f * ⊥(h𝐮) )...,
        (u_sym  .=> ∂t.(𝐮)  + ω * ⊥(𝐮) + ∇(G) )...,
        :b       => ∂t(b)
    )
    #! format: on

    make_source_modifier_from_syms(StateVars, eqs, grid)
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

    (Dx₊, Dx₋), (Dy₊, Dy₋) = fdop
    Dx_split = (Dx₊ - Dx₋) / 2
    Dy_split = (Dy₊ - Dy₋) / 2

    ∂xs(f) = ∂x2D(mempool, Dx_split, f)
    ∂ys(f) = ∂y2D(mempool, Dy_split, f)

    (ds, s, t) -> begin
        @unpack h, hu, hv, b = s

        λx = zero(T)
        λy = zero(T)
        @inbounds for i in eachindex(h)
            p = NaNMath.sqrt(g * h[i])
            λx = NaNMath.max(λx, abs(hu[i] / h[i]) + p)
            λy = NaNMath.max(λy, abs(hv[i] / h[i]) + p)
        end
        λx, λy = splitting.scaling .* (λx, λy)

        @. ds.h  += λx * $∂xs(h) + λy * $∂ys(h)
        @. ds.hu += λx * $∂xs(hu) + λy * $∂ys(hu)
        @. ds.hv += λx * $∂xs(hv) + λy * $∂ys(hv)
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

    (Dx₊, Dx₋), (Dy₊, Dy₋) = fdop
    Dx_split = (Dx₊ - Dx₋) / 2
    Dy_split = (Dy₊ - Dy₋) / 2

    ∂xs(f) = ∂x2D(mempool, Dx_split, f)
    ∂ys(f) = ∂y2D(mempool, Dy_split, f)

    u  = alloc()
    v  = alloc()
    gh = alloc()

    (ds, s, t) -> begin
        @unpack h, hu, hv, b = s

        @. u  = hu / h
        @. v  = hv / h
        @. gh = g * (h + b) - 1 // 2 * (u^2 + v^2)

        λh   = zero(T) # s
        λh𝐮₌ = zero(T) # m²s⁻¹
        λh𝐮₊ = zero(T) # m²s⁻¹
        @inbounds for i in eachindex(h)
            p = NaNMath.sqrt(g * h[i])
            c = hypot(u[i], v[i]) + p

            λh   = NaNMath.max(λh, h[i] / c)
            λh𝐮₌ = NaNMath.max(λh𝐮₌, h[i] * (c - 1 // 2 * p))
            λh𝐮₊ = NaNMath.max(λh𝐮₊, abs(hv[i] * hu[i]))
        end
        λh   = 2λh * splitting.scaling
        λh𝐮₌ = 4λh𝐮₌ * splitting.scaling
        λh𝐮₊ = 4√λh𝐮₊ * splitting.scaling

        @. ds.h  += λh * $∂xs(gh) + λh * $∂ys(gh)
        @. ds.hu += λh𝐮₌ * $∂xs(u) + λh𝐮₊ * $∂ys(u)
        @. ds.hv += λh𝐮₊ * $∂xs(v) + λh𝐮₌ * $∂ys(v)
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

    (Dx₊, Dx₋), (Dy₊, Dy₋) = fdop
    Dx_split = (Dx₊ - Dx₋) / 2
    Dy_split = (Dy₊ - Dy₋) / 2

    ∂xs(f) = ∂x2D(mempool, Dx_split, f)
    ∂ys(f) = ∂y2D(mempool, Dy_split, f)

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
        α, β = splitting.scaling .* (α, β)

        @. ds.h += α * ($∂xs(G) + $∂ys(G))
        @. ds.u += β * ($∂xs(hu) + $∂ys(hu))
        @. ds.v += β * ($∂xs(hv) + $∂ys(hv))
    end
end
