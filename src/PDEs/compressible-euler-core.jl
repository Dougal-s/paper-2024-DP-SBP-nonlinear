using HyperbolicPDEs: NullOperator
using LinearAlgebra: ⋅
using RecursiveArrayTools: NamedArrayPartition
using StaticArrays
using NaNMath
using MuladdMacro: @muladd
using UnPack: @unpack

abstract type CompressibleEulerScheme{N, T <: Real, StateVars} end;

function thermodynamic_entropy(γ, (ϱ, v..., p))
    return (p > 0 && ϱ > 0) ?
           -ϱ * muladd(-γ, log(ϱ), log(p)) :
           oftype(ϱ, NaN64)
end

function harten_entropy((γ, α), (ϱ, v..., p))
    return (p > 0 && ϱ > 0) ?
           -(γ + α) / (γ - 1) * (p * ϱ^α)^(1 / (γ + α)) :
           oftype(ϱ, NaN64)
end

function to_primitive_vars(eqs::CompressibleEulerScheme{N, T}, state) where {N, T}
    sz = size(getproperty(state, propertynames(state)[begin]))
    dst = ntuple(_ -> Array{T}(undef, sz), Val(2 + N))
    to_primitive_vars!(dst, eqs, state)
    return dst
end

@kwdef struct FluxLaxFriedrichs{T <: Real}
    volume::Bool  = true
    surface::Bool = true
    scaling::T    = 1.0
end

"""
Meng-Sing Liou and Chris J. Steffen Jr.
High-Order Polynomial Expansions (HOPE) for flux-vector splitting.
https://ntrs.nasa.gov/citations/19910016425
"""
@kwdef struct FluxVanLeerHanel{T <: Real}
    volume::Bool  = true
    surface::Bool = true
    scaling::T    = 1.0
end

@kwdef struct FluxEntropyStable{T <: Real}
    volume::Bool  = true
    surface::Bool = true
    scaling::T    = 1.0
end

"""
The "entropy" stable upwinding used in the old 2024 version
https://arxiv.org/abs/2411.06629v2
"""
@kwdef struct FluxDSL2024{T <: Real}
    volume::Bool  = true
    surface::Bool = true
    scaling::T    = 1.0
end

"""
The entropy stable upwinding used in the revised version
"""
@kwdef struct FluxDSL2025{T <: Real}
    volume::Bool  = true
    surface::Bool = true
    scaling::T    = 1.0
end

struct SourceMMS{T}
    exact::T
end

#########################
# 1D Compressible Euler #
#########################

"""
"""
@kwdef struct CompEulerFluxForm1D{T <: Real, Modifiers} <:
              CompressibleEulerScheme{1, T, (:ϱ, :ϱu, :ϱe)}
    γ::T = 1.4

    modifiers::Modifiers = ()
end

function CompEulerFluxForm1D(mod1, mods...; kwargs...)
    CompEulerFluxForm1D(; modifiers = (mod1, mods...), kwargs...)
end

function from_primitive_vars(
        info::CompressibleEulerScheme{1, T, (:ϱ, :ϱu, :ϱe)}, (ϱ, u, p)) where {T}
    @unpack γ = info
    ϱe = @. p / (γ - 1) + 1 // 2 * ϱ * u^2
    return NamedArrayPartition(ϱ = ϱ, ϱu = ϱ .* u, ϱe = ϱe)
end

function to_primitive_vars!(
        dst, info::CompressibleEulerScheme{1, T, (:ϱ, :ϱu, :ϱe)}, state) where {T}
    @unpack γ = info
    @unpack ϱ, ϱu, ϱe = state
    dst[1] .= @. ϱ
    dst[2] .= @. ϱu / ϱ
    dst[3] .= @. (γ - 1) * (ϱe - 1 // 2 * ϱu^2 / ϱ)
    return ()
end

function semidiscretise(info::CompEulerFluxForm1D{T}, grid, fdop;
        alloc = () -> Array{T}(undef, length(grid))
) where {T}
    @unpack γ = info

    D = (@unpack D, B = fdop[1]; D + B)
    ∂x(m) = D * m

    apply_modifier!s = map(
        mod -> make_modifier(info, mod, grid, fdop, alloc), info.modifiers)

    u  = alloc()
    p  = alloc()
    f₂ = alloc()
    f₃ = alloc()

    (ds, s, _, t) -> begin
        @unpack ϱ, ϱu, ϱe = s

        @. u = ϱu / ϱ
        @. p = (γ - 1) * (ϱe - ϱu * u / 2)
        @. f₂ = ϱu * u + p
        @. f₃ = u * (ϱe + p)

        @. ds.ϱ  = -$∂x(ϱu)
        @. ds.ϱu = -$∂x(f₂)
        @. ds.ϱe = -$∂x(f₃)

        for apply_mod! in apply_modifier!s
            apply_mod!(ds, s, t, (; u = u, p = p))
        end

        return nothing
    end
end

"""
"""
@kwdef struct ReissSesterhennCompEuler1D{T <: Real, Modifiers} <:
              CompressibleEulerScheme{1, T, (:ϕ, :ϕu, :p)}
    γ::T = 1.4

    modifiers::Modifiers = ()
end

function ReissSesterhennCompEuler1D(mod1, mods...; kwargs...)
    ReissSesterhennCompEuler1D(; modifiers = (mod1, mods...), kwargs...)
end

function from_primitive_vars(::ReissSesterhennCompEuler1D, (ϱ, u, p))
    ϕ = .√ϱ
    return NamedArrayPartition(ϕ = ϕ, ϕu = ϕ .* u, p = p)
end

function to_primitive_vars!(dst, ::ReissSesterhennCompEuler1D, state)
    @unpack ϕ, ϕu, p = state
    dst[1] .= @. ϕ^2
    dst[2] .= @. ϕu / ϕ
    dst[3] .= @. p
    return ()
end

function semidiscretise(info::ReissSesterhennCompEuler1D{T}, grid, fdop;
        alloc = () -> Array{T}(undef, length(grid))
) where {T}
    @unpack γ = info

    D = (@unpack D, B = fdop[1]; D + B)
    ∂x(m) = D * m

    apply_modifier!s = map(
        mod -> make_modifier(info, mod, grid, fdop, alloc), info.modifiers)

    ϕ⁻¹ = alloc()
    u   = alloc()
    ϱu  = alloc()

    (ds, s, _, t) -> begin
        @unpack ϕ, ϕu, p = s

        @. ϕ⁻¹ = 1 / ϕ
        @. u   = ϕu * ϕ⁻¹
        @. ϱu  = ϕu * ϕ

        @. ds.ϕ  = -1 // 2 * $∂x(ϱu) * ϕ⁻¹
        @. ds.ϕu = -1 // 2 * ($∂x(ϕu^2 + 2p) * ϕ⁻¹ + ϕu * $∂x(u))
        @. ds.p  = -γ * $∂x(u * p) + (γ - 1) * u * $∂x(p)

        for apply_mod! in apply_modifier!s
            apply_mod!(ds, s, t, (; ϕ⁻¹ = ϕ⁻¹, u = u, ϱu = ϱu))
        end
    end
end

"""

"""
@kwdef struct NordstromCompEuler1D{T <: Real, Modifiers} <:
              CompressibleEulerScheme{1, T, (:ϕ, :ϕu, :q)}
    γ::T = 1.4

    modifiers::Modifiers = ()
end

function NordstromCompEuler1D(mod1, mods...; kwargs...)
    NordstromCompEuler1D(; modifiers = (mod1, mods...), kwargs...)
end

function from_primitive_vars(::NordstromCompEuler1D, (ϱ, u, p))
    ϕ = .√ϱ
    return NamedArrayPartition(ϕ = ϕ, ϕu = ϕ .* u, q = .√p)
end

function to_primitive_vars!(dst, ::NordstromCompEuler1D, state)
    @unpack ϕ, ϕu, q = state
    dst[1] .= @. ϕ^2
    dst[2] .= @. ϕu / ϕ
    dst[3] .= @. q^2
    return ()
end

@muladd function semidiscretise(info::NordstromCompEuler1D{T}, grid, fdop;
        alloc = () -> Array{T}(undef, length(grid))
) where {T}
    @unpack γ = info

    D = (@unpack D, B = fdop[1]; D + B)
    ∂x!(dst, m) = mul!(dst, D, m)

    apply_modifier!s = map(
        mod -> make_modifier(info, mod, grid, fdop, alloc), info.modifiers)

    u   = alloc()
    ϕ⁻¹ = alloc()
    ϕuu = alloc()
    qu  = alloc()

    ∂x_ϕ   = alloc()
    ∂x_ϕu  = alloc()
    ∂x_q   = alloc()
    ∂x_qu  = alloc()
    ∂x_ϕuu = alloc()

    (ds, s, _, t) -> begin
        @unpack ϕ, ϕu, q = s

        @inbounds @simd ivdep for i in eachindex(ϕ)
            ϕᵢ, ϕuᵢ, qᵢ = ϕ[i], ϕu[i], q[i]
            ϕ⁻¹ᵢ = inv(ϕᵢ)
            uᵢ   = ϕuᵢ * ϕ⁻¹ᵢ

            ϕ⁻¹[i], u[i] = ϕ⁻¹ᵢ, uᵢ
            ϕuu[i] = ϕuᵢ * uᵢ
            qu[i]  = qᵢ * uᵢ
        end

        ∂x!(∂x_q, q)
        ∂x!(∂x_qu, qu)
        ∂x!(∂x_ϕ, ϕ)
        ∂x!(∂x_ϕu, ϕu)
        ∂x!(∂x_ϕuu, ϕuu)

        @inbounds @simd ivdep for i in eachindex(ϕ)
            ϕ⁻¹ᵢ, qᵢ, uᵢ = ϕ⁻¹[i], q[i], u[i]
            ∂x_qᵢ, ∂x_quᵢ, ∂x_ϕᵢ, ∂x_ϕuᵢ, ∂x_ϕuuᵢ =
                ∂x_q[i], ∂x_qu[i], ∂x_ϕ[i], ∂x_ϕu[i], ∂x_ϕuu[i]

            ds.ϕ[i]  = -1 // 2 * (∂x_ϕuᵢ + uᵢ * ∂x_ϕᵢ)
            ds.ϕu[i] = -1 // 2 * (∂x_ϕuuᵢ + uᵢ * ∂x_ϕuᵢ + 4qᵢ * ϕ⁻¹ᵢ * ∂x_qᵢ)
            ds.q[i]  = -1 // 2 * (γ * ∂x_quᵢ + (2 - γ) * uᵢ * ∂x_qᵢ)
        end

        for apply_mod! in apply_modifier!s
            apply_mod!(ds, s, t, (; ϕ⁻¹ = ϕ⁻¹, u = u))
        end
    end
end

###################
# Flux Splittings #
###################

@muladd function make_modifier(
        eqs::NordstromCompEuler1D{T},
        splitting::FluxDSL2025,
        grid,
        fdop,
        alloc
) where {T}
    @unpack γ = eqs

    Diᵥ = splitting.volume ? splitting.scaling * fdop[1].Diᵥ : NullOperator()
    Diₛ = splitting.surface ? splitting.scaling * fdop[1].Diₛ : NullOperator()

    sqrtγ = √γ

    w_ϱ   = alloc()
    w_ϱu  = alloc()
    w_ϱe  = alloc()
    ∂s_ϱ  = alloc()
    ∂s_ϱu = alloc()
    ∂s_ϱe = alloc()
    λ_ϱ   = alloc()
    λ_ϱu  = alloc()
    λ_ϱe  = alloc()

    tmp = alloc()

    (∂ₜstate, state, t, diagnostic) -> begin
        @unpack ϕ, ϕu, q = state
        @unpack ϕ⁻¹, u = diagnostic

        @inbounds @simd ivdep for i in eachindex(ϕ)
            ϕᵢ, ϕ⁻¹ᵢ, ϕuᵢ, uᵢ, qᵢ = ϕ[i], ϕ⁻¹[i], ϕu[i], u[i], q[i]
            p⁻¹ = qᵢ^-2
            p   = qᵢ^2
            ϱ   = ϕᵢ^2
            K   = ϕuᵢ^2 / 2
            s   = 2(NaNMath.log(qᵢ) - γ * NaNMath.log(ϕᵢ))

            M² = 1/γ * ϕuᵢ^2 * p⁻¹ * ϕ⁻¹ᵢ^2
            a  = sqrtγ * abs(qᵢ * ϕ⁻¹ᵢ)
            λ  = abs(uᵢ) + a

            s_ϱ = (γ - 1) * ϱ * p^2 / ((γ - 1)^2 * K^2 + γ * p^2)
            s_ϱu = p^2 / (p + (γ - 1) * ϕuᵢ^2)
            s_ϱe = p^2 / ((γ - 1) * ϱ)

            w_ϱ[i]  = (γ - s) / (γ - 1) - K * p⁻¹
            w_ϱu[i] = ϕᵢ * ϕuᵢ * p⁻¹
            w_ϱe[i] = -ϕᵢ^2 * p⁻¹

            λ_ϱ[i]  = s_ϱ * λ
            λ_ϱu[i] = s_ϱu * λ
            λ_ϱe[i] = s_ϱe * λ * 2M² / (1 + M²)
        end

        for (∂s, w, λs) in (
            (∂s_ϱ, w_ϱ, λ_ϱ),
            (∂s_ϱu, w_ϱu, λ_ϱu),
            (∂s_ϱe, w_ϱe, λ_ϱe)
        )
            fill!(∂s, false)
            add_splitting!(∂s, Val(1), grid, Diᵥ, Diₛ, λs, w, tmp)
        end

        @inbounds for i in eachindex(ϕ)
            ∂s_ϕ  = ∂s_ϱ[i] * ϕ⁻¹[i] / 2
            ∂s_ϕu = (∂s_ϱu[i] - ϕu[i] * ∂s_ϕ) * ϕ⁻¹[i]
            ∂s_p  = (γ - 1) * (∂s_ϱe[i] - ϕu[i] * ∂s_ϕu)

            ∂ₜstate.ϕ[i]  += ∂s_ϕ
            ∂ₜstate.ϕu[i] += ∂s_ϕu
            ∂ₜstate.q[i]  += ∂s_p / (2q[i])
        end
    end
end

function make_modifier(
        eqs::CompEulerFluxForm1D,
        splitting::FluxLaxFriedrichs,
        grid,
        fdop,
        alloc
)
    @unpack γ = eqs

    Diᵥ = splitting.volume ? splitting.scaling * fdop[1].Diᵥ : NullOperator()
    Diₛ = splitting.surface ? splitting.scaling * fdop[1].Diₛ : NullOperator()

    λs = alloc()
    tmp = alloc()

    (ds, s, t, diagnostic) -> begin
        @unpack ϱ, ϱu, ϱe = s

        @inbounds for i in eachindex(ϱ)
            ϱp = (γ - 1) * (ϱ[i] * ϱe[i] - 1 // 2 * ϱu[i] * ϱu[i])
            ϱa = NaNMath.sqrt(γ * ϱp)
            λs[i] = (abs(ϱu[i]) + ϱa) / ϱ[i]
        end

        add_splitting!(ds.ϱ, Val(1), grid, Diᵥ, Diₛ, λs, ϱ, tmp)
        add_splitting!(ds.ϱu, Val(1), grid, Diᵥ, Diₛ, λs, ϱu, tmp)
        add_splitting!(ds.ϱe, Val(1), grid, Diᵥ, Diₛ, λs, ϱe, tmp)
    end
end

function make_modifier(
        eqs::NordstromCompEuler1D,
        splitting::FluxDSL2024,
        grid,
        fdop,
        allocator
)
    @unpack γ = eqs

    Dₛ = splitting.scaling * ((splitting.volume ? fdop[1].Diᵥ : NullOperator()) +
                              (splitting.surface ? fdop[1].Diₛ : NullOperator()))
    ∂xs(f) = Dₛ * f

    sqrtγ = √γ

    (ds, s, t, diagnostic) -> begin
        @unpack ϕ, ϕu, q = s
        @unpack ϕ⁻¹, u = diagnostic

        λϕ = 1 // 4 * maximum(zip(ϕu, q)) do (ϕu, q)
            abs(ϕu) + sqrtγ * abs(q)
        end
        λϕu = 1 // 2 * maximum(zip(ϕ, ϕu, q)) do (ϕ, ϕu, q)
            ϕ * (abs(ϕu) + sqrtγ * abs(q))
        end
        λq = 1 // 4 * maximum(zip(ϕ⁻¹, u, q)) do (ϕ⁻¹, u, q)
            abs(u) + sqrtγ * abs(q * ϕ⁻¹)
        end

        @. ds.ϕ  += λϕ * $∂xs(ϕ) * ϕ⁻¹
        @. ds.ϕu += (λϕu * ϕ⁻¹ - λϕ) * $∂xs(u) + λϕ * $∂xs(ϕu) * ϕ⁻¹
        @. ds.q  += λq * $∂xs(q)
    end
end

function make_modifier(
        eqs::CompEulerFluxForm1D,
        splitting::FluxVanLeerHanel{T},
        grid,
        fdop,
        alloc
) where {T}
    @unpack γ = eqs

    Dₛ = splitting.scaling * ((splitting.volume ? fdop[1].Diᵥ : NullOperator()) +
                              (splitting.surface ? fdop[1].Diₛ : NullOperator()))
    ∂xs(f) = Dₛ * f

    pₛ = alloc()
    a  = alloc()
    M  = alloc()
    fₛ = ntuple(_ -> alloc(), Val(3))

    (ds, s, t, diagnostic) -> begin
        @unpack ϱ, ϱu, ϱe = s
        @unpack p, u = diagnostic

        @. a     = NaNMath.sqrt(γ * p / ϱ)
        @. M     = u / a
        @. pₛ    = γ * M * p
        @. fₛ[1] = ϱ * a * (M^2 + 1) / 2
        @. fₛ[2] = fₛ[1] * u + pₛ
        @. fₛ[3] = fₛ[1] * (ϱe + p) / ϱ

        @. ds.ϱ  += $∂xs(fₛ[1])
        @. ds.ϱu += $∂xs(fₛ[2])
        @. ds.ϱe += $∂xs(fₛ[3])
    end
end

#######
# MMS #
#######

function make_modifier(
        info::CompressibleEulerScheme{N, T, StateVars},
        mms::SourceMMS,
        grid,
        _...
) where {N, T, StateVars}
    @unpack γ = info

    ∂t = SymUtils.∂t
    ∇  = ntuple(i -> SymUtils.SpatialDerivative{i}(), Val(N))
    ∀  = Base.Fix{2}(ntuple, Val(N))

    ϱ, v⃗..., p = SymUtils.LazyField.(mms.exact)

    ϱv⃗ = ϱ .* v⃗
    ϱe = p / (γ - 1) + ϱ * (v⃗ ⋅ v⃗) / 2

    ϱu_sym = (:ϱu, :ϱv, :ϱw)[1:N]
    ϕu_sym = (:ϕu, :ϕv, :ϕw)[1:N]

    eqs_ϱ = ∂t(ϱ) + ∇ ⋅ ϱv⃗
    eqs_p = ∂t(p) + v⃗ ⋅ (∇ .* p) + γ * p * (∇ ⋅ v⃗)
    eqs_ϱv⃗ = ∀(i -> ∂t(ϱv⃗[i]) + ∇ ⋅ (ϱv⃗[i] .* v⃗) + ∇[i]p)

    eqs = (;
        ϱe = ∂t(ϱe) + ∇ ⋅ ((ϱe + p) .* v⃗),
        ϱ = eqs_ϱ,
        ϕ = eqs_ϱ / (2√ϱ),
        p = eqs_p,
        q = eqs_p / (2√p),
        (@. ϱu_sym => eqs_ϱv⃗)...,
        (@. ϕu_sym => (eqs_ϱv⃗ - v⃗ * eqs_ϱ / 2) / √ϱ)...
    )

    SymUtils.make_source_modifier_from_syms(StateVars, eqs, grid)
end

#########################
# 2D Compressible Euler #
#########################

"""
"""
@kwdef struct CompEulerFluxForm2D{T <: Real, Modifiers} <:
              CompressibleEulerScheme{2, T, (:ϱ, :ϱu, :ϱv, :ϱe)}
    γ::T = 1.4

    modifiers::Modifiers = ()
end

function CompEulerFluxForm2D(mod1, mods...; kwargs...)
    CompEulerFluxForm2D(; modifiers = (mod1, mods...), kwargs...)
end

function from_primitive_vars(info::CompEulerFluxForm2D, (ϱ, u, v, p))
    ϱe = @. p / (info.γ - 1) + 1 / 2 * ϱ * (u^2 + v^2)
    return NamedArrayPartition(ϱ = ϱ, ϱu = ϱ .* u, ϱv = ϱ .* v, ϱe = ϱe)
end

function to_primitive_vars!(dst, info::CompEulerFluxForm2D, state)
    @unpack ϱ, ϱu, ϱv, ϱe = state
    dst[1] .= ϱ
    dst[2] .= @. ϱu / ϱ
    dst[3] .= @. ϱv / ϱ
    dst[4] .= @. (info.γ - 1) * (ϱe - 1 / 2 * (ϱu^2 + ϱv^2) / ϱ)
    return ()
end

"""
Jan Nordström
A skew-symmetric energy and entropy stable formulation of the compressible
Euler equations
https://arxiv.org/pdf/2201.05423.pdf
"""
@kwdef struct NordstromCompEuler2D{T <: Real, Modifiers} <:
              CompressibleEulerScheme{2, T, (:ϕ, :ϕu, :ϕv, :q)}
    γ::T = 1.4

    modifiers::Modifiers = ()
end

function NordstromCompEuler2D(mod1, mods...; kwargs...)
    NordstromCompEuler2D(; modifiers = (mod1, mods...), kwargs...)
end

function from_primitive_vars(::NordstromCompEuler2D, (ϱ, u, v, p))
    ϕ = .√ϱ
    return NamedArrayPartition(ϕ = ϕ, ϕu = ϕ .* u, ϕv = ϕ .* v, q = .√p)
end

function to_primitive_vars!(dst, ::NordstromCompEuler2D, state)
    @unpack ϕ, ϕu, ϕv, q = state
    dst[1] .= @. ϕ^2
    dst[2] .= @. ϕu / ϕ
    dst[3] .= @. ϕv / ϕ
    dst[4] .= @. q^2
    return ()
end

"""
Julius Reiss, Jörn Sesterhenn
A conservative, skew-symmetric Finite Difference Scheme for the compressible Navier--Stokes Equations
https://arxiv.org/abs/1308.6672
"""
@kwdef struct ReissSesterhennCompEuler2D{T <: Real, Modifiers} <:
              CompressibleEulerScheme{2, T, (:ϕ, :ϕu, :ϕv, :p)}
    γ::T = 1.4

    modifiers::Modifiers = ()
end

function ReissSesterhennCompEuler2D(mod1, mods...; kwargs...)
    ReissSesterhennCompEuler2D(; modifiers = (mod1, mods...), kwargs...)
end

function from_primitive_vars(::ReissSesterhennCompEuler2D, (ϱ, u, v, p))
    ϕ = .√ϱ
    return NamedArrayPartition(ϕ = ϕ, ϕu = ϕ .* u, ϕv = ϕ .* v, p = p)
end

function to_primitive_vars!(dst, ::ReissSesterhennCompEuler2D, state)
    @unpack ϕ, ϕu, ϕv, p = state
    dst[1] .= @. ϕ^2
    dst[2] .= @. ϕu / ϕ
    dst[3] .= @. ϕv / ϕ
    dst[4] .= @. p
    return ()
end
