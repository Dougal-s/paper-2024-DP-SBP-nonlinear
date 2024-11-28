include("../pde-utils.jl")
include("compressible-euler-core.jl")

using NaNMath
using StaticArrays
using UnPack

function semidiscretise(info::CompEulerFluxForm2D{T}, s₀, Dx₊, Dx₋, Dy₊, Dy₋) where {T}
    @unpack γ, flux_splitting = info

    Dx = (Dx₋ + Dx₊) / 2
    Dx_split = (Dx₋ - Dx₊) / 2

    Dy = (Dy₋ + Dy₊) / 2
    Dy_split = (Dy₋ - Dy₊) / 2

    mempool = MemoryPool(T, size(s₀[1]))
    ∂y(m) = ∂y2D(mempool, Dy, m)
    ∂ys(m) = ∂y2D(mempool, Dy_split, m)
    ∂x(m) = ∂x2D(mempool, Dx, m)
    ∂xs(m) = ∂x2D(mempool, Dx_split, m)

    add_dissipation! = make_dissipation(info, flux_splitting, s₀, (∂xs, ∂ys))

    ϱ⁻¹ = similar(s₀[1])
    u = similar(s₀[1])
    v = similar(s₀[1])
    p = similar(s₀[1])

    # fluxes
    ϱuv = similar(s₀[1])
    fϱux = similar(s₀[1])
    fϱex = similar(s₀[1])
    fϱvy = similar(s₀[1])
    fϱey = similar(s₀[1])

    (ds, s, _, t) -> begin
        @unpack ϱ, ϱu, ϱv, ϱe = s
        returnblocks(mempool)

        @. ϱ⁻¹ = one(T) / ϱ
        @. u = ϱu * ϱ⁻¹
        @. v = ϱv * ϱ⁻¹
        @. ϱuv = ϱu * ϱv * ϱ⁻¹
        @. p = (γ - 1) * (ϱe - 1 // 2 * (ϱu * u + ϱv * v))

        @. fϱux = ϱu * u + p
        @. fϱex = u * (ϱe + p)

        @. fϱvy = ϱv * v + p
        @. fϱey = v * (ϱe + p)

        ds.ϱ .= @. -$∂x(ϱu) - $∂y(ϱv)
        ds.ϱu .= @. -$∂x(fϱux) - $∂y(ϱuv)
        ds.ϱv .= @. -$∂x(ϱuv) - $∂y(fϱvy)
        ds.ϱe .= @. -$∂x(fϱex) - $∂y(fϱey)

        add_dissipation!(ds, (s, ϱ⁻¹, u, v, p))
    end
end

function make_dissipation(eqs::CompEulerFluxForm2D{T}, ::FluxLaxFriedrichs, _, (∂xs, ∂ys)) where {T}
    @unpack γ = eqs
    (ds, (s, ϱ⁻¹, u, v, p)) -> begin
        @unpack ϱ, ϱu, ϱv, ϱe = s
        λx = zero(T)
        λy = zero(T)
        @inbounds for i in eachindex(ϱ)
            a = √max(γ * p[i] * ϱ⁻¹[i], zero(T))
            λx = max(λx, abs(u[i]) + a)
            λy = max(λy, abs(v[i]) + a)
        end

        ds.ϱ .+= @. -λx * $∂xs(ϱ) - λy * $∂ys(ϱ)
        ds.ϱu .+= @. -λx * $∂xs(ϱu) - λy * $∂ys(ϱu)
        ds.ϱv .+= @. -λx * $∂xs(ϱv) - λy * $∂ys(ϱv)
        ds.ϱe .+= @. -λx * $∂xs(ϱe) - λy * $∂ys(ϱe)

        nothing
    end
end

function semidiscretise(info::NordstromCompEuler2D{T}, s₀, Dx₊, Dx₋, Dy₊, Dy₋) where {T}
    @unpack γ, flux_splitting = info

    Dx = (Dx₋ + Dx₊) / 2
    Dx_split = (Dx₋ - Dx₊) / 2

    Dy = (Dy₋ + Dy₊) / 2
    Dy_split = (Dy₋ - Dy₊) / 2

    mempool = MemoryPool(T, size(s₀[1]))
    ∂y(m) = ∂y2D(mempool, Dy, m)
    ∂x(m) = ∂x2D(mempool, Dx, m)
    ∂ys(m) = ∂y2D(mempool, Dy_split, m)
    ∂xs(m) = ∂x2D(mempool, Dx_split, m)

    add_dissipation! = make_dissipation(info, flux_splitting, s₀, (∂xs, ∂ys))

    ϕ⁻¹ = similar(s₀[1])
    u = similar(s₀[1])
    v = similar(s₀[1])
    ϕuu = similar(s₀[1])
    ϕuv = similar(s₀[1])
    ϕvv = similar(s₀[1])
    qu = similar(s₀[1])
    qv = similar(s₀[1])

    ∂x_ϕu = similar(s₀[1])
    ∂y_ϕu = similar(s₀[1])
    ∂x_ϕv = similar(s₀[1])
    ∂y_ϕv = similar(s₀[1])
    ∂x_q = similar(s₀[1])
    ∂y_q = similar(s₀[1])

    (ds, s, _, t) -> begin
        @unpack ϕ, ϕu, ϕv, q = s
        returnblocks(mempool)

        @. ϕ⁻¹ = one(T) / ϕ

        @. u = ϕu * ϕ⁻¹
        @. v = ϕv * ϕ⁻¹

        @. ϕuu = ϕu * u
        @. ϕuv = ϕu * v
        @. ϕvv = ϕv * v

        @. qu = q * u
        @. qv = q * v

        ∂x2D!(∂x_ϕu, Dx, ϕu)
        ∂y2D!(∂y_ϕu, Dy, ϕu; cache=getblock(mempool))
        ∂x2D!(∂x_ϕv, Dx, ϕv)
        ∂y2D!(∂y_ϕv, Dy, ϕv; cache=getblock(mempool))
        ∂x2D!(∂x_q, Dx, q)
        ∂y2D!(∂y_q, Dy, q; cache=getblock(mempool))

        # Entropy: ϕα^2
        ds.ϕ .= @. -1 // 2 * (u * $∂x(ϕ) + ∂x_ϕu + v * $∂y(ϕ) + ∂y_ϕv)

        # Entropy: (γ-1)/2 * ϕu
        ds.ϕu .= @. -1 // 2 * (
            (u * ∂x_ϕu + $∂x(ϕuu)) + (v * ∂y_ϕu + $∂y(ϕuv)) +
            4q * ϕ⁻¹ * ∂x_q
        )

        # Entropy: (γ-1)/2 * ϕv
        ds.ϕv .= @. -1 // 2 * (
            (u * ∂x_ϕv + $∂x(ϕuv)) + (v * ∂y_ϕv + $∂y(ϕvv)) +
            4q * ϕ⁻¹ * ∂y_q
        )

        # Entropy: q
        ds.q .= @. -1 // 2 * (
            γ * ($∂x(qu) + $∂y(qv)) +
            (2 - γ) * (u * ∂x_q + v * ∂y_q)
        )

        add_dissipation!(ds, (s, ϕ⁻¹, u, v))
        nothing
    end
end

function make_dissipation(eqs::NordstromCompEuler2D{T}, splitting::FluxEntropyStable, _, (∂xs, ∂ys)) where {T}
    @unpack γ = eqs
    @unpack scaling = splitting

    sqrtγ = √(γ)
    (ds, (s, ϕ⁻¹, u, v)) -> begin
        @unpack ϕ, ϕu, ϕv, q = s

        λϕx = zero(T)
        λϕy = zero(T)
        λϕux = zero(T)
        λϕuy = zero(T)
        λϕvx = zero(T)
        λϕvy = zero(T)
        λqx = zero(T)
        λqy = zero(T)

        @inbounds for i in eachindex(u)
            ϕa = sqrtγ * abs(q[i])
            ϕeigvalx = abs(ϕu[i]) + ϕa
            ϕeigvaly = abs(ϕv[i]) + ϕa

            λϕx = max(λϕx, ϕeigvalx)
            λϕy = max(λϕy, ϕeigvaly)
            λϕux = max(λϕux, ϕ[i] * ϕeigvalx)
            λϕuy = max(λϕuy, ϕ[i] * ϕeigvaly)
            λϕvx = max(λϕvx, ϕ[i] * ϕeigvalx)
            λϕvy = max(λϕvy, ϕ[i] * ϕeigvaly)
            λqx = max(λqx, ϕ⁻¹[i] * ϕeigvalx)
            λqy = max(λqy, ϕ⁻¹[i] * ϕeigvaly)
        end

        λϕx *= 1 // 4 * scaling
        λϕy *= 1 // 4 * scaling
        λϕux *= 1 // 2 * scaling
        λϕuy *= 1 // 2 * scaling
        λϕvx *= 1 // 2 * scaling
        λϕvy *= 1 // 2 * scaling
        λqx *= 1 // 4 * scaling
        λqy *= 1 // 4 * scaling

        # Entropy: ϕα^2
        ds.ϕ .+= @. -(λϕx * $∂xs(ϕ) + λϕy * $∂ys(ϕ)) * ϕ⁻¹

        # Entropy: (γ-1)/2 * ϕu
        ds.ϕu .+= @. -(
            (λϕux * ϕ⁻¹ - λϕx) * $∂xs(u) +
            (λϕuy * ϕ⁻¹ - λϕy) * $∂ys(u) +
            (λϕx * $∂xs(ϕu) + λϕy * $∂ys(ϕu)) * ϕ⁻¹
        )

        # Entropy: (γ-1)/2 * ϕv
        ds.ϕv .+= @. -(
            (λϕvx * ϕ⁻¹ - λϕx) * $∂xs(v) +
            (λϕvy * ϕ⁻¹ - λϕy) * $∂ys(v) +
            (λϕx * $∂xs(ϕv) + λϕy * $∂ys(ϕv)) * ϕ⁻¹
        )

        # Entropy: q
        ds.q .+= @. -λqx * $∂xs(q) - λqy * $∂ys(q)

        nothing
    end
end

function semidiscretise(info::ReissSesterhennCompEuler2D{T}, s₀, Dx₊, Dx₋, Dy₊, Dy₋) where {T}
    @unpack γ, flux_splitting = info

    Dx = (Dx₋ + Dx₊) / 2
    Dx_split = (Dx₋ - Dx₊) / 2

    Dy = (Dy₋ + Dy₊) / 2
    Dy_split = (Dy₋ - Dy₊) / 2

    mempool = MemoryPool(T, size(s₀[1]))
    ∂y(m) = ∂y2D(mempool, Dy, m)
    ∂ys(m) = ∂y2D(mempool, Dy_split, m)
    ∂x(m) = ∂x2D(mempool, Dx, m)
    ∂xs(m) = ∂x2D(mempool, Dx_split, m)

    add_dissipation! = make_dissipation(info, flux_splitting, s₀, (∂xs, ∂ys))

    u = similar(s₀[1])
    v = similar(s₀[1])
    ϱu = similar(s₀[1])
    ϱv = similar(s₀[1])
    ϕ⁻¹ = similar(s₀[1])

    (ds, s, _, t) -> begin
        @unpack ϕ, ϕu, ϕv, p = s
        returnblocks(mempool)

        @. ϕ⁻¹ = $one(T) / ϕ
        @. u = ϕu * ϕ⁻¹
        @. v = ϕv * ϕ⁻¹
        @. ϱu = ϕu * ϕ
        @. ϱv = ϕv * ϕ

        # Entropy: none
        ds.ϕ .= @. -1 // 2 * ($∂x(ϱu) + $∂y(ϱv)) * ϕ⁻¹

        # Entropy: ϕu
        ds.ϕu .= @. -1 // 2 * (
            ($∂x(ϕu^2 + 2p) + $∂y(ϕu * ϕv)) * ϕ⁻¹ +
            ϕu * $∂x(u) + ϕv * $∂y(u)
        )

        # Entropy: ϕv
        ds.ϕv .= @. -1 // 2 * (
            ($∂y(ϕv^2 + 2p) + $∂x(ϕu * ϕv)) * ϕ⁻¹ +
            ϕu * $∂x(v) + ϕv * $∂y(v)
        )

        # Entropy: 1 / (γ - 1)
        ds.p .= @. (γ - 1) * (u * $∂x(p) + v * $∂y(p)) - γ * ($∂x(p * u) + $∂y(p * v))

        add_dissipation!(ds, (s, ϕ⁻¹, u, v))
        nothing
    end
end

function semidiscretise(info::VanLeerHanelCompEuler2D{T}, s₀, Dx₊, Dx₋, Dy₊, Dy₋) where {T}
    @unpack γ = info

    mempool = MemoryPool(T, size(s₀[1]))
    ∂y(p, m) = @. $∂y2D(mempool, Dy₋, p) += $∂y2D(mempool, Dy₊, m)
    ∂x(p, m) = @. $∂x2D(mempool, Dx₋, p) += $∂x2D(mempool, Dx₊, m)

    u = similar(s₀[1])
    v = similar(s₀[1])
    p = similar(s₀[1])
    a = similar(s₀[1])
    H = similar(s₀[1])
    M = similar(s₀[1])
    p₊ = similar(s₀[1])
    p₋ = similar(s₀[1])

    fxp = @SVector [similar(s₀[1]), similar(s₀[1]), similar(s₀[1]), similar(s₀[1])]
    fxm = @SVector [similar(s₀[1]), similar(s₀[1]), similar(s₀[1]), similar(s₀[1])]
    fyp = @SVector [similar(s₀[1]), similar(s₀[1]), similar(s₀[1]), similar(s₀[1])]
    fym = @SVector [similar(s₀[1]), similar(s₀[1]), similar(s₀[1]), similar(s₀[1])]

    (ds, s, _, t) -> begin
        @unpack ϱ, ϱu, ϱv, ϱe = s

        returnblocks(mempool)
        @. u = ϱu / ϱ
        @. v = ϱv / ϱ
        @. p = (γ - 1) * (ϱe - 1 / 2 * (ϱu^2 + ϱv^2) / ϱ)
        @. a = NaNMath.sqrt(γ * p / ϱ)
        @. H = (ϱe + p) / ϱ

        # ∂x
        @. M = u / a

        @. p₊ = 0.5 * (1 + γ * M) * p
        @. p₋ = 0.5 * (1 - γ * M) * p

        @. fxp[1] = 0.25 * ϱ * a * (M + 1)^2
        @. fxp[2] = fxp[1] * u + p₊
        @. fxp[3] = fxp[1] * v
        @. fxp[4] = fxp[1] * H

        @. fxm[1] = -0.25 * ϱ * a * (M - 1)^2
        @. fxm[2] = fxm[1] * u + p₋
        @. fxm[3] = fxm[1] * v
        @. fxm[4] = fxm[1] * H

        @. M = v / a

        @. p₊ = 0.5 * (1 + γ * M) * p
        @. p₋ = 0.5 * (1 - γ * M) * p

        @. fyp[1] = 0.25 * ϱ * a * (M + 1)^2
        @. fyp[2] = fyp[1] * u
        @. fyp[3] = fyp[1] * v + p₊
        @. fyp[4] = fyp[1] * H

        @. fym[1] = -0.25 * ϱ * a * (M - 1)^2
        @. fym[2] = fym[1] * u
        @. fym[3] = fym[1] * v + p₋
        @. fym[4] = fym[1] * H

        ds.ϱ .= @. -$∂x(fxp[1], fxm[1]) - $∂y(fyp[1], fym[1])
        ds.ϱu .= @. -$∂x(fxp[2], fxm[2]) - $∂y(fyp[2], fym[2])
        ds.ϱv .= @. -$∂x(fxp[3], fxm[3]) - $∂y(fyp[3], fym[3])
        ds.ϱe .= @. -$∂x(fxp[4], fxm[4]) - $∂y(fyp[4], fym[4])
        nothing
    end
end
