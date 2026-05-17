include("compressible-euler-core.jl")

using NaNMath
using UnPack
using StaticArrays
using LinearAlgebra
using MuladdMacro: @muladd

@muladd begin
    function semidiscretise(info::CompEulerFluxForm2D{T}, grid, fdop;
            alloc = () -> Array{T}(undef, size(grid))
    ) where {T}
        @unpack γ = info

        mempool = MemoryPool(alloc())
        permute_cache = alloc()

        Dx = (@unpack D, B = fdop[1]; D + B)
        ∂x(m) = axis_mul!(mempool, Val(1), Dx, m, permute_cache)

        Dy = (@unpack D, B = fdop[2]; D + B)
        ∂y(m) = axis_mul!(mempool, Val(2), Dy, m, permute_cache)

        apply_modifier!s = map(
            mod -> make_modifier(info, mod, grid, fdop, alloc, mempool), info.modifiers)

        ϱ⁻¹ = alloc()
        u   = alloc()
        v   = alloc()
        p   = alloc()

        # fluxes
        ϱuv  = alloc()
        fϱux = alloc()
        fϱex = alloc()
        fϱvy = alloc()
        fϱey = alloc()

        (ds, s, _, t) -> begin
            @unpack ϱ, ϱu, ϱv, ϱe = s
            returnblocks(mempool)

            @inbounds @simd ivdep for i in eachindex(ϱ)
                ϱᵢ, ϱv⃗ᵢ, ϱeᵢ = ϱ[i], @SArray[ϱu[i], ϱv[i]], ϱe[i]

                ϱ⁻¹ᵢ = inv(ϱᵢ)
                v⃗ᵢ = ϱv⃗ᵢ * ϱ⁻¹ᵢ
                ϱv⃗v⃗ᵢ = ϱv⃗ᵢ * v⃗ᵢ'
                pᵢ = (γ - 1) * (ϱeᵢ - 1 // 2 * tr(ϱv⃗v⃗ᵢ))

                # fluxes
                fϱux[i], ϱuv[i], _, fϱvy[i] = ϱv⃗v⃗ᵢ + pᵢ * I
                fϱex[i], fϱey[i] = v⃗ᵢ * (ϱeᵢ + pᵢ)

                # intermediate qtys
                ϱ⁻¹[i], u[i], v[i], p[i] = ϱ⁻¹ᵢ, v⃗ᵢ..., pᵢ
            end

            ds.ϱ  .= @. -$∂x(ϱu) - $∂y(ϱv)
            ds.ϱu .= @. -$∂x(fϱux) - $∂y(ϱuv)
            ds.ϱv .= @. -$∂x(ϱuv) - $∂y(fϱvy)
            ds.ϱe .= @. -$∂x(fϱex) - $∂y(fϱey)

            for apply_mod! in apply_modifier!s
                apply_mod!(ds, s, t, (; ϱ⁻¹ = ϱ⁻¹, u = u, v = v, p = p))
            end
        end
    end

    function make_modifier(
            eqs::CompEulerFluxForm2D{T},
            splitting::FluxLaxFriedrichs,
            grid,
            fdop,
            alloc,
            mempool
    ) where {T}
        @unpack γ = eqs

        Dixᵥ = (splitting.volume ? splitting.scaling * fdop[1].Diᵥ : NullOperator())
        Dixₛ = (splitting.surface ? splitting.scaling * fdop[1].Diₛ : NullOperator())
        Diyᵥ = (splitting.volume ? splitting.scaling * fdop[2].Diᵥ : NullOperator())
        Diyₛ = (splitting.surface ? splitting.scaling * fdop[2].Diₛ : NullOperator())

        permute_cache = alloc()
        tmp = alloc()

        λs = ntuple(_ -> alloc(), 2)

        (ds, s, t, diagnostic) -> begin
            @unpack ϱ, ϱu, ϱv, ϱe = s
            @unpack ϱ⁻¹, u, v, p = diagnostic

            @inbounds for i in eachindex(ϱ)
                a = NaNMath.sqrt(γ * p[i] * ϱ⁻¹[i])
                λs[1][i] = abs(u[i]) + a
                λs[2][i] = abs(v[i]) + a
            end

            for (dqdt, q, λs) in (
                (ds.ϱ, ϱ, λs),
                (ds.ϱu, ϱu, λs),
                (ds.ϱv, ϱv, λs),
                (ds.ϱe, ϱe, λs),
            )
                add_splitting!(dqdt, Val(1), grid, Dixᵥ, Dixₛ, λs[1], q, tmp, permute_cache)
                add_splitting!(dqdt, Val(2), grid, Diyᵥ, Diyₛ, λs[2], q, tmp, permute_cache)
            end
        end
    end

    function make_modifier(
            eqs::CompEulerFluxForm2D{T},
            splitting::FluxVanLeerHanel,
            grid,
            fdop,
            alloc,
            mempool
    ) where {T}
        @unpack γ = eqs

        permute_cache = alloc()
        Dxₛ = splitting.scaling * ((splitting.volume ? fdop[1].Diᵥ : NullOperator()) +
                                   (splitting.surface ? fdop[1].Diₛ : NullOperator()))
        ∂xs(m) = axis_mul!(mempool, Val(1), Dxₛ, m, permute_cache)

        Dyₛ = splitting.scaling * ((splitting.volume ? fdop[2].Diᵥ : NullOperator()) +
                                   (splitting.surface ? fdop[2].Diₛ : NullOperator()))
        ∂ys(m) = axis_mul!(mempool, Val(2), Dyₛ, m, permute_cache)

        pₛ = alloc()
        a  = alloc()
        M  = alloc()
        fₛ = ntuple(_ -> alloc(), Val(4))

        (ds, s, t, diagnostic) -> begin
            @unpack ϱ, ϱu, ϱv, ϱe = s
            @unpack ϱ⁻¹, u, v, p = diagnostic

            @. a = NaNMath.sqrt(γ * p * ϱ⁻¹)

            @. M     = u / a
            @. pₛ    = γ * M * p
            @. fₛ[1] = ϱ * a * (M^2 + 1) / 2
            @. fₛ[2] = fₛ[1] * u + pₛ
            @. fₛ[3] = fₛ[1] * v
            @. fₛ[4] = fₛ[1] * (ϱe + p) * ϱ⁻¹

            @. ds.ϱ  += $∂xs(fₛ[1])
            @. ds.ϱu += $∂xs(fₛ[2])
            @. ds.ϱv += $∂xs(fₛ[3])
            @. ds.ϱe += $∂xs(fₛ[4])

            @. M     = v / a
            @. pₛ    = γ * M * p
            @. fₛ[1] = ϱ * a * (M^2 + 1) / 2
            @. fₛ[2] = fₛ[1] * u
            @. fₛ[3] = fₛ[1] * v + pₛ
            @. fₛ[4] = fₛ[1] * (ϱe + p) * ϱ⁻¹

            @. ds.ϱ  += $∂ys(fₛ[1])
            @. ds.ϱu += $∂ys(fₛ[2])
            @. ds.ϱv += $∂ys(fₛ[3])
            @. ds.ϱe += $∂ys(fₛ[4])

            nothing
        end
    end

    function semidiscretise(info::NordstromCompEuler2D{T}, grid, fdop;
            alloc = () -> Array{T}(undef, size(grid))
    ) where {T}
        @unpack γ = info

        dims = 2
        allocn(n) = ntuple(_ -> alloc(), n)

        mempool = MemoryPool(alloc())
        permute_cache = alloc()

        Dx = (@unpack D, B = fdop[1]; D + B)
        ∂x!(dst, m) = axis_mul!(dst, Val(1), Dx, m, permute_cache)

        Dy = (@unpack D, B = fdop[2]; D + B)
        ∂y!(dst, m) = axis_mul!(dst, Val(2), Dy, m, permute_cache)

        tmp = alloc()
        function gradient!(x, y, src)
            ∂x!(x, src)
            ∂y!(y, src)
        end
        function divergence!(dst, x, y)
            ∂x!(dst, x)
            ∂y!(tmp, y)
            @. dst += tmp
        end

        apply_modifier!s = map(
            mod -> make_modifier(info, mod, grid, fdop, alloc, mempool), info.modifiers)

        ϕ⁻¹           = alloc()
        u, v          = allocn(Val(dims))
        ϕuu, ϕuv, ϕvv = allocn(Val(dims * (dims+1) ÷ 2))
        qu, qv        = allocn(Val(dims))

        ∂x_ϕ, ∂y_ϕ   = allocn(Val(dims))
        ∂x_q, ∂y_q   = allocn(Val(dims))
        ∂x_ϕu, ∂y_ϕu = allocn(Val(dims))
        ∂x_ϕv, ∂y_ϕv = allocn(Val(dims))

        div_qv⃗ = alloc()
        div_ϕuv⃗, div_ϕvv⃗ = allocn(Val(dims))

        (ds, s, _, t) -> begin
            @unpack ϕ, ϕu, ϕv, q = s
            returnblocks(mempool)

            @inbounds @simd ivdep for i in eachindex(ϕ)
                ϕᵢ, ϕv⃗ᵢ, qᵢ = ϕ[i], @SVector[ϕu[i], ϕv[i]], q[i]

                ϕ⁻¹ᵢ = inv(ϕᵢ)
                v⃗ᵢ = ϕv⃗ᵢ * ϕ⁻¹ᵢ

                ϕ⁻¹[i] = ϕ⁻¹ᵢ
                u[i], v[i] = v⃗ᵢ
                ϕuu[i], ϕuv[i], _, ϕvv[i] = ϕv⃗ᵢ * v⃗ᵢ'
                qu[i], qv[i] = qᵢ * v⃗ᵢ
            end

            gradient!(∂x_ϕ, ∂y_ϕ, ϕ)
            gradient!(∂x_q, ∂y_q, q)
            gradient!(∂x_ϕu, ∂y_ϕu, ϕu)
            gradient!(∂x_ϕv, ∂y_ϕv, ϕv)
            divergence!(div_qv⃗, qu, qv)
            divergence!(div_ϕuv⃗, ϕuu, ϕuv)
            divergence!(div_ϕvv⃗, ϕuv, ϕvv)

            @inbounds @simd ivdep for i in eachindex(ϕ)
                ϕ⁻¹ᵢ, qᵢ   = ϕ⁻¹[i], q[i]
                v⃗ᵢ         = @SArray[u[i], v[i]]
                ∇⃗ϕᵢ        = @SArray[∂x_ϕ[i], ∂y_ϕ[i]]
                ∇⃗qᵢ        = @SArray[∂x_q[i], ∂y_q[i]]
                ∇⃗ϕv⃗ᵢ       = @SArray[∂x_ϕu[i] ∂x_ϕv[i]; ∂y_ϕu[i] ∂y_ϕv[i]]
                div_ϕv⃗v⃗ᵢ   = @SArray[div_ϕuv⃗[i], div_ϕvv⃗[i]]
                div_qv⃗ᵢ    = div_qv⃗[i]

                ds.ϕ[i]            = -1 // 2 * (∇⃗ϕᵢ ⋅ v⃗ᵢ + tr(∇⃗ϕv⃗ᵢ))
                ds.ϕu[i], ds.ϕv[i] = -1 // 2 * (∇⃗ϕv⃗ᵢ' * v⃗ᵢ + div_ϕv⃗v⃗ᵢ) - 2qᵢ * ϕ⁻¹ᵢ * ∇⃗qᵢ
                ds.q[i]            = -1 // 2 * (γ * div_qv⃗ᵢ + (2 - γ) * (∇⃗qᵢ ⋅ v⃗ᵢ))
            end

            for apply_mod! in apply_modifier!s
                apply_mod!(ds, s, t, (; ϕ⁻¹ = ϕ⁻¹, u = u, v = v))
            end
        end
    end

    function make_modifier(
            eqs::NordstromCompEuler2D{T},
            splitting::FluxDSL2024,
            _,
            fdop,
            alloc,
            mempool
    ) where {T}
        @unpack γ = eqs
        permute_cache = alloc()

        Dxₛ = splitting.scaling * ((splitting.volume ? fdop[1].Diᵥ : NullOperator()) +
                                   (splitting.surface ? fdop[1].Diₛ : NullOperator()))
        ∂xs(m) = axis_mul!(mempool, Val(1), Dxₛ, m, permute_cache)

        Dyₛ = splitting.scaling * ((splitting.volume ? fdop[2].Diᵥ : NullOperator()) +
                                   (splitting.surface ? fdop[2].Diₛ : NullOperator()))
        ∂ys(m) = axis_mul!(mempool, Val(2), Dyₛ, m, permute_cache)

        sqrtγ = √(γ)
        (ds, s, t, diagnostic) -> begin
            @unpack ϕ, ϕu, ϕv, q = s
            @unpack ϕ⁻¹, u, v = diagnostic

            λϕx  = zero(T)
            λϕy  = zero(T)
            λϕux = zero(T)
            λϕuy = zero(T)
            λϕvx = zero(T)
            λϕvy = zero(T)
            λqx  = zero(T)
            λqy  = zero(T)

            @inbounds for i in eachindex(u)
                ϕa = sqrtγ * abs(q[i])
                ϕeigvalx = abs(ϕu[i]) + ϕa
                ϕeigvaly = abs(ϕv[i]) + ϕa

                λϕx  = max(λϕx, ϕeigvalx)
                λϕy  = max(λϕy, ϕeigvaly)
                λϕux = max(λϕux, ϕ[i] * ϕeigvalx)
                λϕuy = max(λϕuy, ϕ[i] * ϕeigvaly)
                λϕvx = max(λϕvx, ϕ[i] * ϕeigvalx)
                λϕvy = max(λϕvy, ϕ[i] * ϕeigvaly)
                λqx  = max(λqx, ϕ⁻¹[i] * ϕeigvalx)
                λqy  = max(λqy, ϕ⁻¹[i] * ϕeigvaly)
            end

            λϕx *= 1 // 4
            λϕy *= 1 // 4
            λϕux *= 1 // 2
            λϕuy *= 1 // 2
            λϕvx *= 1 // 2
            λϕvy *= 1 // 2
            λqx *= 1 // 4
            λqy *= 1 // 4

            # Entropy Functions:
            #     Thermodynamic Entropy : -ϱs = -ϱ (log p - γ log ϱ)
            #     Energy                : ϱe  = p / (γ - 1) + ½ ϕ𝐮 ⋅ ϕ𝐮

            # ∂ᵩ  ϱe = ϕα²
            # ∂ᵩ -ϱs = 2 √ϱ (γ + γ log ϱ - log p) = 2 √ϱ (γ - s)
            @. ds.ϕ += (λϕx * $∂xs(ϕ) + λϕy * $∂ys(ϕ)) * ϕ⁻¹

            # ∂ᵩᵤ  ϱe = ½ ϕu
            # ∂ᵩᵤ -ϱs = 0
            @. ds.ϕu += (
                ((λϕux - λϕx * ϕ) * $∂xs(u) + λϕx * $∂xs(ϕu)) +
                ((λϕuy - λϕy * ϕ) * $∂ys(u) + λϕy * $∂ys(ϕu))
            ) * ϕ⁻¹

            @. ds.ϕv += (
                ((λϕvx - λϕx * ϕ) * $∂xs(v) + λϕx * $∂xs(ϕv)) +
                ((λϕvy - λϕy * ϕ) * $∂ys(v) + λϕy * $∂ys(ϕv))
            ) * ϕ⁻¹

            # ∂_q  ϱe    = 2 q / (γ - 1)
            # ∂_q -ϱs    = -2 ϱ / q
            @. ds.q += λqx * $∂xs(q) + λqy * $∂ys(q)
            nothing
        end
    end

    function make_modifier(
            eqs::NordstromCompEuler2D{T},
            splitting::FluxDSL2025,
            grid,
            fdop,
            alloc,
            mempool
    ) where {T}
        @unpack γ = eqs
        permute_cache = alloc()

        Dixᵥ = (splitting.volume ? splitting.scaling * fdop[1].Diᵥ : NullOperator())
        Dixₛ = (splitting.surface ? splitting.scaling * fdop[1].Diₛ : NullOperator())

        Diyᵥ = (splitting.volume ? splitting.scaling * fdop[2].Diᵥ : NullOperator())
        Diyₛ = (splitting.surface ? splitting.scaling * fdop[2].Diₛ : NullOperator())

        w_ϱ  = alloc()
        w_ϱu = alloc()
        w_ϱv = alloc()
        w_ϱe = alloc()

        ∂s_ϱ = alloc()
        ∂s_ϱu = alloc()
        ∂s_ϱv = alloc()
        ∂s_ϱe = alloc()

        λ_ϱ  = ntuple(_ -> alloc(), Val(2))
        λ_ϱu = ntuple(_ -> alloc(), Val(2))
        λ_ϱv = ntuple(_ -> alloc(), Val(2))
        λ_ϱe = ntuple(_ -> alloc(), Val(2))

        tmp = alloc()

        sqrtγ = √(γ)
        @inbounds (∂ₜstate, state, t, diagnostic) -> begin
            @unpack ϕ, ϕu, ϕv, q = state
            @unpack ϕ⁻¹, u, v = diagnostic

            @simd ivdep for i in eachindex(ϕ)
                ϕᵢ, ϕ⁻¹ᵢ, qᵢ = ϕ[i], ϕ⁻¹[i], q[i]
                ϕv⃗ᵢ = @SArray[ϕu[i], ϕv[i]]
                v⃗   = @SArray[u[i], v[i]]

                p⁻¹ = qᵢ^-2
                p   = qᵢ^2
                p²  = p^2
                ϱ   = ϕᵢ^2
                s   = 2 * (NaNMath.log(qᵢ) - γ * NaNMath.log(ϕᵢ))
                ϱuu, ϱvv = ϕv⃗ᵢ.^2
                K        = ϕv⃗ᵢ ⋅ ϕv⃗ᵢ / 2

                w_ϱ[i]           = (γ - s) / (γ - 1) - K * p⁻¹
                w_ϱu[i], w_ϱv[i] = ϕᵢ * p⁻¹ * ϕv⃗ᵢ
                w_ϱe[i]          = -ϱ * p⁻¹

                a  = sqrtγ * abs(qᵢ * ϕ⁻¹ᵢ)
                λ  = @. abs(v⃗) + a
                M² = @. 1 / γ * ϕv⃗ᵢ^2 * p⁻¹

                s_ϱ = (γ - 1) * ϱ * p² / ((γ - 1)^2 * K^2 + γ * p²)
                s_ϱu = p² / (p + (γ - 1) * ϱuu)
                s_ϱv = p² / (p + (γ - 1) * ϱvv)
                s_ϱe = 1 / (γ - 1) * p² * ϕ⁻¹ᵢ^2

                for d in 1:2
                    λ_ϱ[d][i]  = s_ϱ * λ[d]
                    λ_ϱu[d][i] = s_ϱu * λ[d]
                    λ_ϱv[d][i] = s_ϱv * λ[d]
                    λ_ϱe[d][i] = s_ϱe * λ[d] * 2M²[d] / (1 + M²[d])
                end
            end

            for (∂s, w, λs) in (
                (∂s_ϱ, w_ϱ, λ_ϱ),
                (∂s_ϱu, w_ϱu, λ_ϱu),
                (∂s_ϱv, w_ϱv, λ_ϱv),
                (∂s_ϱe, w_ϱe, λ_ϱe)
            )
                fill!(∂s, false)
                add_splitting!(∂s, Val(1), grid, Dixᵥ, Dixₛ, λs[1], w, tmp, permute_cache)
                add_splitting!(∂s, Val(2), grid, Diyᵥ, Diyₛ, λs[2], w, tmp, permute_cache)
            end

            @simd ivdep for i in eachindex(ϕ)
                ∂s_ϕ  = ∂s_ϱ[i] * ϕ⁻¹[i] / 2
                ∂s_ϕu = (∂s_ϱu[i] - ϕu[i] * ∂s_ϕ) * ϕ⁻¹[i]
                ∂s_ϕv = (∂s_ϱv[i] - ϕv[i] * ∂s_ϕ) * ϕ⁻¹[i]
                ∂s_p  = (γ - 1) * (∂s_ϱe[i] - ϕu[i] * ∂s_ϕu - ϕv[i] * ∂s_ϕv)

                ∂ₜstate.ϕ[i]  += ∂s_ϕ
                ∂ₜstate.ϕu[i] += ∂s_ϕu
                ∂ₜstate.ϕv[i] += ∂s_ϕv
                ∂ₜstate.q[i]  += ∂s_p / (2q[i])
            end
            nothing
        end
    end

    function semidiscretise(info::ReissSesterhennCompEuler2D{T}, grid, fdop;
            alloc = () -> Array{T}(undef, size(grid))
    ) where {T}
        @unpack γ = info

        mempool = MemoryPool(alloc())
        permute_cache = alloc()

        Dx = (@unpack D, B = fdop[1]; D + B)
        ∂x(m) = axis_mul!(mempool, Val(1), Dx, m, permute_cache)

        Dy = (@unpack D, B = fdop[2]; D + B)
        ∂y(m) = axis_mul!(mempool, Val(2), Dy, m, permute_cache)

        apply_modifier!s = map(
            mod -> make_modifier(info, mod, grid, fdop, alloc, mempool), info.modifiers)

        u   = alloc()
        v   = alloc()
        ϱu  = alloc()
        ϱv  = alloc()
        ϕ⁻¹ = alloc()

        fux = alloc()
        fvy = alloc()
        ϱuv = alloc()
        pu  = alloc()
        pv  = alloc()

        (ds, s, _, t) -> begin
            @unpack ϕ, ϕu, ϕv, p = s
            returnblocks(mempool)

            @. ϕ⁻¹ = inv(ϕ)
            @. u   = ϕu * ϕ⁻¹
            @. v   = ϕv * ϕ⁻¹
            @. ϱu  = ϕu * ϕ
            @. ϱv  = ϕv * ϕ

            @. pu  = pu * ϕu
            @. fux = ϕu * ϕu + 2p
            @. ϱuv = ϕu * ϕv
            @. fvy = ϕv * ϕv + 2p

            ds.ϕ .= @. -1 // 2 * ($∂x(ϱu) + $∂y(ϱv)) * ϕ⁻¹
            ds.ϕu .= @. -1 // 2 * (
                ($∂x(fux) + $∂y(ϱuv)) * ϕ⁻¹ +
                ϕu * $∂x(u) + ϕv * $∂y(u)
            )
            ds.ϕv .= @. -1 // 2 * (
                ($∂x(ϱuv) + $∂y(fvy)) * ϕ⁻¹ +
                ϕu * $∂x(v) + ϕv * $∂y(v)
            )
            ds.p .= @. (γ - 1) * (u * $∂x(p) + v * $∂y(p)) - γ * ($∂x(pu) + $∂y(pv))

            for apply_mod! in apply_modifier!s
                apply_mod!(ds, s, t, (; ϕ⁻¹ = ϕ⁻¹, u = u, v = v))
            end

            nothing
        end
    end
end
