using UnPack
using FastGaussQuadrature

"""
Hendrik Ranocha, Andrew R. Winters,
Michael Schlottke-Lakemper, Philipp Öffner, Jan Glaubitz, and
Gregor J. Gassner
High-order upwind summation-by-parts methods for nonlinear conservation laws
https://arxiv.org/abs/2311.13888
"""
@kwdef struct IsentropicVortex
    ε::Float64 = 10
    γ::Float64 = 1.4

    p₀::Float64 = 10
    ϱ₀::Float64 = 1
    u₀::Float64 = 1
    v₀::Float64 = 1

    xrange::Tuple{Float64, Float64} = (-5, 5)
    yrange::Tuple{Float64, Float64} = (-5, 5)
end

function (params::IsentropicVortex)(t, xs)
    bl = (params.xrange[1], params.yrange[1])
    tr = (params.xrange[2], params.yrange[2])
    v⃗₀ = (params.u₀, params.v₀)
    meshgrid = ntuple(_ -> Array{Float64}(undef, size(xs)), Val(4))
    for (I, x) in pairs(xs)
        xwrapped = @. bl + mod(x - v⃗₀ * t - bl, tr - bl)
        setindex!.(meshgrid, initial_solution(params, xwrapped), I)
    end
    return meshgrid
end

function initial_solution(params::IsentropicVortex, (x,y))
    T₀ = params.p₀ / params.ϱ₀
    r = hypot(x, y)
    T = T₀ - (params.γ - 1) * params.ε^2 / (8 * params.γ * π^2) * exp(1 - r^2)
    ϱ = params.ϱ₀ * (T / T₀)^(1 / (params.γ - 1))
    u = params.u₀ - params.ε / (2π) * exp((1 - r^2) / 2) * y
    v = params.v₀ + params.ε / (2π) * exp((1 - r^2) / 2) * x
    p = ϱ * T
    (ϱ, u, v, p)
end

"""

"""
@kwdef struct SedovBlast{T <: Real}
    γ::T = 1.4

    ϱ₀::T  = 1.0
    p₀::T  = 1e-5
    σ_ϱ::T = 0.25
    σ_p::T = 0.15
end

function (params::SedovBlast{T})(t::Real, xs) where {T}
    @assert iszero(t) "Analytic solution for the Sedov Blast has not been implemented for t > 0."
    @unpack ϱ₀, p₀, σ_ϱ, σ_p, γ = params

    r²(x) = sum(xᵢ -> xᵢ^2, x)

    ϱ = @. ϱ₀ + exp(-r²(xs) / (2σ_ϱ^2)) / (4π * σ_ϱ^2)
    u = @. zero(first(xs))
    v = @. zero(first(xs))
    p = @. p₀ + (γ - 1) / (4π * σ_p^2) * exp(-r²(xs) / (2σ_p^2))
    return (ϱ, u, v, p)
end

"""
Sod's shock tube problem.
"""
struct SodShockTube{T}
    γ::T

    ϱₗ::T; pₗ::T; uₗ::T; aₗ::T
    ϱ₃::T; p₃::T; u₃::T
    ϱ₄::T; p₄::T; u₄::T
    ϱᵣ::T; pᵣ::T; uᵣ::T; aᵣ::T

    c_fanleft::T
    c_contact::T
    c_shock::T
    c_fanright::T

    xₛ::T # initial shock location
end

function Base.show(io::IO, s::SodShockTube)
    print(io, "SodShockTube(;",
        "γ=" , s.γ , ", ",
        "ϱₗ=", s.ϱₗ, ", ",
        "pₗ=", s.pₗ, ", ",
        "uₗ=", s.uₗ, ", ",
        "ϱᵣ=", s.ϱᵣ, ", ",
        "pᵣ=", s.pᵣ, ", ",
        "uᵣ=", s.uᵣ, ", ",
        "xₛ=", s.xₛ, ")")
end

function bisection_root_find(f, xᵐⁱⁿ, xᵐᵃˣ; reltol = 1e-10)
    if !isfinite(xᵐᵃˣ)
        xᵐᵃˣ = 1.0
        while f(xᵐᵃˣ) < 0
            xᵐᵃˣ *= 2
        end
    end

    while xᵐᵃˣ > (1 + reltol) * xᵐⁱⁿ
        xᵐⁱᵈ = (xᵐⁱⁿ + xᵐᵃˣ) / 2
        if f(xᵐⁱᵈ) > 0
            xᵐᵃˣ = xᵐⁱᵈ
        else
            xᵐⁱⁿ = xᵐⁱᵈ
        end
    end

    (xᵐⁱⁿ + xᵐᵃˣ) / 2
end

function SodShockTube(;
    γ::T = 1.4,
    ϱₗ::T = 1.0,
    pₗ::T = 1.0,
    uₗ::T = 0.0,
    ϱᵣ::T = 0.125,
    pᵣ::T = 0.1,
    uᵣ::T = 0.0,
    xₛ::T = 0.0
) where {T}
    # Diagnostic Variables
    # a: speed of sound
    # c: interface speed

    Γ = (γ - 1) / (γ + 1)
    β = (γ - 1) / (2γ)

    # compute speed of sound
    aₗ = √(γ * pₗ / ϱₗ)
    aᵣ = √(γ * pᵣ / ϱᵣ)

    # compute p₃ = P * pᵣ
    P = bisection_root_find(0.0, Inf) do P
        a = (γ - 1) * (aᵣ / aₗ) * (P - 1)
        b = sqrt(2γ * (2γ + (γ + 1) * (P - 1)))
        return P - pₗ / pᵣ * (1 - a / b)^(1 / β)
    end
    p₃ = P * pᵣ
    p₄ = p₃

    u₄ = uₗ + 2aₗ / (γ - 1) * (1 - (p₃ / pₗ)^β)
    u₃ = u₄

    ϱ₃ = ϱₗ * (p₃ / pₗ)^(1 / γ)
    a₃ = √(γ * p₃ / ϱ₃)

    ϱ₄ = ϱᵣ * (pᵣ * Γ + p₃) / (pᵣ + Γ * p₃)

    # compute boundaries
    c_fanleft  = -aₗ
    c_fanright = u₄ - a₃
    c_contact  = u₄
    c_shock    = uᵣ + aᵣ * √((γ - 1 + P * (γ + 1)) / (2 * γ))

    return SodShockTube(γ,
        ϱₗ, pₗ, uₗ, aₗ,
        ϱ₃, p₃, u₃,
        ϱ₄, p₄, u₄,
        ϱᵣ, pᵣ, uᵣ, aᵣ,
        c_fanleft,
        c_contact,
        c_shock,
        c_fanright,
        xₛ)
end

function integrate_solution(fn, params::SodShockTube{T}, t::Real, (xₗ, xᵣ); numerical_res=100) where {T}
    @unpack γ, xₛ = params
    @unpack ϱₗ, pₗ, uₗ, aₗ, ϱᵣ, pᵣ, uᵣ, aᵣ = params
    @unpack c_fanleft, c_fanright, c_contact, c_shock = params
    @unpack ϱ₃, p₃, u₃ = params
    @unpack ϱ₄, p₄, u₄ = params

    β = (γ - 1) / (2γ)
    x₂ = clamp(xₛ + c_fanleft * t, xₗ, xᵣ)
    x₃ = clamp(xₛ + c_fanright * t, xₗ, xᵣ)
    x₄ = clamp(xₛ + c_contact * t, xₗ, xᵣ)
    x₅ = clamp(xₛ + c_shock * t, xₗ, xᵣ)
    ∫dI1 = fn((ϱₗ, uₗ, pₗ)) * (x₂ - xₗ)
    ∫dI3 = fn((ϱ₃, u₃, p₃)) * (x₄ - x₃)
    ∫dI4 = fn((ϱ₄, u₄, p₄)) * (x₅ - x₄)
    ∫dI5 = fn((ϱᵣ, uᵣ, pᵣ)) * (xᵣ - x₅)

    # numerically integrate
    xs, ws = gausslegendre(numerical_res)
    ∫dI2 = zero(T)
    for (x_ref, w) in zip(xs, ws)
        x = ((x₃ + x₂) + x_ref * (x₃ - x₂)) / 2
        u = 2 / (γ + 1) * (aₗ + (x - xₛ) / t)
        ϱ = ϱₗ * (1 - (γ - 1) / 2 * u / aₗ)^(2 / (γ - 1))
        p = pₗ * (1 - (γ - 1) / 2 * u / aₗ)^(1 / β)
        ∫dI2 += w * fn((ϱ, u, p))
    end
    ∫dI2 *= (x₃ - x₂) / 2

    return ∫dI1 + ∫dI2 + ∫dI3 + ∫dI4 + ∫dI5
end

function (params::SodShockTube{T})(t::Real, xs) where {T}
    """
    The solution is composed of 5 different sections:
        | I | II | III | IV | V |
    For derivation, see:
        https://arxiv.org/pdf/2103.02794
        https://en.wikipedia.org/wiki/Sod_shock_tube#Analytic_derivation
    """
    @unpack γ, xₛ = params
    @unpack ϱₗ, pₗ, uₗ, aₗ, ϱᵣ, pᵣ, uᵣ, aᵣ = params
    @unpack c_fanleft, c_fanright, c_contact, c_shock = params
    @unpack ϱ₃, p₃, u₃ = params
    @unpack ϱ₄, p₄, u₄ = params

    β = (γ - 1) / (2γ)

    ϱ_analytic = map(zero ∘ only, xs)
    u_analytic = map(zero ∘ only, xs)
    p_analytic = map(zero ∘ only, xs)

    @inbounds for i in eachindex(xs)
        x = only(xs[i])
        if x < xₛ + c_fanleft * t # region I
            ϱ_analytic[i] = ϱₗ
            u_analytic[i] = uₗ
            p_analytic[i] = pₗ
        elseif x < xₛ + c_fanright * t # region II
            u₂ = 2 / (γ + 1) * (aₗ + (x - xₛ) / t)
            ϱ_analytic[i] = ϱₗ * (1 - (γ - 1) / 2 * u₂ / aₗ)^(2 / (γ - 1))
            u_analytic[i] = u₂
            p_analytic[i] = pₗ * (1 - (γ - 1) / 2 * u₂ / aₗ)^(1 / β)
        elseif x < xₛ + c_contact * t # region III
            ϱ_analytic[i] = ϱ₃
            u_analytic[i] = u₃
            p_analytic[i] = p₃
        elseif x < xₛ + c_shock * t # region IV
            ϱ_analytic[i] = ϱ₄
            u_analytic[i] = u₄
            p_analytic[i] = p₄
        else # region V
            ϱ_analytic[i] = ϱᵣ
            u_analytic[i] = uᵣ
            p_analytic[i] = pᵣ
        end
    end

    return ϱ_analytic, u_analytic, p_analytic
end
