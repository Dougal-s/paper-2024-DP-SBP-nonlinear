using Polynomials
using UnPack
using FastGaussQuadrature

"""
[1] O. Delestre, C. Lucas, P.-A. Ksinant, F. Darboux, C. Laguerre, T.N.T. Vo, F. James and S. Cordier
SWASHES: a compilation of Shallow Water Analytic Solutions for Hydraulic and Environmental Studies
https://arxiv.org/pdf/1110.0288.pdf
"""

struct DambreakPeriodic
    hₗ::Float64
    hᵣ::Float64
    x₀::Float64
    g::Float64

    cₘ::Float64
    aₗ::Float64
end

function Base.show(io::IO, s::DambreakPeriodic)
    print(io, "DambreakPeriodic(;",
        "hₗ=", s.hₗ, ", ",
        "hᵣ=", s.hᵣ, ", ",
        "x₀=", s.x₀, ", ",
        "g=", s.g, ")")
end

function DambreakPeriodic(; hₗ, hᵣ, x₀, g)
    cₘ = solve_for_cm(hₗ, hᵣ, g)
    aₗ = sqrt(g * hₗ)
    DambreakPeriodic(hₗ, hᵣ, x₀, g, cₘ, aₗ)
end

function (params::DambreakPeriodic)(t, xs)
    exact_a = [dambreak_exact_aperiodic(params, t, abs(only(x))) for x in xs]
    h = getindex.(exact_a, 1)
    u = sign.(only.(xs)) .* getindex.(exact_a, 2)
    (h, u, zero(only.(xs)))
end

function integrate_solution(
    fn,
    params::DambreakPeriodic,
    t::Real,
    (xₗ, xᵣ);
    numerical_res = 20
)
    @unpack hₗ, hᵣ, x₀, g, cₘ, aₗ = params
    xA = x₀ - t * √(g * hₗ)
    xB = x₀ + t * (2 * √(g * hₗ) - 3cₘ)
    xC = x₀ + t * 2cₘ^2 * (√(g * hₗ) - cₘ) / (cₘ^2 - g * hᵣ)

    ∫dI1 = (xA - 0.0) * 2fn((hₗ, 0.0, 0.0))
    ∫dI3 = (xC - xB) * let
        h, v = (cₘ^2 / g, 2(aₗ - cₘ))
        fn((h, v, 0.0)) + fn((h, -v, 0.0))
    end
    ∫dI4 = (xᵣ - xC) * 2fn((hᵣ, 0.0, 0.0))

    # numerically integrate
    xs, ws = gausslegendre(numerical_res)
    ∫dI2 = 0.0
    if t > 0
        for (x_ref, w) in zip(xs, ws)
            x = ((xB + xA) + x_ref * (xB - xA)) / 2
            h = 4 / (9g) * (aₗ - (x - x₀) / (2t))^2
            v = 2 / 3 * (aₗ + (x - x₀) / t)
            ∫dI2 += w * (fn((h, v, 0.0)) + fn((h, -v, 0.0)))
        end
    end
    ∫dI2 *= (xB - xA) / 2

    return ∫dI1 + ∫dI2 + ∫dI3 + ∫dI4
end

function dambreak_exact_aperiodic(params::DambreakPeriodic, t, x)
    @unpack hₗ, hᵣ, x₀, g, cₘ, aₗ = params
    xA = x₀ - t * aₗ
    xB = x₀ + t * (2 * aₗ - 3cₘ)
    xC = x₀ + t * 2cₘ^2 * (aₗ - cₘ) / (cₘ^2 - g * hᵣ)

    if x < xA
        (hₗ, 0.0)
    elseif x < xB
        (4 / (9g) * (aₗ - (x - x₀) / (2t))^2,
            2 / 3 * (aₗ + (x - x₀) / t))
    elseif x < xC
        (cₘ^2 / g,
            2 * (aₗ - cₘ))
    else
        (hᵣ, 0.0)
    end
end

function solve_for_cm(hₗ, hᵣ, g)
    x        = Polynomial([0, 1])
    cm_poly  = -8 * g * hᵣ * x^2 * (√(g * hₗ) - x)^2 + (x^2 - g * hᵣ)^2 * (x^2 + g * hᵣ)
    dcm_poly = derivative(cm_poly)
    cₘ       = √(g * hₗ / 2)
    while cm_poly(cₘ)^2 > 1e-13
        cₘ -= cm_poly(cₘ) / dcm_poly(cₘ)
    end
    cₘ
end