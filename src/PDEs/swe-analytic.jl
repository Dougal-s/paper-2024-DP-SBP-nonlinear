using Polynomials
using UnPack

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
end

function DambreakPeriodic(; hₗ, hᵣ, x₀, g)
    cₘ = solve_for_cm(hₗ, hᵣ, g)
    DambreakPeriodic(hₗ, hᵣ, x₀, g, cₘ)
end

function (params::DambreakPeriodic)(t, xs)
    mirrored   = xs .> (xs[begin] + xs[end]) / 2
    xs_wrapped = ifelse.(mirrored, xs[begin] + xs[end] .- xs, xs)
    exact      = [dambreak_exact_aperiodic(params, t, x) for x in xs_wrapped]
    h          = getindex.(exact, 1)
    u          = getindex.(exact, 2)
    (h, ifelse.(mirrored, -u, u))
end

function dambreak_energy_aperiodic(params::DambreakPeriodic, t, xrange)
    @unpack hₗ, hᵣ, x₀, g, cₘ = params
    xA = x₀ - t * √(g * hₗ)
    xB = x₀ + t * (2 * √(g * hₗ) - 3cₘ)
    xC = x₀ + t * 2cₘ^2 * (√(g * hₗ) - cₘ) / (cₘ^2 - g * hᵣ)

    # energy = 1/2 hu^2 + 1/2 gh^2
    e1 = (xA - xrange[1]) * 1 / 2 * g * hₗ^2
    e2 = let
        x              = Polynomial([0, 1])
        h              = 4 / (9g) * (√(g * hₗ) - (x - x₀) / (2t))^2
        u              = 2 / 3 * (√(g * hₗ) + (x - x₀) / t)
        energy_density = 1 / 2 * h * u^2 + 1 / 2 * g * h^2
        energy         = integrate(energy_density)
        return energy(xB) - energy(xA)
    end

    e3 = let
        h = cₘ^2 / g
        u = 2 * (√(g * hₗ) - cₘ)
        (xC - xB) * 1 / 2 * (h * u^2 + g * h^2)
    end

    e4 = let
        (xrange[2] - xC) * 1 / 2 * g * hᵣ^2
    end

    return e1 + e2 + e3 + e4
end

function energyexact(params::DambreakPeriodic, t, xrange)
    return 2 *
           dambreak_energy_aperiodic(params, t, (xrange[1], (xrange[2] + xrange[1]) / 2))
end

function dambreak_exact_aperiodic(params::DambreakPeriodic, t, x)
    @unpack hₗ, hᵣ, x₀, g, cₘ = params
    xA = x₀ - t * √(g * hₗ)
    xB = x₀ + t * (2 * √(g * hₗ) - 3cₘ)
    xC = x₀ + t * 2cₘ^2 * (√(g * hₗ) - cₘ) / (cₘ^2 - g * hᵣ)

    if x < xA
        (hₗ, 0.0)
    elseif x < xB
        (4 / (9g) * (√(g * hₗ) - (x - x₀) / (2t))^2,
            2 / 3 * (√(g * hₗ) + (x - x₀) / t))
    elseif x < xC
        (cₘ^2 / g,
            2 * (√(g * hₗ) - cₘ))
    else
        (hᵣ, 0)
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