using UnPack

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

function (params::IsentropicVortex)(t, xs, ys)
    meshgrid = (
        Matrix{Float64}(undef, length(xs), length(ys)),
        Matrix{Float64}(undef, length(xs), length(ys)),
        Matrix{Float64}(undef, length(xs), length(ys)),
        Matrix{Float64}(undef, length(xs), length(ys))
    )
    for (n, x) in enumerate(xs), (m, y) in enumerate(ys)
        xwrapped = params.xrange[1] + mod(
            -params.xrange[1] + x - params.u₀ * t,
            params.xrange[2] - params.xrange[1]
        )
        ywrapped = params.yrange[1] + mod(
            -params.yrange[1] + y - params.v₀ * t,
            params.yrange[2] - params.yrange[1]
        )
        setindex!.(meshgrid, initial_solution(params, xwrapped, ywrapped), n, m)
    end
    return meshgrid
end

function initial_solution(params::IsentropicVortex, x::Real, y::Real)
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

    x = xs

    r²(x...) = sum(xᵢ -> xᵢ^2, x)

    ϱ = @. ϱ₀ + exp(-r²(x) / (2σ_ϱ^2)) / (4π * σ_ϱ^2)
    u = @. zero(x)
    p = @. p₀ + (γ - 1) / (4π * σ_p^2) * exp(-r²(x) / (2σ_p^2))

    return (ϱ, u, p)
end

function (params::SedovBlast{T})(t::Real, xs, ys) where {T}
    @assert iszero(t) "Analytic solution for the Sedov Blast has not been implemented for t > 0."
    @unpack ϱ₀, p₀, σ_ϱ, σ_p, γ = params

    x = xs
    y = ys'

    r²(x...) = sum(xᵢ -> xᵢ^2, x)

    ϱ = @. ϱ₀ + exp(-r²(x, y) / (2σ_ϱ^2)) / (4π * σ_ϱ^2)
    u = @. zero(x + y)
    v = @. zero(x + y)
    p = @. p₀ + (γ - 1) / (4π * σ_p^2) * exp(-r²(x, y) / (2σ_p^2))
    return (ϱ, u, v, p)
end
