using HyperbolicPDEs: NullOperator
using UnPack: @unpack

"""
    BurgersScheme{StateVars}

Abstract supertype for all 1-dimensional Burgers' equation semidiscretisations
with prognostic variables `StateVars`.
"""
abstract type Burgers1DScheme{StateVars} end;

function to_primitive_vars(eqs::Burgers1DScheme, state)
    dst = (similar(state),)
    to_primitive_vars!(dst, eqs, state)
    return dst
end

(S::Type{<:Burgers1DScheme})(mods...; kwargs...) = S(; modifiers = Tuple(mods), kwargs...)

from_primitive_vars(::Burgers1DScheme{(:u,)}, (u,)) = u

function to_primitive_vars!(dst, ::Burgers1DScheme{(:u,)}, u)
    dst[1] .= u
    return
end


#######################
# 1D Burgers Equation #
#######################

@kwdef struct BurgersFluxForm1D{Modifiers <: Tuple} <: Burgers1DScheme{(:u,)}
    modifiers::Modifiers = ()
end

function semidiscretise(info::BurgersFluxForm1D, xs, fdop;
        alloc = () -> Array{Float64}(undef, size(xs))
)
    D = (@unpack D, B = fdop[1]; D + B)
    ∂x(m) = D * m

    apply_modifier!s = map(mod -> make_modifier(info, mod, xs, fdop, alloc), info.modifiers)

    f = alloc()
    (du, u, _, t::Real) -> begin
        @. f = u^2 / 2
        @. du = -$∂x(f)

        for apply_mod! in apply_modifier!s
            apply_mod!(du, u, t)
        end

        return nothing
    end
end

"""
"""
@kwdef struct BurgersSkewSym1D{Modifiers <: Tuple} <: Burgers1DScheme{(:u,)}
    modifiers::Modifiers = ()
end

function semidiscretise(info::BurgersSkewSym1D, xs, fdop;
        alloc = () -> Array{Float64}(undef, size(xs))
)
    D = (@unpack D, B = fdop[1]; D + B)
    ∂x(m) = D * m

    apply_modifier!s = map(mod -> make_modifier(info, mod, xs, fdop, alloc), info.modifiers)

    u² = alloc()

    (du, u, _, t::Real) -> begin
        @. u² = u^2
        @. du = -1 / 3 * u * $∂x(u) - $∂x(u²) / 3

        for apply_mod! in apply_modifier!s
            apply_mod!(du, u, t)
        end

        return nothing
    end
end

"""
The Advective form of burgers equation
```math
    ∂ₜ u + u ∂ₓ u = 0.
```
"""
@kwdef struct BurgersAdvective1D{Modifiers <: Tuple} <: Burgers1DScheme{(:u,)}
    modifiers::Modifiers = ()
end

function semidiscretise(info::BurgersAdvective1D, xs, fdop;
        alloc = () -> Array{Float64}(undef, size(xs))
)
    D = (@unpack D, B = fdop[1]; D + B)
    ∂x(m) = D * m

    apply_modifier!s = map(mod -> make_modifier(info, mod, xs, fdop, alloc), info.modifiers)

    (du, u, _, t::Real) -> begin
        @. du = -u * $∂x(u)

        for apply_mod! in apply_modifier!s
            apply_mod!(du, u, t)
        end

        return
    end
end

#######
# MMS #
#######

struct SourceMMS{T}
    exact::T
end

function make_modifier(::Burgers1DScheme{(:u,)}, mms::SourceMMS, grid, _...)
    ∂t = SymUtils.∂t
    ∂x = SymUtils.SpatialDerivative{1}()
    u, = SymUtils.LazyField.(mms.exact)

    source_term = ∂t(u) + u * ∂x(u)
    xs = collect(grid)
    return (ds, s, t::Real, _...) -> begin
        @inbounds @simd ivdep for i in eachindex(xs)
            ds[i] += source_term(t, xs[i])
        end
    end
end

###################
# Flux Splittings #
###################

@kwdef struct FluxLaxFriedrichs{T}
    volume::Bool  = true
    surface::Bool = true
    scaling::T    = 1.0
end
"""
    FluxEntropyStable{T}

An entropy stable flux splitting with respect to the kinetic energy/square entropy
``η(u) = ½u²``.
"""
@kwdef struct FluxEntropyStable{T}
    volume::Bool  = true
    surface::Bool = true
    scaling::T    = 1.0
end
"""
    FluxLogEntropyStable{T}

An entropy stable flux splitting with respect to the log entropy ``η(u) = -\\log(u)``.
"""
@kwdef struct FluxLogEntropyStable{T}
    volume::Bool  = true
    surface::Bool = true
    scaling::T    = 1.0
end

function make_modifier(
        ::Burgers1DScheme{(:u,)},
        splitting::Union{FluxLaxFriedrichs, FluxEntropyStable},
        grid,
        fdop,
        alloc
)
    Diᵥ = (splitting.volume ? splitting.scaling * fdop[1].Diᵥ : NullOperator())
    Diₛ = (splitting.surface ? splitting.scaling * fdop[1].Diₛ : NullOperator())

    tmp = alloc()
    λ = alloc()

    (du, u, _) -> begin
        @. λ = abs(u)
        add_splitting!(du, Val(1), grid, Diᵥ, Diₛ, λ, u, tmp)
    end
end

function make_modifier(
        ::Burgers1DScheme{(:u,)},
        splitting::FluxLogEntropyStable,
        grid,
        fdop,
        alloc
)
    Diᵥ = (splitting.volume ? splitting.scaling * fdop[1].Diᵥ : NullOperator())
    Diₛ = (splitting.surface ? splitting.scaling * fdop[1].Diₛ : NullOperator())

    tmp = alloc()
    λ = alloc()
    r = alloc()

    # -∂ₓ u⁻¹ = u⁻² ∂ₓ u

    (du, u, _) -> begin
        @. r = -inv(u)
        @. λ = abs(u)^3
        add_splitting!(du, Val(1), grid, Diᵥ, Diₛ, λ, r, tmp)
    end
end
