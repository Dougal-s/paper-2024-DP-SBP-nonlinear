using Symbolics
using UnPack: @unpack

"""
    BurgersScheme{StateVars}

Abstract supertype for all 1-dimensional Burgers' equation semidiscretisations
with prognostic variables `StateVars`.
"""
abstract type Burgers1DScheme{StateVars} end;

"""
    global_cons_qtys(eq, (u, ), Hx)

Get the following conserved global quantities from the primitive variables:
    the total energy (u^2),
    the total velocity (u).
"""
function global_cons_qtys(eqs::Burgers1DScheme, state, Hx)
    total_energy(s) =
        energy_norm(s, Hx) do (u,)
            u^2
        end
    total_velocity(s) =
        energy_norm(s, Hx) do (u,)
            u
        end

    return (total_energy(state), total_velocity(state))
end

function to_primitive_vars(eqs::Burgers1DScheme, state)
    dst = (similar(state),)
    to_primitive_vars!(dst, eqs, state)
    return dst
end

#######################
# 1D Burgers Equation #
#######################

@kwdef struct BurgersFluxForm1D{Modifiers<:Tuple} <: Burgers1DScheme{(:u,)}
    modifiers::Modifiers = ()
end

function BurgersFluxForm1D(mods...; kwargs...)
    BurgersFluxForm1D(; modifiers=Tuple(mods), kwargs...)
end

from_primitive_vars(::BurgersFluxForm1D, (u,)) = u

function to_primitive_vars!(dst, ::BurgersFluxForm1D, u)
    dst[1] .= u
    return ()
end

function semidiscretise(info::BurgersFluxForm1D, xs, fdop)
    D₊, D₋ = fdop
    D = (D₋ + D₊) / 2
    ∂x(m) = D * m

    apply_modifier!s = map(mod -> make_modifier(info, mod, xs, fdop), info.modifiers)

    (du, u, _, t::Real) -> begin
        @. du = -1 // 2 * $∂x(u^2)

        for apply_mod! in apply_modifier!s
            apply_mod!(du, u, t)
        end

        return nothing
    end
end

"""
"""
@kwdef struct BurgersSkewSym1D{Modifiers<:Tuple} <: Burgers1DScheme{(:u,)}
    modifiers::Modifiers = ()
end

function BurgersSkewSym1D(mods...; kwargs...)
    BurgersSkewSym1D(; modifiers=Tuple(mods), kwargs...)
end

from_primitive_vars(::BurgersSkewSym1D, (u,)) = u

function to_primitive_vars!(dst, ::BurgersSkewSym1D, u)
    dst[1] .= u
    return ()
end

function semidiscretise(info::BurgersSkewSym1D, xs, fdop)
    D₊, D₋ = fdop
    D = (D₋ + D₊) / 2
    ∂x(m) = D * m

    apply_modifier!s = map(mod -> make_modifier(info, mod, xs, fdop), info.modifiers)

    (du, u, _, t::Real) -> begin
        @. du = -1 / 3 * u * $∂x(u) - $∂x(u^2) / 3

        for apply_mod! in apply_modifier!s
            apply_mod!(du, u, t)
        end

        return nothing
    end
end

#######
# MMS #
#######

struct SourceMMS{T}
    exact::T
end

function make_modifier(::Burgers1DScheme{(:u,)}, mms::SourceMMS, xs, fdop)
    @variables t x
    u, = map(exact -> exact(t, x), mms.exact)
    ∂x = Differential(x)
    ∂t = Differential(t)

    source_term_sym = ∂t(u) + u * ∂x(u)
    source_term = (Symbolics.toexpr ∘ simplify ∘ expand_derivatives)(source_term_sym)
    add_source_terms! = eval(quote
        (du, t::Real) -> begin
            for (i, x) in enumerate($xs)
                du[i] += $(source_term)
            end
            return nothing
        end
    end)

    return (ds, s, t::Real, _...) -> begin
        @invokelatest add_source_terms!(ds, t)
        return nothing
    end
end

###################
# Flux Splittings #
###################

@kwdef struct FluxLaxFriedrichs{T}
    scaling::T = 1.0
end
@kwdef struct FluxEntropyStable{T}
    scaling::T = 1.0
end

function make_modifier(
    ::Burgers1DScheme{(:u,)},
    splitting::Union{FluxLaxFriedrichs,FluxEntropyStable},
    xs,
    fdop
)
    D₊, D₋ = fdop
    Dₛ = (D₊ - D₋) / 2
    ∂xs(f) = Dₛ * f

    (du, u, _) -> begin
        γ = splitting.scaling * maximum(abs, u)
        @. du += γ * $∂xs(u)

        return nothing
    end
end
