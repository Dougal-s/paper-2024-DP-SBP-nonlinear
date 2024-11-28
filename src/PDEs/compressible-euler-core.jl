using Symbolics
using RecursiveArrayTools: NamedArrayPartition
using UnPack: @unpack

abstract type CompressibleEuler1DScheme end;
abstract type CompressibleEuler2DScheme end;

"""
    global_cons_qtys(eq, (ϱ, u[, v], p), Hx[, Hy])

Get the following conserved global quantities from the primitive variables:
    the total energy
    the total mass
    the total x momentum
    [the total y momentum]
    the total physical entropy
"""
function global_cons_qtys(eqs::CompressibleEuler1DScheme, state, Hx)
    γ = eqs.γ

    total_energy(s)  = energy_norm(s, Hx) do (ϱ, u, p)
        p / (γ - 1) + ϱ * u^2 / 2
    end
    total_mass(s)    = energy_norm(s, Hx) do (ϱ, u, p)
        ϱ
    end
    total_ϱu(s)      = energy_norm(s, Hx) do (ϱ, u, p)
        ϱ * u
    end
    total_entropy(s) = energy_norm(s, Hx) do (ϱ, u, p)
        -ϱ * (log(p) - γ * log(ϱ))
    end

    return (total_energy(state), total_mass(state), total_ϱu(state), total_entropy(state))
end

function global_cons_qtys(eqs::CompressibleEuler2DScheme, state, Hx, Hy)
    γ = eqs.γ

    total_energy(s)  = energy_norm(s, Hx, Hy) do (ϱ, u, v, p)
        p / (γ - 1) + ϱ * (u^2 + v^2) / 2
    end
    total_mass(s)    = energy_norm(s, Hx, Hy) do (ϱ, u, v, p)
        ϱ
    end
    total_ϱu(s)      = energy_norm(s, Hx, Hy) do (ϱ, u, v, p)
        ϱ * u
    end
    total_ϱv(s)      = energy_norm(s, Hx, Hy) do (ϱ, u, v, p)
        ϱ * v
    end
    total_entropy(s) = energy_norm(s, Hx, Hy) do (ϱ, u, v, p)
        -ϱ * (log(p) - γ * log(ϱ))
    end

    return (total_energy(state), total_mass(state),
        total_ϱu(state), total_ϱv(state), total_entropy(state))
end

function to_primitive_vars(eqs::CompressibleEuler1DScheme, state)
    T = eltype(state)
    sz = size(getproperty(state, propertynames(state)[begin]))
    dst = (Array{T}(undef, sz), Array{T}(undef, sz), Array{T}(undef, sz))
    to_primitive_vars!(dst, eqs, state)
    return dst
end

function to_primitive_vars(eqs::CompressibleEuler2DScheme, state)
    T = eltype(state)
    sz = size(getproperty(state, propertynames(state)[begin]))
    dst = (
        Array{T}(undef, sz), Array{T}(undef, sz), Array{T}(undef, sz), Array{T}(undef, sz))
    to_primitive_vars!(dst, eqs, state)
    return dst
end

@kwdef struct FluxLaxFriedrichs{T <: Real}
    scaling::T = 1.0
end

@kwdef struct FluxEntropyStable{T <: Real}
    scaling::T = 1.0
end

function make_dissipation(::CompressibleEuler1DScheme, ::Nothing, _, _)
    return (_, _) -> nothing
end

function make_dissipation(::CompressibleEuler2DScheme, ::Nothing, _, _)
    return (_, _) -> nothing
end

#########################
# 1D Compressible Euler #
#########################

@kwdef struct CompEulerFluxForm1D{T <: Real, Splitting} <: CompressibleEuler1DScheme
    γ::T                      = 1.4
    flux_splitting::Splitting = nothing
end

function from_primitive_vars(info::CompEulerFluxForm1D, (ϱ, u, p))
    @unpack γ = info
    ϱe = @. p / (γ - 1) + 1 // 2 * ϱ * u^2
    return NamedArrayPartition(ϱ = ϱ, ϱu = ϱ .* u, ϱe = ϱe)
end

function to_primitive_vars!(dst, info::CompEulerFluxForm1D, state)
    @unpack γ = info
    @unpack ϱ, ϱu, ϱe = state
    dst[1] .= @. ϱ
    dst[2] .= @. ϱu / ϱ
    dst[3] .= @. (γ - 1) * (ϱe - 1 // 2 * ϱu^2 / ϱ)
    return ()
end

function MMS_source_terms(info::CompEulerFluxForm1D, solution)
    @unpack γ = info

    @variables t x
    ϱ, u, p = Ref((t, x)) .|> Base.splat.(solution)
    ϱu      = ϱ * u
    ϱe      = p / (γ - 1) + 1 // 2 * ϱ * u^2
    ∂x      = Differential(x)
    ∂t      = Differential(t)

    source_term_ϱ  = ∂t(ϱ) + ∂x(ϱu)
    source_term_ϱu = ∂t(ϱu) + ∂x(ϱ * u^2 + p)
    source_term_ϱe = ∂t(ϱe) + ∂x(u * (ϱe + p))

    source_terms_sym = map(Symbolics.toexpr ∘ simplify ∘ expand_derivatives,
        (source_term_ϱ, source_term_ϱu, source_term_ϱe))
    source_terms = eval(quote
        (t::Real, xs) -> (
            map(x -> begin
                    $(source_terms_sym[1])
                end, xs),
            map(x -> begin
                    $(source_terms_sym[2])
                end, xs),
            map(x -> begin
                    $(source_terms_sym[3])
                end, xs)
        )
    end)
    (t::Real, xs) -> @invokelatest source_terms(t, xs)
end

function semidiscretise(
        info::CompEulerFluxForm1D{T}, grid, fdop; source = nothing) where {T}
    @unpack γ, flux_splitting = info

    xs = grid

    D₊, D₋  = fdop
    D       = (D₋ + D₊) / 2
    D_split = (D₋ - D₊) / 2

    ∂x(m)  = D * m
    ∂xs(m) = D_split * m

    add_dissipation! = make_dissipation(info, flux_splitting, grid, ∂xs)

    u  = Vector{T}(undef, length(xs))
    p  = Vector{T}(undef, length(xs))
    f₂ = Vector{T}(undef, length(xs))
    f₃ = Vector{T}(undef, length(xs))

    (ds, s, _, t) -> begin
        @unpack ϱ, ϱu, ϱe = s

        src_ϱ, src_ϱu, src_ϱe = isnothing(source) ?
                                (zero(T), zero(T), zero(T)) :
                                source(t, xs)

        @. u = ϱu / ϱ
        @. p = (γ - 1) * (ϱe - 1 / 2 * ϱu * u)
        @. f₂ = ϱu * u + p
        @. f₃ = u * (ϱe + p)

        ds.ϱ  .= @. -$∂x(ϱu) + src_ϱ
        ds.ϱu .= @. -$∂x(f₂) + src_ϱu
        ds.ϱe .= @. -$∂x(f₃) + src_ϱe

        add_dissipation!(ds, (s, u))
        nothing
    end
end

function make_dissipation(eqs::CompEulerFluxForm1D, ::FluxLaxFriedrichs, grid, ∂xs)
    @unpack γ = eqs
    (ds, (s, u)) -> begin
        @unpack ϱ, ϱu, ϱe = s
        λ = maximum(zip(ϱ, ϱu, u, ϱe)) do (ϱ, ϱu, u, ϱe)
            p = (γ - 1) * (ϱe - 1 // 2 * ϱu * u)
            a = √(γ * p / ϱ)
            abs(u) + a
        end

        ds.ϱ  .+= @. -λ * $∂xs(ϱ)
        ds.ϱu .+= @. -λ * $∂xs(ϱu)
        ds.ϱe .+= @. -λ * $∂xs(ϱe)
    end
end

"""

"""
@kwdef struct ReissSesterhennCompEuler1D{T <: Real, Splitting} <: CompressibleEuler1DScheme
    γ::T                      = 1.4
    flux_splitting::Splitting = nothing
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

function MMS_source_terms(info::ReissSesterhennCompEuler1D, solution)
    @unpack γ = info

    @variables t x
    ϱ, u, p = Ref((t, x)) .|> Base.splat.(solution)
    ϕ       = √ϱ
    ϕu      = ϕ * u
    ϱu      = ϱ * u
    ∂x      = Differential(x)
    ∂t      = Differential(t)

    source_term_ϕ  = (∂t(ϱ) + ∂x(ϱu)) / (2 * ϕ)
    source_term_ϕu = ∂t(ϕu) + (∂x(ϕu^2 + 2 * p) + ϱu * ∂x(u)) / (2 * ϕ)
    source_term_p  = ∂t(p) + γ * ∂x(u * p) - (γ - 1) * u * ∂x(p)

    source_terms_sym = map(Symbolics.toexpr ∘ simplify ∘ expand_derivatives,
        (source_term_ϕ, source_term_ϕu, source_term_p))
    source_terms = eval(quote
        (t::Real, xs) -> (
            map(x -> begin
                    $(source_terms_sym[1])
                end, xs),
            map(x -> begin
                    $(source_terms_sym[2])
                end, xs),
            map(x -> begin
                    $(source_terms_sym[3])
                end, xs)
        )
    end)
    (t::Real, xs) -> @invokelatest source_terms(t, xs)
end

function semidiscretise(
        info::ReissSesterhennCompEuler1D{T}, grid, fdop; source = nothing) where {T}
    @unpack γ, flux_splitting = info

    xs = grid

    D₊, D₋  = fdop
    D       = (D₋ + D₊) / 2
    D_split = (D₋ - D₊) / 2

    ∂x(f)  = D * f
    ∂xs(f) = D_split * f

    add_dissipation! = make_dissipation(info, flux_splitting, grid, ∂xs)

    ϕ⁻¹ = Array{T}(undef, length(xs))
    u   = Array{T}(undef, length(xs))
    ϱu  = Array{T}(undef, length(xs))

    (ds, s, _, t) -> begin
        @unpack ϕ, ϕu, p = s

        @. ϕ⁻¹ = 1 / ϕ
        @. u   = ϕu * ϕ⁻¹
        @. ϱu  = ϕu * ϕ

        src_ϕ, src_ϕu, src_p = isnothing(source) ?
                               (zero(T), zero(T), zero(T)) :
                               source(t, xs)

        ds.ϕ  .= @. -1 // 2 * $∂x(ϱu) * ϕ⁻¹ + src_ϕ
        ds.ϕu .= @. -1 // 2 * ($∂x(ϕu^2 + 2p) * ϕ⁻¹ + ϕu * $∂x(u)) + src_ϕu
        ds.p  .= @. -γ * $∂x(u * p) + (γ - 1) * u * $∂x(p) + src_p

        add_dissipation!(ds, (s, ϕ⁻¹, u, ϱu))
    end
end

function make_dissipation(eqs::ReissSesterhennCompEuler1D, ::FluxEntropyStable, grid, ∂xs)
    @unpack γ = eqs
    (ds, (s, ϕ⁻¹, u, ϱu)) -> begin
        @unpack ϕ, ϕu, p = s

        λϕ  = 1 // 2 * maximum(abs, u)
        λϕu = maximum(abs, ϱu)
        λp  = (γ - 1) * maximum(x -> x^2, ϕu)

        ds.ϕ  .= @. -λϕ * $∂xs(ϕ^2) * ϕ⁻¹
        ds.ϕu .= @. -λϕu * $∂xs(u) * ϕ⁻¹
        ds.p  .= @. -λp * u * $∂xs(u)
    end
end

"""

"""
@kwdef struct NordstromCompEuler1D{T <: Real, Splitting} <: CompressibleEuler1DScheme
    γ::T                      = 1.4
    flux_splitting::Splitting = nothing
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

function MMS_source_terms(info::NordstromCompEuler1D, solution)
    @unpack γ = info

    @variables t x
    ϱ, u, p = Ref((t, x)) .|> Base.splat.(solution)
    ϕ       = √ϱ
    ϕu      = ϕ * u
    q       = √p
    ∂x      = Differential(x)
    ∂t      = Differential(t)

    source_term_ϕ  = ∂t(ϕ) + 1 // 2 * (∂x(ϕu) + u * ∂x(ϕ))
    source_term_ϕu = ∂t(ϕu) + 1 // 2 * (∂x(u * ϕu) + u * ∂x(ϕu) + 4q / ϕ * ∂x(q))
    source_term_q  = ∂t(q) + 1 // 2 * (∂x(γ * u * q) + (2 - γ) * u * ∂x(q))

    source_terms_sym = map(Symbolics.toexpr ∘ simplify ∘ expand_derivatives,
        (source_term_ϕ, source_term_ϕu, source_term_q))
    source_terms = eval(quote
        (t::Real, xs) -> (
            map(x -> begin
                    $(source_terms_sym[1])
                end, xs),
            map(x -> begin
                    $(source_terms_sym[2])
                end, xs),
            map(x -> begin
                    $(source_terms_sym[3])
                end, xs)
        )
    end)
    (t::Real, xs) -> @invokelatest source_terms(t, xs)
end

function semidiscretise(
        info::NordstromCompEuler1D{T}, grid, fdop; source = nothing) where {T}
    @unpack γ, flux_splitting = info

    xs = grid

    D₊, D₋  = fdop
    D       = (D₋ + D₊) / 2
    D_split = (D₋ - D₊) / 2

    ∂x(f)  = D * f
    ∂xs(f) = D_split * f

    add_dissipation! = make_dissipation(info, flux_splitting, grid, ∂xs)

    u   = Vector{T}(undef, length(xs))
    ϕ⁻¹ = Vector{T}(undef, length(xs))
    ϕuu = Vector{T}(undef, length(xs))
    qu  = Vector{T}(undef, length(xs))

    (ds, s, _, t) -> begin
        @unpack ϕ, ϕu, q = s

        @. ϕ⁻¹ = 1 / ϕ
        @. u   = ϕu * ϕ⁻¹
        @. ϕuu = ϕu * u
        @. qu  = q * u

        ∂x_ϕu = ∂x(ϕu)
        ∂x_q  = ∂x(q)

        src_ϕ, src_ϕu, src_q = isnothing(source) ?
                               (zero(T), zero(T), zero(T)) :
                               source(t, xs)

        # Entropy: ϕα^2
        ds.ϕ .= @. -1 // 2 * (∂x_ϕu + u * $∂x(ϕ)) + src_ϕ

        # Entropy: ½(γ-1)ϕu
        ds.ϕu .= @. -1 // 2 * ($∂x(ϕuu) + u * ∂x_ϕu + 4q * ϕ⁻¹ * ∂x_q) + src_ϕu

        # Entropy: q
        ds.q .= @. -1 // 2 * (γ * $∂x(qu) + (2 - γ) * u * ∂x_q) + src_q

        add_dissipation!(ds, (s, ϕ⁻¹, u))
    end
end

function make_dissipation(eqs::NordstromCompEuler1D, ::FluxEntropyStable, grid, ∂xs)
    @unpack γ = eqs
    sqrtγ = √γ
    (ds, (s, ϕ⁻¹, u)) -> begin
        @unpack ϕ, ϕu, q = s
        λϕ = 1 // 4 * maximum(zip(ϕu, q)) do (ϕu, q)
            abs(ϕu) + sqrtγ * abs(q)
        end
        λϕu = 1 // 2 * maximum(zip(ϕ, ϕu, q)) do (ϕ, ϕu, q)
            ϕ * (abs(ϕu) + sqrtγ * abs(q))
        end
        λq = 1 // 4 * maximum(zip(ϕ⁻¹, u, q)) do (ϕ⁻¹, u, q)
            abs(u) + sqrtγ * abs(q * ϕ⁻¹)
        end

        ds.ϕ  .+= @. -λϕ * $∂xs(ϕ) * ϕ⁻¹
        ds.ϕu .+= @. -(λϕu * ϕ⁻¹ - λϕ) * $∂xs(u) - λϕ * $∂xs(ϕu) * ϕ⁻¹
        ds.q  .+= @. -λq * $∂xs(q)
    end
end

#########################
# 2D Compressible Euler #
#########################
@kwdef struct CompEulerFluxForm2D{T <: Real, Splitting} <: CompressibleEuler2DScheme
    γ::T                      = 1.4
    flux_splitting::Splitting = nothing
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
@kwdef struct NordstromCompEuler2D{T <: Real, Splitting} <: CompressibleEuler2DScheme
    γ::T                      = 1.4
    flux_splitting::Splitting = nothing
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
@kwdef struct ReissSesterhennCompEuler2D{T <: Real, Splitting} <: CompressibleEuler2DScheme
    γ::T                      = 1.4
    flux_splitting::Splitting = nothing
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

"""
Meng-Sing Liou and Chris J. Steffen Jr.
High-Order Polynomial Expansions (HOPE) for flux-vector splitting.
https://ntrs.nasa.gov/citations/19910016425
"""
@kwdef struct VanLeerHanelCompEuler2D{T <: Real} <: CompressibleEuler2DScheme
    γ::T = 1.4
end

function from_primitive_vars(info::VanLeerHanelCompEuler2D, (ϱ, u, v, p))
    ϱe = @. p / (info.γ - 1) + 1 / 2 * ϱ * (u^2 + v^2)
    return NamedArrayPartition(ϱ = ϱ, ϱu = ϱ .* u, ϱv = ϱ .* v, ϱe = ϱe)
end

function to_primitive_vars!(dst, info::VanLeerHanelCompEuler2D, state)
    @unpack ϱ, ϱu, ϱv, ϱe = state
    dst[1] .= ϱ
    dst[2] .= @. ϱu / ϱ
    dst[3] .= @. ϱv / ϱ
    dst[4] .= @. (info.γ - 1) * (ϱe - 1 / 2 * (ϱu^2 + ϱv^2) / ϱ)
    return ()
end
