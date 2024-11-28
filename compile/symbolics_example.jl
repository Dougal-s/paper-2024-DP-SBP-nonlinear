using DrWatson
@quickactivate

using Symbolics

@variables t x

u = 1 + 0.1 * sinpi(t + x)
∂t = Differential(t)
∂x = Differential(x)
(Symbolics.toexpr ∘ simplify ∘ expand_derivatives)(∂t(u) + u * ∂x(u))
