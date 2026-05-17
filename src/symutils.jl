export SymUtils

"""
Contains various utilities used to generate MMS source terms.
"""
module SymUtils

using StaticArrays: setindex
using ForwardDiff
using LinearAlgebra

struct LazyField{Fn}
    fn::Fn
end

Broadcast.broadcastable(f::LazyField) = (f,)

(f::LazyField)(t, x::Tuple) = f.fn(t, x)

for op in [:(Base.:+), :(Base.:-), :(Base.:*), :(Base.:/), :(LinearAlgebra.:⋅)]
    @eval begin
        function $op(lhs::LazyField, rhs::LazyField)
            return LazyField((t, x) -> $op(lhs(t, x), rhs(t, x)))
        end
        function $op(lhs::LazyField, rhs::Real)
            return LazyField((t, x) -> $op(lhs(t, x), rhs))
        end
        function $op(lhs::Real, rhs::LazyField)
            return LazyField((t, x) -> $op(lhs, rhs(t, x)))
        end
    end
end

for op in [:(Base.sqrt), :(Base.:-)]
    @eval $op(f::LazyField) = LazyField((t, x) -> $op(f(t, x)))
end

∂t(f::Real) = zero(f)
∂t(f::LazyField) = LazyField((t, x) -> ForwardDiff.derivative(τ -> f(τ, x), t))

struct SpatialDerivative{D} end

Base.:*(∂xⁱ::SpatialDerivative, f) = ∂xⁱ(f)
LinearAlgebra.:⋅(∂xⁱ::SpatialDerivative, f) = ∂xⁱ(f)

(::SpatialDerivative)(f::Real) = zero(f)
function (::SpatialDerivative{i})(f::LazyField) where {i}
    LazyField((t, x) -> ForwardDiff.derivative(ξⁱ -> f(t, setindex(x, ξⁱ, i)), x[i]))
end

function make_source_modifier_from_syms(
        statevars::Tuple,
        source_terms::NamedTuple,
        grid
)
    xs = collect(grid)
    return (ds, s, t::Real, _...) -> begin
        @inbounds @simd ivdep for i in eachindex(xs)
            x = xs[i]
            for var in statevars
                getproperty(ds, var)[i] += getproperty(source_terms, var)(t, x)
            end
        end
    end
end

end
