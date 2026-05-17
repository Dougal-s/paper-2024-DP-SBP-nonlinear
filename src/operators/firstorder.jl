using LinearAlgebra
using SummationByPartsOperators
using StaticArrays

export FirstOrderDP

struct FirstOrderDP end

function local_dp_operator(
        T::Type,
        coefficient_source::FirstOrderDP,
        accuracy_order::Int,
        Ns::NTuple{Dims, Int}
) where {Dims}
    return ntuple(Val(Dims)) do i
        xs = range(zero(T), oneunit(T), Ns[i])
        left_boundary_plus = (SummationByPartsOperators.DerivativeCoefficientRow{T, 1, 2}(SVector(T(-2), T(2))),)
        right_boundary_plus = (SummationByPartsOperators.DerivativeCoefficientRow{T, 1, 2}(SVector(T(0), T(0))),)
        upper_coef_plus = SVector(T(1),)
        central_coef_plus = T(-1)
        lower_coef_plus = SVector{0, T}()
        left_weights = SVector(T(1 // 2),)
        right_weights = left_weights
        left_boundary_derivatives = Tuple{}()
        right_boundary_derivatives = left_boundary_derivatives

        left_boundary_minus = (SummationByPartsOperators.DerivativeCoefficientRow{T, 1, 2}(SVector(T(0), T(0))),)
        right_boundary_minus = (SummationByPartsOperators.DerivativeCoefficientRow{T, 1, 2}(SVector(T(2), T(-2))),)

        upper_coef_minus = .-lower_coef_plus
        central_coef_minus = .-central_coef_plus
        lower_coef_minus = .-upper_coef_plus

        left_boundary_central = (left_boundary_plus .+ left_boundary_minus) ./ 2
        right_boundary_central = (right_boundary_plus .+ right_boundary_minus) ./ 2
        upper_coef_central = SummationByPartsOperators.widening_plus(upper_coef_plus, upper_coef_minus) / 2
        central_coef_central = (central_coef_plus + central_coef_minus) / 2
        lower_coef_central = SummationByPartsOperators.widening_plus(lower_coef_plus, lower_coef_minus) / 2

        D₋ = DerivativeOperator(
            SummationByPartsOperators.DerivativeCoefficients(
                left_boundary_minus, right_boundary_minus,
                left_boundary_derivatives, right_boundary_derivatives,
                lower_coef_minus, central_coef_minus, upper_coef_minus,
                left_weights, right_weights,
                SummationByPartsOperators.FastMode(), 1, 1, nothing),
            xs
        )
        D = DerivativeOperator(
            SummationByPartsOperators.DerivativeCoefficients(
                left_boundary_central, right_boundary_central,
                left_boundary_derivatives, right_boundary_derivatives,
                lower_coef_central, central_coef_central, upper_coef_central,
                left_weights, right_weights,
                SummationByPartsOperators.FastMode(), 1, 1, nothing),
            xs
        )
        D₊ = DerivativeOperator(
            SummationByPartsOperators.DerivativeCoefficients(
                left_boundary_plus, right_boundary_plus,
                left_boundary_derivatives, right_boundary_derivatives,
                lower_coef_plus, central_coef_plus, upper_coef_plus,
                left_weights, right_weights,
                SummationByPartsOperators.FastMode(), 1, 1, nothing),
            xs
        )

        UpwindOperators(D₋, D, D₊)
    end
end
