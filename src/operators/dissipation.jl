import SummationByPartsOperators as SBP
using SummationByPartsOperators
using StaticArrays

SBP.dissipation_operator(op::UpwindOperators) = (op.plus - op.minus) / 2

function SBP.dissipation_operator(
        op::UpwindOperators{T,
        P,
        <:DerivativeOperator,
        <:DerivativeOperator}
) where {T, P}
    cc = op.central.coefficients
    cm = op.minus.coefficients

    left_boundary = ntuple(Val(max(
        length(cc.left_boundary), length(cm.left_boundary)))) do i
        cc.left_boundary[i] + -cm.left_boundary[i]
    end
    right_boundary = ntuple(Val(max(
        length(cc.right_boundary), length(cm.right_boundary)))) do i
        cc.right_boundary[i] + -cm.right_boundary[i]
    end
    upper_coef = SBP.widening_plus(cc.upper_coef, -cm.upper_coef)
    central_coef = cc.central_coef - cm.central_coef
    lower_coef = SBP.widening_plus(cc.lower_coef, -cm.lower_coef)

    left_boundary_derivatives = Tuple{}()
    right_boundary_derivatives = left_boundary_derivatives

    left_weights = cc.left_weights
    right_weights = cc.right_weights

    coefs = SBP.DerivativeCoefficients(
        left_boundary, right_boundary,
        left_boundary_derivatives, right_boundary_derivatives,
        lower_coef, central_coef, upper_coef,
        left_weights, right_weights,
        cc.mode,
        1, # derivative order
        minimum(accuracy_order, (op.plus, op.minus, op.central)),
        source_of_coefficients(op)
    )
    return DerivativeOperator(coefs, grid(op))
end

function SBP.dissipation_operator(
        op::UpwindOperators{T,
        <:MatrixDerivativeOperator,
        C,
        <:MatrixDerivativeOperator}
) where {T, C}
    Di = (op.plus.D - op.minus.D) / 2
    return MatrixDerivativeOperator(
        SBP.xmin(op), SBP.xmax(op),
        grid(op),
        parent(mass_matrix(op)),
        Di,
        minimum(accuracy_order, (op.plus, op.minus, op.central)),
        source_of_coefficients(op)
    )
end
