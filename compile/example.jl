using DrWatson
@quickactivate

using StaticArrays
using LinearAlgebra
using SummationByPartsOperators
using OrdinaryDiffEqSSPRK

## Sample PDE workload

xspan = (-5.0, 5.0)
tspan = (0.0, 1.0)

opts = dict_list(Dict(
    :solver      => [SSPRK54(), SSPRK43()],
    :deriv  => [(Mattsson2017, [2,3,4,5,6,7,8,9]), (WilliamsDuru2024, [4,5,6,7])],
    :gridsize    => [64, 128],
))

for opt in opts
    for p in opt[:deriv][2]
        op = upwind_operators(opt[:deriv][1];
            derivative_order = 1,
            accuracy_order = p,
            xmin = xspan[begin],
            xmax = xspan[end],
            N = opt[:gridsize]
        )
        D = (op.plus + op.minus) / 2

        xs = grid(op)

        u₀ = @. exp(-2xs^2)

        dudt! = (du, u, p, t) -> mul!(du, D, u)

        prob = ODEProblem(dudt!, u₀, tspan)

        sol = solve(prob, opt[:solver]; dt = 0.1)
    end
end
