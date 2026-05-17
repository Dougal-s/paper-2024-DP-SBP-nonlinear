# Plots the convergence results stored in <results_file.jl>. See burgers-results-dg.jl for
# an example of a results file.
# Usage:
#     julia convergence.jl <results_file.jl>

using DrWatson
@quickactivate :HyperbolicPDEs
using CairoMakie
using LaTeXStrings

include(srcdir("plotting-utils.jl"))
include(srcdir("makie-theme.jl"))

include(ARGS[1])
hs  = @. L / Ks

fileprefix, xlabel, xs, conv_rate = if length(Ks) == 1
    "fd", L"Δx", (@. hs / (Ns - 1)),
        (L"O\left(Δx^{p/2+1}\right)", p -> floor(Int, p / 2) + 1)
else
    "dg", L"h", hs,
        (L"O\left(h^p\right)", p -> p)
end

fig = Figure(; figure_padding = 8)

ax = Axis(fig[1, 1];
    xscale = log2, yscale = log10,
    yminorticks = IntervalsBetween(5),
    yminorgridvisible = true,
    xlabel = L"Δx", ylabel = L"$L^2$ error",
    width = 160, height = 130,
    limits = (nothing, ylimits)
)

cycle = Cycle([:color, :marker], covary = true)
colorrange = (-1, 1) .+ extrema(first, ϵs)
for (p, ϵ) in ϵs
    scatterlines!(ax, xs, ϵ; label = L"p=%$p", cycle, color = p,
        colorrange, colormap = :inferno)
end
for (p, ϵ) in ϵs
    e_x = [xs[end - 1] / 1.08, xs[end] * 1.05]
    e_O = conv_rate[2](p)
    lines!(ax, e_x, 1.3ϵ[end - 1] * (e_x ./ xs[end - 1]) .^ e_O;
        color = :black, linestyle = (:dash, :dense),
        label = conv_rate[1])
end
Legend(fig[1, 2], ax; rowgap=0, merge = true, unique = true)
# Legend(fig[0, 1], ax;
#     merge = true, unique = true,
#     tellheight = true, tellwidth = false,
#     orientation = :horizontal, nbanks = 2
# )

colgap!(fig.layout, 5)
rowgap!(fig.layout, 5)
resize_to_layout!(fig)

wsave(joinpath(savedir, "$fileprefix-convergence.pdf"), fig)
