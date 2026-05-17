using DrWatson
@quickactivate :HyperbolicPDEs
using CairoMakie
using LaTeXStrings

include(srcdir("plotting-utils.jl"))
include(srcdir("makie-theme.jl"))

savedir = plotsdir("comp-euler-2D", "kelvin-helmholtz")

const schemes = [
    (; label   = "DGSEM/SBP FD",
        marker  = :utriangle,
        color   = Makie.wong_colors()[7]
    ),
    (; label   = "DP DG/FD",
        marker  = :+,
        color   = Makie.wong_colors()[2]
    ),
    (; label   = "Lax-Friedrichs",
        marker  = :circle,
        color   = Makie.wong_colors()[1]
    ),
    (; label   = "van Leer-Hänel",
        marker  = :rect,
        color   = Makie.wong_colors()[4]
    )
]

As = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99]

# ϵ[scheme][A]

# FD
# 6th order
final_times = [
    [12.314341978445944, 4.953772728081835, 3.744391254117227, 3.0014309691844643, 2.822070569007186, 2.8477530873311143, 1.6144285184820544, 1.0459317610803456],
    [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 3.3920853440353516],
    [15.0, 15.0, 4.666377285605431, 3.1667667301025735, 2.902219261456314, 3.2320612321953788, 2.509519741171137, 1.5729523929300122],
    [15.0, 15.0, 4.353701764824227, 4.0482609722434155, 3.165281228963701, 2.347927369106959, 2.469331105781397, 1.0025995251819746]
]

# 7th order
# final_times = [
#     [ 8.977857504424772, 5.496151906116563, 3.9965998523415984, 2.9704190152252123, 2.6444900704396037, 2.896870552218041, 1.3708738427810874, 1.171663158663441, ],
#     [ 15.0, 15.0, 15.0, 15.0, 15.0, 5.715928044274637, 15.0, 3.366826043195796 ],
#     [ 15.0, 4.774862338063525, 4.09818980585383, 3.0493657623681223, 2.889615808147023, 3.2850747506430205, 2.5112863863513746, 0.9966941635845465, ],
#     [ 15.0, 4.752664454338926, 4.00685466943632, 2.455721329172344, 2.779218159631255, 2.305769914928343, 2.420165722726177, 0.876307147340529 ]
# ]

# DG
# final_times = [
#     [ 5.81633879159883, 3.824927829979527, 3.171350490679567, 2.7275740915400553, 2.562477966090536, 2.0898333827834996, 0.9095258390434285, 0.683657342797158 ],
#     [ 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0 ],
#     [ 15.0, 4.465012891849024, 3.4546317143101937, 2.7425754751708125, 1.905940792644678, 1.564987005739462, 1.0908083504560266, 0.6090013297705364 ],
#     [ 15.0, 4.782877253604791, 3.5783555916252596, 3.1675010212597274, 1.9099053184843058, 2.0803446831634695, 0.9908622713974273, 0.594905939557283 ],
# ]

fig = Figure()
ax = Axis(fig[1,1];
           xlabel=L"1 - A", ylabel="final time",
           # xticks=(
           #     [0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8],
           #     ["0.99", "0.975", "0.95", "0.9", "0.8", "0.6", "0.2"]
           # ),
           xscale=log10, xminorgridvisible=true,
           xminorticks=IntervalsBetween(4),
           # yticks=[1,2,5,15],
           # yscale=log10, yminorgridvisible=true,
           # yminorticks=IntervalsBetween(4),
           limits=(nothing,(-0.75,15.75)),
           width=210, height=130)

for (scheme, final_time) in zip(schemes, final_times)
    lines!(ax, 1 .- As, final_time;
        label=scheme.label, color=scheme.color)
    scatter!(ax, 1 .- As, final_time;
        label=scheme.label, color=scheme.color, marker=scheme.marker)
end
lines!(1 .- As[[begin,end]], [15, 15]; color=:black, linestyle=:dash, label=L"T_{\mathrm{final}}")

resize_to_layout!(fig)
wsave(joinpath(savedir, "atwood.pdf"), fig)

save_legend(joinpath(savedir, "atwood-legend.pdf"), ax, :horizontal)
