using Makie
const hpdes_makie_theme = let
    presentation = false

    inch = 96
    pt = 4/3
    cm = inch / 2.54

    w1 = 0.5pt
    w2 = 0.3pt
    merge(
        theme_latexfonts(),
        Theme(
            fontsize = (9 + 1presentation)pt,
            figure_padding = 9pt,
            Axis = (;
                titlefont = :regular,
                xticklabelsize     = (8 + 2presentation)pt,
                yticklabelsize     = (8 + 2presentation)pt,
                xtickalign         = 1,
                ytickalign         = 1,
                xminortickalign    = 1,
                yminortickalign    = 1,
                xminorticksvisible = true,
                yminorticksvisible = true,
                spinewidth         = w1,
                xgridwidth         = w1,
                ygridwidth         = w1,
                xminorgridwidth    = w2,
                yminorgridwidth    = w2,
                xtickwidth         = w1,
                ytickwidth         = w1,
                xticksize          = 4,
                yticksize          = 4,
                xminortickwidth    = w2,
                yminortickwidth    = w2,
                xminorticksize     = 2,
                yminorticksize     = 2,
            ),
            Legend = (;
                labelsize = (8 + 2presentation)pt,
                framevisible = false
            ),
            Lines = (;
                linewidth = 1pt
            ),
            Scatter = (;
                markersize = 7pt
            ),
            Heatmap = (;
                colormap = :inferno
            ),
            Colorbar = (;
                ticklabelsize = 8pt,
                spinewidth    = w1,
                ticksize      = 4,
                tickwidth     = w1,
                size          = 7pt
            )
        )
    )
end
set_theme!(hpdes_makie_theme)
