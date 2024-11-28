using Makie
const hpdes_makie_theme = merge(
    theme_latexfonts(),
    Theme(
        fontsize = 12,
        Axis = (;
            xtickalign = 1,
            xminortickalign = 1,
            xminorticksvisible = true,
            ytickalign = 1,
            yminortickalign = 1,
            yminorticksvisible = true,
            spinewidth = 1/4,
            xgridwidth = 1/4,
            xminorgridwidth = 1/8,
            xtickwidth = 1/4,
            xminortickwidth = 1/4,
            ygridwidth = 1/4,
            yminorgridwidth = 1/8,
            ytickwidth = 1/4,
            yminortickwidth = 1/4
        ),
        Legend = (;
            framevisible = false
        ),
        Lines = (;
            linewidth = 1.5
        ),
        Heatmap = (;
            colormap = :inferno
        ),
        screen_config = (;
            px_per_unit = 4,
            pt_per_unit = 1
        )
    )
)
