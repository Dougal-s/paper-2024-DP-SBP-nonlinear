using Makie
using CairoMakie
using DrWatson
using Statistics
using Printf
using Base.Iterators
using LaTeXStrings

function pows10_ytickformatter(xs)
    map(xs) do x
        if x == 0
            L"0"
        else
            str = @sprintf("%s10^{%d}",
                (x >= 0 ? "" : "-"),
                round(Int32, log10(abs(x))))
            str = replace(str, "-" => "-\\!")
            latexstring(str)
        end
    end
end

function pseudolog_tol(transition)
    return Makie.ReversibleScale(
        x -> sign(x) * log10(abs(x) / transition + 1),
        x -> sign(x) * transition * (exp10(abs(x)) - 1);
        limits = (0.0f0, 3.0f0),
        name = :pseudolog_tol
    )
end

function last_valid(isvalid, us)
    count = 0
    for entry in us
        isvalid(entry) || break
        count += 1
    end
    count
end

function pick_extrema(uss...)
    maxs = flatmap(uss) do us
        [maximum(u) for u in takewhile(contains_only_valid, us)]
    end |> collect
    mins = flatmap(uss) do us
        [minimum(u) for u in takewhile(contains_only_valid, us)]
    end |> collect
    max_avg = mean(Float64, maxs)
    max_std = stdm(maxs, max_avg)
    min_avg = mean(Float64, mins)
    min_std = stdm(mins, min_avg)

    return (
        minimum(filter(x -> x ≥ min_avg - 3 * min_std, mins)),
        maximum(filter(x -> x ≤ max_avg + 3 * max_std, maxs))
    )
end

function add_margin(interval; margin = 0.1)
    range = max(interval[2] - interval[1], 1e-5 * middle(interval...))
    return @. interval + (-margin, margin) * range
end

contains_only_valid(x) = all(isfinite ∘ Float16, x)

function plot_animation(
        filename::String,
        ts,
        xs,
        us...;
        duration_s::Float64 = 10.0,
        fps::Int            = 30,
        energy              = nothing,
        exact               = nothing,
        exact_energy        = nothing,
        yrange::Tuple       = (nothing, nothing),
        energy_range::Tuple = (nothing, nothing),
        ylabel              = "u",
        labels              = nothing,
        size                = (1280, 1080)
)
    print("Plotting '", filename, "'")
    # filter out invalid timesteps
    last = maximum(@. last_valid(contains_only_valid, eachslice(us; dims = 2)))

    duration_s = duration_s * (ts[last] - ts[begin]) / (ts[end] - ts[begin])

    us = map(u -> view(u, :, 1:last), us)
    @views ts = ts[1:last]
    isnothing(energy) || (@views energy = energy[1:last])

    fig = Figure(size = size)
    curr_step = Observable(1)
    curr_energies = Observable(Point2f[])
    curr_exact_energies = Observable(Point2f[])

    if !isnothing(energy)
        axenergy = Axis(fig[1, 1],
            ylabel = "energy",
            xlabel = "t",
            height = 240,
            title  = @lift "t=$(ts[$curr_step])"
        )
        lines!(axenergy, curr_energies)
        isnothing(exact_energy) || lines!(axenergy, curr_exact_energies, color = :black)
        xlims!(axenergy, ts[begin], ts[end])
        energy_extrema = add_margin(pick_extrema(energy))
        energy_range = (
            isnothing(energy_range[1]) ? energy_extrema[1] : energy_range[1],
            isnothing(energy_range[2]) ? energy_extrema[2] : energy_range[2]
        )
        ylims!(axenergy, energy_range)

        axstate = Axis(fig[2, 1], xlabel = "x", ylabel = ylabel)
    else
        axstate = Axis(
            fig[1, 1], xlabel = "x", ylabel = ylabel, title = @lift "t=$(ts[$curr_step])")
    end

    for (i, u) in enumerate(us)
        if isnothing(labels)
            lines!(axstate, xs, @lift(u[:, $curr_step]))
        else
            lines!(axstate, xs, @lift(u[:, $curr_step]), label = labels[i])
        end
    end
    isnothing(exact) ||
        lines!(axstate, xs, @lift(exact[$curr_step]), label = "Exact", color = :black)
    xlims!(axstate, xs[begin], xs[end])

    yextrema = pick_extrema(us...) |> add_margin
    yrange = (
        isnothing(yrange[1]) ? yextrema[1] : yrange[1],
        isnothing(yrange[2]) ? yextrema[2] : yrange[2]
    )
    ylims!(axstate, yrange...)

    if !isnothing(labels)
        axislegend()
    end

    num_steps = floor(Int, fps * duration_s)
    filepath = plotsdir(filename)
    mkpath(dirname(filepath))
    record(fig, filepath, round.(Int, range(1, length(ts), num_steps));
        framerate = fps) do step
        curr_step[] = step
        if !isnothing(energy)
            curr_energies[] = push!(curr_energies[], Point2f(ts[step], energy[step]))
        end
        if !isnothing(exact_energy)
            curr_exact_energies[] = push!(
                curr_exact_energies[], Point2f(ts[step], exact_energy[step]))
        end
    end
    print("\n")
end

function plot_animation_2d(
        filename,
        ts,
        xs,
        ys,
        us,
        duration_s   = 10,
        fps          = 30;
        energy       = nothing,
        zrange       = (nothing, nothing),
        energy_range = (nothing, nothing)
)
    print("Plotting '", filename, "'")
    # filter out invalid timesteps
    last = last_valid(contains_only_valid, us)
    isnothing(energy) || (last = min(last, last_valid(contains_only_valid, energy)))

    duration_s = duration_s * (ts[last] - ts[begin]) / (ts[end] - ts[begin])

    @views us = us[1:last]
    @views ts = ts[1:last]
    isnothing(energy) || (@views energy = energy[1:last])

    curr_step = Observable(1)
    curr_energies = Observable(Point2f[])

    fig = Figure(size = (1280, 1080))

    if !isnothing(energy)
        axenergy = Axis(fig[1, 1],
            ylabel = "energy",
            xlabel = "t",
            height = 240,
            title  = @lift "t=$(ts[$curr_step])"
        )
        lines!(axenergy, curr_energies)
        xlims!(axenergy, ts[begin], ts[end])
        energy_extrema = add_margin(pick_extrema(energy))
        energy_range = (
            isnothing(energy_range[1]) ? energy_extrema[1] : energy_range[1],
            isnothing(energy_range[2]) ? energy_extrema[2] : energy_range[2]
        )
        ylims!(axenergy, energy_range)

        ax = Axis3(fig[2, 1], xlabel = "x", ylabel = "y")
    else
        ax = Axis3(
            fig[1, 1], xlabel = "x", ylabel = "y", title = @lift "t=$(ts[$curr_step])")
    end

    zextrema = add_margin(pick_extrema(us))
    zrange = (
        isnothing(zrange[1]) ? zextrema[1] : zrange[1],
        isnothing(zrange[2]) ? zextrema[2] : zrange[2]
    )
    hm = surface!(ax, xs, ys, @lift(us[$curr_step]), colorrange = zrange)
    Colorbar(fig[isnothing(energy) ? 1 : 2, 2], hm)
    xlims!(ax, xs[begin], xs[end])
    ylims!(ax, ys[begin], ys[end])
    zlims!(ax, zrange)

    num_steps = floor(Int, fps * duration_s)
    filepath  = plotsdir(filename)
    mkpath(dirname(filepath))
    record(fig, filepath, round.(Int, range(1, length(ts), num_steps));
        framerate = fps) do step
        curr_step[] = step
        if !isnothing(energy)
            curr_energies[] = push!(curr_energies[], Point2f(ts[step], energy[step]))
        end
    end
    print("\n")
end

function plot_heatmap_animation(
        filename,
        ts,
        xs,
        ys,
        us;
        duration_s   = 10,
        fps          = 24,
        energy       = nothing,
        zrange       = (nothing, nothing),
        energy_range = (nothing, nothing)
)
    print("Plotting '", filename, "'")
    # filter out invalid timesteps
    last = last_valid(contains_only_valid, us)
    isnothing(energy) || (last = min(last, last_valid(contains_only_valid, energy)))

    duration_s = duration_s * (ts[last] - ts[begin]) / (ts[end] - ts[begin])

    @views us = us[1:last]
    @views ts = ts[1:last]
    isnothing(energy) || (@views energy = energy[1:last])

    curr_step = Observable(1)
    curr_energies = Observable(Point2f[])

    fig = Figure(size = (1280, 1080))

    if !isnothing(energy)
        axenergy = Axis(fig[1, 1],
            ylabel = "energy",
            xlabel = "t",
            height = 240,
            title  = @lift "t=$(ts[$curr_step])"
        )
        lines!(axenergy, curr_energies)
        xlims!(axenergy, ts[begin], ts[end])
        energy_extrema = add_margin(pick_extrema(energy))
        energy_range = (
            isnothing(energy_range[1]) ? energy_extrema[1] : energy_range[1],
            isnothing(energy_range[2]) ? energy_extrema[2] : energy_range[2]
        )
        ylims!(axenergy, energy_range)

        axstate = Axis(fig[2, 1], aspect = 1, xlabel = "x", ylabel = "y")
    else
        axstate = Axis(fig[1, 1], aspect = 1, xlabel = "x", ylabel = "y",
            title = @lift "t=$(ts[$curr_step])")
    end

    zextrema = add_margin(pick_extrema(us))
    zrange = (
        isnothing(zrange[1]) ? zextrema[1] : zrange[1],
        isnothing(zrange[2]) ? zextrema[2] : zrange[2]
    )
    hm = heatmap!(axstate, xs, ys, @lift(us[$curr_step]), colorrange = zrange)
    Colorbar(fig[isnothing(energy) ? 1 : 2, 2], hm)
    xlims!(axstate, xs[begin], xs[end])
    ylims!(axstate, ys[begin], ys[end])

    num_steps = floor(Int, fps * duration_s)
    filepath = plotsdir(filename)
    mkpath(dirname(filepath))
    record(fig, filepath, round.(Int, range(1, length(ts), num_steps));
        framerate = fps) do step
        curr_step[] = step
        if !isnothing(energy)
            curr_energies[] = push!(curr_energies[], Point2f(ts[step], energy[step]))
        end
    end
    print("\n")
end

function plot_error(filename, ts, errors, Ns; ylabel = L"$∥\text{error}∥_H$")
    filepath = plotsdir(filename)
    print("Plotting '", filename, "'")

    fig     = Figure()
    ax      = Axis(fig[1, 1], xlabel = L"$t$", ylabel = ylabel, yscale = log10)
    extrema = (Inf64, -Inf64)
    for (N, err) in zip(Ns, errors)
        combined_err = max.(1e-18, err)
        new_extrema  = exp.(add_margin(pick_extrema(log.(combined_err))))
        extrema      = (min(extrema[1], new_extrema[1]), max(extrema[2], new_extrema[2]))
        lines!(ax, ts, combined_err, label = "N = $N")
    end
    xlims!(ax, add_margin((ts[begin], ts[end]); margin = 0.05))
    ylims!(ax, extrema)

    axislegend(position = :rb)

    wsave(filepath, fig; pt_per_unit = 1)
    print("\n")
end

function plot_convergence(
        filename, labels, errors, Δxs;
        figargs = (),
        colors = fill(nothing, length(labels)),
        markers = fill(nothing, length(labels)),
        legendname = nothing
)
    filepath = isabspath(filename) ? filename : plotsdir(filename)
    print("Plotting '", filename, "'")

    fig = Figure(; figargs...)
    Label(fig[1, 1], L"$\log_2∥\text{error}∥_H$"; rotation = π / 2, tellheight = false)
    ax = Axis(fig[1, 2], xlabel = L"$\log_2\,\Delta x$")
    for (label, color, marker, err) in zip(labels, colors, markers, errors)
        # line of best fit
        m         = isfinite.(err)
        line      = hcat(log2.(Δxs), ones(length(Δxs)))[m, :] \ log2.(err[m])
        linestart = (log2(Δxs[begin]), line[1] * log2(Δxs[begin]) + line[2])
        linestop  = (log2(Δxs[end]), line[1] * log2(Δxs[end]) + line[2])

        opts = (; label = label)
        isnothing(color) || (opts = (opts..., color = color))
        isnothing(marker) || (opts = (opts..., marker = marker))
        scatter!(ax, log2.(Δxs), log2.(err); opts...)

        opts = (; linestyle = :dash)
        isnothing(color) || (opts = (opts..., color = color))
        lines!(ax,
            [linestart, linestop];
            # label = latexstring(@sprintf "\$y = %.2fx %+.1f\$" line[1] line[2])
            opts...
        )
    end
    # axislegend(position = :lt)
    # For putting the legend below the plot
    # Legend(fig[2,1:2], ax; tellwidth=false, nbanks = 3)
    # rowsize!(fig.layout, 2, Fixed(60))
    wsave(filepath, fig)
    if !isnothing(legendname)
        legendpath = isabspath(legendname) ? legendname : plotsdir(legendname)
        fig = Figure()
        Legend(fig[1, 1], ax, halign = :center, orientation = :horizontal,
            tellwidth = true, tellheight = true)
        resize_to_layout!(fig)
        wsave(legendpath, fig)
    end
    print("\n")
end

function plot_globals_change_over_time(outdir, qty_labels, globals)
    print("Plotting '", outdir, "'")
    lins = []
    for quantity in keys(qty_labels)
        fig = Figure(size = (400, 200))
        ax = Axis(fig[1, 1],
            ylabel = qty_labels[quantity],
            xlabel = "time"
        )

        lins = []
        for (name, g) in pairs(globals)
            q0 = g[quantity][1]
            qs = (g[quantity] .- q0) ./ q0
            push!(lins, lines!(ax, g[:time], qs, label = name))
        end
        wsave(joinpath(outdir, "global-$quantity-change.png"), fig)
    end

    fig = Figure(size = (150, 20 + 20 * length(qty_labels)))
    Legend(fig[1, 1], lins, collect(keys(globals)); tellheight = false)
    wsave(joinpath(outdir, "global-change-legend.png"), fig)

    print("\n")
end

function plot_globals_over_time(filename, qty_labels, globals)
    print("Plotting '", filename, "'")
    fig = Figure()
    ax = ()
    for (i, quantity) in enumerate(keys(qty_labels))
        Label(fig[i, 1], qty_labels[quantity], rotation = π / 2, tellheight = false)
        ax = Axis(fig[i, 2], xlabel = "time")
        if i ≠ length(keys(qty_labels))
            hidexdecorations!(ax, grid = false)
        end

        yrange = (Inf64, -Inf64)
        for (name, g) in pairs(globals)
            lines!(ax, g[:time], g[quantity], label = name)
            yrange = (
                min(yrange[1], minimum(g[quantity])),
                max(yrange[2], maximum(g[quantity]))
            )
        end
        yrange = add_margin(yrange)
        ylims!(ax, yrange)
    end

    Legend(fig[1:length(qty_labels), 3], ax; tellheight = false)

    wsave(plotsdir(filename), fig)
    print("\n")
end
