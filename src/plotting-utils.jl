using Makie
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
        x -> sign(x) * log1p(abs(x) / oftype(x, transition)),
        x -> sign(x) * oftype(x, transition) * expm1(abs(x));
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

function save_legend(filepath, ax, orientation)
    fig = Figure(; figure_padding = 5)
    Legend(fig[1, 1], ax;
           halign = :center, orientation,
           merge = true, unique = true,
           tellwidth = true, tellheight = true)
    resize_to_layout!(fig)
    wsave(filepath, fig)
end
