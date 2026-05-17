using Makie

function Makie.convert_arguments(
        ::PointBased,
        cell,
        xs::TensorProdGrid{T, 1},
        ys::Vector,
        oversample::Int
) where {T}
    ps = Point2f[]
    sizehint!(ps, oversample * length(xs))
    foreachelement(xs) do I
        xsᴵ = only.(xs[I])
        ysᴵ = ys[I]
        xs_over = interpolate_onto(
            1:(1 // oversample):length(xsᴵ), 1:1:length(xsᴵ), xsᴵ)
        ys_over = interpolate_onto(cell[1], xs_over, xsᴵ, ysᴵ)
        append!(ps, Point2f.(xs_over, ys_over))
        push!(ps, Point2f(NaN, NaN))
    end
    return (ps,)
end

function Makie.convert_arguments(
        ::PointBased,
        xs::TensorProdGrid{T, 1},
        ys::Vector
) where {T}
    ps = Point2f[]
    sizehint!(ps, length(xs))
    foreachelement(xs) do I
        append!(ps, Point2f.(only.(xs[I]), ys[I]))
        push!(ps, Point2f(NaN, NaN))
    end
    return (ps,)
end
