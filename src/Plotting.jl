module Plotting

using GLMakie

export plot_slices, plot_field_directions, plot_graphs

# ── Slice plot ────────────────────────────────────────────────────────────────

# Positions of the data along x, y and z: cell centres for cell data, nodes for nodal data such
# as V. x and y are centred, z runs from 0 at the top face.
function data_coordinates(grid_params, dims)
    axis(l, n, m) = m == n ? range(l / 2n, l - l / 2n, n) : range(0, l, n + 1)
    return axis(grid_params.lx, grid_params.nx, dims[1]) .- grid_params.lx / 2,
           axis(grid_params.ly, grid_params.ny, dims[2]) .- grid_params.ly / 2,
           axis(grid_params.lz, grid_params.nz, dims[3])
end

# data may be an Observable, as in the live heat plot.
function plot_slices(data, grid_params; title = "", colormap = :heat, colorrange = extrema(data),
                     ticks = range(colorrange..., length = 5), label = "")
    fig = Figure(size = (700, 700))
    Label(fig[0, 1], title, fontsize = 18, tellwidth = false)

    ax = LScene(fig[1, 1], show_axis=false)
    maximum_length = max(grid_params.lx, grid_params.ly, grid_params.lz)
    scale!(ax.scene, maximum_length/grid_params.lx, maximum_length/grid_params.ly, maximum_length/grid_params.lz)

    x, y, z = data_coordinates(grid_params, size(to_value(data)))

    Colorbar(fig[1, 2]; colormap, limits = colorrange, ticks, label)
    plt = volumeslices!(ax, x, y, grid_params.lz .- z, data; colormap, colorrange)  # top face drawn on top
    slice_controls!(fig[2, 1], plt, length.((x, y, z)))
    return fig
end

# Adds a slider with arrow buttons, a value label and a visibility toggle for each
# slicing plane of the volumeslices plot `plt`. Returns the three sliders.
function slice_controls!(gridpos, plt, dims)
    layout = GridLayout(gridpos)

    return map(1:3, ("x", "y", "z"), (:yz, :xz, :xy), dims) do i, axis, plane, n
        left_btn  = Button(layout[i, 1], label = "◀", width = 30)
        slider    = Slider(layout[i, 2:4], range = 1:n, startvalue = 1)
        right_btn = Button(layout[i, 5], label = "▶", width = 30)
        Label(layout[i, 6], "$axis axis", tellwidth = false, halign = :left)
        Label(layout[i, 7], lift(string, slider.value), tellwidth = false, halign = :center, width = 50)
        toggle    = Toggle(layout[i, 8], active = true)

        on(_ -> set_close_to!(slider, slider.value[] - 1), left_btn.clicks)
        on(_ -> set_close_to!(slider, slider.value[] + 1), right_btn.clicks)
        on(i -> plt[Symbol(plane, :_index)] = i, slider.value)
        on(is_active -> plt[Symbol(:heatmap_, plane)][].visible = is_active, toggle.active)

        set_close_to!(slider, n ÷ 2)
        slider
    end
end

# ── Electric field direction plot ─────────────────────────────────────────────

# Plots uniform-length arrows showing the direction of the electric field on a
# subsampled 3D grid. Arrow length is constant — magnitude information is shown
# separately in the E_mag slice plot. Each arrow shows the field Re(E e^{-iωt}) of its cell at
# the instant it is strongest there, ωt = ½ arg E·E, pointing the same way as the field when the
# electrode potentials peak (t = 0).
#
# arrow_density is the number of arrows along each axis.
function plot_field_directions(E_vec, grid_params; arrow_density = 6, title = "E-field Direction")
    fig = Figure(size = (700, 700))

    if !isempty(title)
        Label(fig[0, 1], title, fontsize=18, tellwidth=false)
    end

    x, y, z = data_coordinates(grid_params, size(E_vec)[2:end])

    # aspect = (1,1,1) forces all three visual axes to the same length regardless
    # of data range, so a flat domain like Example 1 still looks cubic.
    # limits! sets the data extent so Axis3 knows what range to fit into that cube.
    # The z limits are reversed to draw the "top" face (z = 0) on top, as in plot_slices.
    ax = Axis3(fig[1, 1], aspect = (1, 1, 1),
               xlabel = "x [m]", ylabel = "y [m]", zlabel = "z [m]")
    limits!(ax, -grid_params.lx/2, grid_params.lx/2, -grid_params.ly/2, grid_params.ly/2, grid_params.lz, 0)

    # With aspect=(1,1,1) all axes are visually equal, so we use the same
    # number of arrow divisions along each axis regardless of physical size.
    stride_x = max(1, round(Int, grid_params.nx / arrow_density))
    stride_y = max(1, round(Int, grid_params.ny / arrow_density))
    stride_z = max(1, round(Int, grid_params.nz / arrow_density))

    xi = 1:stride_x:length(x)
    yi = 1:stride_y:length(y)
    zi = 1:stride_z:length(z)

    n = length(xi) * length(yi) * length(zi)
    ps = Vector{Point3f}(undef, n)
    ns = Vector{Vec3f}(undef, n)

    k = 0
    for iz in zi, iy in yi, ix in xi
        k += 1
        ps[k] = Point3f(x[ix], y[iy], z[iz])
        E = E_vec[:, ix, iy, iz]
        e = real.(E .* cis(-angle(sum(E .^ 2)) / 2))
        ex, ey, ez = sum(e .* real.(E)) < 0 ? -e : e
        m = sqrt(ex^2 + ey^2 + ez^2)
        ns[k] = m > 0 ? Vec3f(ex/m, ey/m, ez/m) : Vec3f(0f0, 0f0, 0f0)
    end

    min_l = min(grid_params.lx, grid_params.ly, grid_params.lz)
    arrow_scale = 0.9 * min_l / arrow_density
    arrows3d!(ax, ps, ns, color = :white, lengthscale = arrow_scale)

    fig
end

# ── Top-level plotting entry point ────────────────────────────────────────────

function plot_graphs(material_indices, grid_params, Qel, E_mag, E_vec, V, filename=nothing, arrow_density = 6)
    fig1 = plot_slices(material_indices, grid_params,
                       ticks = sort(unique(material_indices)), title = "Media Distribution")
    fig2 = plot_slices(Qel, grid_params, title = "Distribution of Qel [W/m³]")
    fig3 = plot_slices(E_mag, grid_params, title = "Distribution of E [V/m]")
    fig4 = plot_slices(abs.(V), grid_params, title = "Distribution of |V| [V]")
    fig5 = plot_field_directions(E_vec, grid_params; arrow_density, title = "E-field Direction")

    display(GLMakie.Screen(), fig1)
    display(GLMakie.Screen(), fig2)
    display(GLMakie.Screen(), fig3)
    display(GLMakie.Screen(), fig4)
    display(GLMakie.Screen(), fig5)

    if filename !== nothing
        mkpath(dirname(filename))
        save("$(filename)_a.png", fig1)
        save("$(filename)_b.png", fig2)
        save("$(filename)_c.png", fig3)
        save("$(filename)_d.png", fig4)
        save("$(filename)_e.png", fig5)
    end
end

end