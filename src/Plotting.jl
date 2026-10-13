module Plotting

using GLMakie
using ..Config

export plot_slices, plot_field_directions, plot_graphs

# ── Slice plot ────────────────────────────────────────────────────────────────

function plot_slices(V, grid_params; fig = nothing, num_ticks = 5,
                     use_log_scale = false, title = "")
    if fig === nothing
        fig = Figure(size = (700, 700))
    end

    ax = LScene(fig[1, 1], show_axis=false)
    maximum_length = max(grid_params.lx, grid_params.ly, grid_params.lz)
    scale!(ax.scene, maximum_length/grid_params.lx, maximum_length/grid_params.ly, maximum_length/grid_params.lz)

    if !isempty(title)
        Label(fig[0, 1], title, fontsize=18, tellwidth=false)
    end

    x = LinRange(-grid_params.lx/2, grid_params.lx/2, grid_params.nx)
    y = LinRange(-grid_params.ly/2, grid_params.ly/2, grid_params.ny)
    z = LinRange(grid_params.lz, 0, grid_params.nz)

    V_plot     = use_log_scale ? log10.(V) : V
    cmin, cmax = extrema(V_plot)
    ticks      = range(cmin, cmax, length=num_ticks)

    colormap = :heat
    Colorbar(fig[1, 2], limits = (cmin, cmax), colormap = colormap, ticks=ticks)

    plt = volumeslices!(ax, x, y, z, V_plot, colormap=colormap, colorrange=(cmin, cmax))
    slice_controls!(fig[2, 1], plt, length.((x, y, z)))

    fig
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
        on(plt[Symbol(:update_, plane)][], slider.value)
        on(is_active -> plt[Symbol(:heatmap_, plane)][].visible = is_active, toggle.active)

        set_close_to!(slider, n ÷ 2)
        slider
    end
end

# ── Electric field direction plot ─────────────────────────────────────────────

# Plots uniform-length arrows showing the direction of the electric field on a
# subsampled 3D grid. Arrow length is constant — magnitude information is shown
# separately in the E_mag slice plot.
#
# arrow_density controls how many arrows appear along the longest physical axis.
# Each axis gets its own stride derived from a shared target physical spacing,
# so arrow density is visually uniform even on highly anisotropic grids.
function plot_field_directions(E_vec, grid_params; arrow_density = 6, title = "E-field Direction")
    fig = Figure(size = (700, 700))

    if !isempty(title)
        Label(fig[0, 1], title, fontsize=18, tellwidth=false)
    end

    x = LinRange(-grid_params.lx/2, grid_params.lx/2, grid_params.nx)
    y = LinRange(-grid_params.ly/2, grid_params.ly/2, grid_params.ny)
    z = LinRange(0, grid_params.lz, grid_params.nz)

    # aspect = (1,1,1) forces all three visual axes to the same length regardless
    # of data range, so a flat domain like Example 1 still looks cubic.
    # limits! sets the data extent so Axis3 knows what range to fit into that cube.
    # The z limits are reversed to put the "bottom" face on top, as in plot_slices.
    ax = Axis3(fig[1, 1], aspect = (1, 1, 1),
               xlabel = "x [m]", ylabel = "y [m]", zlabel = "z [m]")
    limits!(ax, minimum(x), maximum(x), minimum(y), maximum(y), maximum(z), minimum(z))

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
        ex, ey, ez = E_vec[1, ix, iy, iz], E_vec[2, ix, iy, iz], E_vec[3, ix, iy, iz]
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
                       num_ticks = maximum(material_indices), title = "Media Distribution")
    fig2 = plot_slices(Qel, grid_params, title = "Distribution of Qel [W/m³]")
    fig3 = plot_slices(E_mag, grid_params, title = "Distribution of E [V/m]")
    fig4 = plot_slices(V, Config.GridParams(
                           grid_params.lx, grid_params.ly, grid_params.lz,
                           grid_params.nx + 1, grid_params.ny + 1, grid_params.nz + 1),
                       title = "Distribution of V [V]")
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