# Timelapse.jl
#
# Live-updating heat simulation visualisation using GLMakie observables.
#
# The simulation follows the arbitrary schedule defined in HeatParams.schedule —
# an ordered list of (:on/:off, duration) phases, e.g.:
#
#   [(:on, 30.0), (:off, 60.0), (:on, 15.0)]
#
# Threading model
# ───────────────
# The root cause of VS Code's "not responding" freeze is that the heavy solver
# loop and the GLMakie window event loop both run on the same (main) thread.
# The OS sees the thread busy for seconds at a time and marks the window
# unresponsive.
#
# Fix: the entire heat simulation runs on a worker thread (Threads.@spawn).
# The main thread never blocks on the solver — it sits in a lightweight
# polling loop that (a) drains incoming plot snapshots from a Channel and
# writes them to Observables, and (b) calls sleep(0.05) between polls to
# yield back to the GLMakie event loop.
#
# Why a Channel and not direct Observable writes from the worker?
# GLMakie requires all Observable assignments to happen on the main thread.
# Writing obs[] = ... from a worker thread causes race conditions and
# occasional crashes.  The Channel decouples the two threads cleanly:
# the worker puts a cheap snapshot struct into it; the main thread takes
# snapshots out and does all Observable writes itself.
#
# Layout:
#   worker thread → solve_heat_phase × N → put!(channel, snapshot)
#   main thread   → polling loop { take! + obs[] = ..., sleep(0.05) }

module Timelapse

using GLMakie
using Printf
using Base.Threads
using ..Config
using ..HeatSolver

export run_heat_timelapse

# ── Snapshot carried through the channel ─────────────────────────────────────

struct PlotSnapshot
    T     :: Array{Float64,3}  # full temperature field, copied on worker thread
    t_sim :: Float64            # simulation time [s]
    label :: String
    T_min :: Float64
    T_max :: Float64
end

# ── Main entry point ──────────────────────────────────────────────────────────

"""
    run_heat_timelapse(Qel, heat_params, grid_params)

Run the multi-phase heat simulation on a worker thread while displaying a
live-updating 3D volumeslices plot (with X/Y/Z sliders) on the main thread.

The window stays fully responsive because the main thread never blocks on
the solver — it only services the GLMakie event loop and drains a Channel
of plot snapshots produced by the worker.
"""
function run_heat_timelapse(Qel::Array{Float64,3},
                            heat_params::Config.HeatParams,
                            grid_params::Config.GridParams)

    nx, ny, nz = grid_params.nx, grid_params.ny, grid_params.nz
    Qzero = zeros(Float64, nx, ny, nz)

    x = LinRange(-grid_params.lx/2, grid_params.lx/2, nx)
    y = LinRange(-grid_params.ly/2, grid_params.ly/2, ny)
    z = LinRange(grid_params.lz, 0, nz)

    # ── Channel ───────────────────────────────────────────────────────────────
    # Capacity 4: small buffer so the worker is never held up long waiting for
    # the main thread to drain, but also doesn't queue stale frames.
    # Nothing (::Nothing) is the sentinel that signals simulation is done.
    snapshot_ch = Channel{Union{PlotSnapshot, Nothing}}(4)

    # ── Build the Makie scene on the main thread ──────────────────────────────
    T0 = heat_params.T_initial

    # Fixed limits when the user supplied both T_plot_min and T_plot_max;
    # otherwise fall back to auto-ranging from the live data.
    fixed_crange = heat_params.T_plot_min !== nothing &&
                   heat_params.T_plot_max !== nothing
    init_cmin = fixed_crange ? heat_params.T_plot_min : minimum(T0)
    init_cmax = fixed_crange ? heat_params.T_plot_max : maximum(T0) + 1.0

    vol_obs    = Observable(T0)
    crange_obs = Observable((init_cmin, init_cmax))
    title_obs  = Observable("t = 0.00 s")

    colormap = :thermal

    fig = Figure(size = (800, 750))

    Label(fig[0, 1], title_obs, fontsize = 18, tellwidth = false)

    ax = LScene(fig[1, 1], show_axis = false)
    maximum_length = max(grid_params.lx, grid_params.ly, grid_params.lz)
    scale!(ax.scene,
           maximum_length / grid_params.lx,
           maximum_length / grid_params.ly,
           maximum_length / grid_params.lz)

    plt = volumeslices!(ax, x, y, z, vol_obs;
                        colormap   = colormap,
                        colorrange = crange_obs)

    Colorbar(fig[1, 2];
             colormap = colormap,
             limits   = crange_obs,
             label    = "Temperature [°C]")

    # ── Sliders ───────────────────────────────────────────────────────────────
    slider_layout = GridLayout(fig[2, 1])

    slider_params = [
        (label = "x axis", range = 1:nx),
        (label = "y axis", range = 1:ny),
        (label = "z axis", range = 1:nz),
    ]

    sliders = []

    for (i, (label, range)) in enumerate(slider_params)
        left_btn = Button(slider_layout[i, 1], label = "◀", width = 30)

        slider = Slider(slider_layout[i, 2:4], range = range, startvalue = range[1])
        push!(sliders, slider)

        right_btn = Button(slider_layout[i, 5], label = "▶", width = 30)

        Label(slider_layout[i, 6], label, tellwidth = false, halign = :left)

        value_label = Label(slider_layout[i, 7], string(slider.value[]),
                            tellwidth = false, halign = :center, width = 50)
        on(slider.value) do val
            value_label.text = string(val)
        end

        on(left_btn.clicks) do _
            current_val = slider.value[]
            current_idx = findfirst(v -> v == current_val, range)
            if current_idx !== nothing && current_idx > 1
                set_close_to!(slider, range[current_idx - 1])
            end
        end

        on(right_btn.clicks) do _
            current_val = slider.value[]
            current_idx = findfirst(v -> v == current_val, range)
            if current_idx !== nothing && current_idx < length(range)
                set_close_to!(slider, range[current_idx + 1])
            end
        end
    end

    sl_yz, sl_xz, sl_xy = sliders

    on(sl_yz.value) do v; plt[:update_yz][](v) end
    on(sl_xz.value) do v; plt[:update_xz][](v) end
    on(sl_xy.value) do v; plt[:update_xy][](v) end

    set_close_to!(sl_yz, nx ÷ 2)
    set_close_to!(sl_xz, ny ÷ 2)
    set_close_to!(sl_xy, nz ÷ 2)

    # Visibility toggles (one per slice plane)
    hmaps   = [plt[Symbol(:heatmap_, s)][] for s ∈ (:yz, :xz, :xy)]
    toggles = [Toggle(slider_layout[i, 8], active = true) for i ∈ 1:3]
    for (hmap, toggle) in zip(hmaps, toggles)
        on(toggle.active) do is_active
            hmap.visible = is_active
        end
    end

    display(fig)

    # ── Worker thread ─────────────────────────────────────────────────────────
    sim_task = Threads.@spawn begin
        T         = copy(T0)
        t_elapsed = 0.0

        for (phase_idx, (state, duration)) in enumerate(heat_params.schedule)

            state in (:on, :off) || error(
                "Unknown phase state $(repr(state)) in schedule entry $phase_idx. " *
                "Expected :on or :off.")

            label = state == :on ? "heating" : "cooling"
            Q_src = state == :on ? Qel : Qzero

            @info "Phase $phase_idx/$(length(heat_params.schedule)): " *
                  "$label for $(duration) s  ($(heat_params.n_update) plot updates)"

            # t_elapsed captured by value for this phase's closure
            t_off = t_elapsed

            cb = (T_curr, t_local) -> begin
                T_snap = copy(T_curr)       # copy on worker thread — safe
                Tmin   = minimum(T_curr)
                Tmax   = maximum(T_curr)
                Tmax   = Tmax ≈ Tmin ? Tmin + 1.0 : Tmax
                put!(snapshot_ch,
                     PlotSnapshot(T_snap, t_off + t_local, label, Tmin, Tmax))
            end

            T = HeatSolver.solve_heat_phase(
                    T, Q_src, heat_params, grid_params, duration;
                    n_update  = heat_params.n_update,
                    update_cb = cb)

            @info "Phase $phase_idx complete.  " *
                  "Peak T = $(round(maximum(T); digits=2)) °C"

            t_elapsed += duration
        end

        put!(snapshot_ch, nothing)  # sentinel: tell main thread we are done
        return T
    end

    # ── Main thread polling loop ──────────────────────────────────────────────
    # Runs until the worker sends the nothing sentinel.
    # sleep(0.05) is the key: it yields to the GLMakie event loop every 50 ms,
    # which is more than enough for a responsive window.
    done = false
    while !done
        # Drain everything currently queued before sleeping
        while isready(snapshot_ch)
            snap = take!(snapshot_ch)
            if snap === nothing
                done = true
                break
            end
            vol_obs[]   = snap.T           # update the full 3D field
            if !fixed_crange
                crange_obs[] = (snap.T_min, snap.T_max)
            end
            title_obs[] = @sprintf("t = %.2f s  [%s]", snap.t_sim, snap.label)
            # volumeslices! doesn't re-extract slices automatically when the
            # volume observable changes — manually re-trigger each plane at
            # the current slider position so the display updates without the
            # user having to touch a slider.
            plt[:update_yz][](sl_yz.value[])
            plt[:update_xz][](sl_xz.value[])
            plt[:update_xy][](sl_xy.value[])
        end
        sleep(0.05)
    end

    # Re-throw any exception that occurred on the worker, and return T_final.
    return fetch(sim_task)
end

end # module Timelapse