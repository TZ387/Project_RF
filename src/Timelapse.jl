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
using ..Config
using ..HeatSolver
using ..Plotting: plot_slices

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
    run_heat_timelapse(Qel, heat_params, grid_params; backend)

Run the multi-phase heat simulation on a worker thread while displaying a
live-updating 3D volumeslices plot (with X/Y/Z sliders) on the main thread.

The window stays fully responsive because the main thread never blocks on
the solver — it only services the GLMakie event loop and drains a Channel
of plot snapshots produced by the worker.
"""
function run_heat_timelapse(Qel::Array{Float64,3},
                            heat_params::Config.HeatParams,
                            grid_params::Config.GridParams;
                            backend = HeatSolver.CPUBackend())

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

    display(plot_slices(vol_obs, grid_params; title = title_obs, colormap = :thermal,
                        colorrange = crange_obs, ticks = GLMakie.Makie.automatic, label = "Temperature [°C]"))

    # ── Worker thread ─────────────────────────────────────────────────────────
    sim_task = Threads.@spawn begin
        T = HeatSolver.solve_heat_schedule(Qel, heat_params, grid_params; backend,
                update_cb = (T_curr, t_sim, label) -> begin
                    T_snap     = Array(T_curr)      # CPU copy on worker thread — safe
                    Tmin, Tmax = extrema(T_snap)
                    Tmax       = Tmax ≈ Tmin ? Tmin + 1.0 : Tmax
                    put!(snapshot_ch, PlotSnapshot(T_snap, t_sim, label, Tmin, Tmax))
                end)

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
        end
        sleep(0.05)
    end

    # Re-throw any exception that occurred on the worker, and return T_final.
    return fetch(sim_task)
end

end # module Timelapse