# HeatSolver.jl
#
# Solves the classic (non-perfused) heat equation on the structured voxel grid:
#
#   VHC(x) · ∂T/∂t = ∇·(k(x) ∇T) + Q(x)
#
# Spatial discretisation: cell-centred finite differences on the uniform
# Cartesian grid that GridSetup produces.
#
# Time integration: explicit forward Euler, with Δt chosen automatically to
# satisfy the stability criterion:
#
#   Δt ≤ min_over_cells( VHC / Σ_faces(k_face / Δ²) )
#
# where k_face is the harmonic mean of the two cells sharing the face.
# A safety factor of 0.9 is applied.
#
# Boundary conditions: zero-flux (Neumann) on all six faces.
#
# ── CPU parallelisation ───────────────────────────────────────────────────────
#
# The solver is called from inside a Threads.@spawn task in TimelapseCreation.
# Threads.@threads is NOT safe when nested inside @spawn — it deadlocks.
#
# Instead, parallelism is achieved by manually spawning one Task per z-slice
# chunk using Threads.@spawn inside heat_step!, then waiting on all of them.  This is safe to call from any task or thread.
#
# Worker count is capped at physical CPU cores (Sys.CPU_THREADS ÷ 2).
# Hyperthreads share the FPU and give no benefit for dense FP arithmetic.
# The cap is also bounded by Threads.nthreads() — the Julia threadpool size.
#
# ── GPU pathway (future) ──────────────────────────────────────────────────────
#
# The hot kernel (heat_step!) is dispatched through an AbstractHeatBackend
# type.  Adding GPU support later means:
#   1. Define  struct CUDABackend <: AbstractHeatBackend end
#   2. Add a CuArray method for heat_step!
#   3. Pass  backend = CUDABackend()  to solve_heat_phase
# Nothing else in this file or in TimelapseCreation needs to change.

module HeatSolver

using ..Config

export compute_stable_dt, solve_heat_phase, cpu_backend

# ── Backend abstraction ───────────────────────────────────────────────────────

abstract type AbstractHeatBackend end

struct CPUBackend <: AbstractHeatBackend
    n_workers :: Int
end

"""
    cpu_backend() -> CPUBackend

Choose worker count at runtime: min(julia_threads, physical_cores).
Physical cores = Sys.CPU_THREADS ÷ 2 (excludes hyperthreads, which share
the FPU and provide no benefit for dense floating-point work).
"""
function cpu_backend()
    physical = max(1, Sys.CPU_THREADS ÷ 2)
    workers  = min(Threads.nthreads(), physical)
    return CPUBackend(workers)
end

# ── Helpers ───────────────────────────────────────────────────────────────────

@inline function harmonic_mean(a::Float64, b::Float64)
    s = a + b
    s == 0.0 && return 0.0
    return 2.0 * a * b / s
end

# Conductivity of the faces between neighbouring cells along x, y and z
# (harmonic mean of the two cells), divided by the squared grid spacing.
function face_conductances(k, grid_params)
    dx = grid_params.lx / grid_params.nx
    dy = grid_params.ly / grid_params.ny
    dz = grid_params.lz / grid_params.nz
    @views return harmonic_mean.(k[1:end-1, :, :], k[2:end, :, :]) ./ dx^2,
                  harmonic_mean.(k[:, 1:end-1, :], k[:, 2:end, :]) ./ dy^2,
                  harmonic_mean.(k[:, :, 1:end-1], k[:, :, 2:end]) ./ dz^2
end

# One explicit Euler step on the z-slices ls:  Tn = T + c·(∇·(k∇T) + Q),  c = Δt/VHC.
# Extracted so it can be called from any task without nesting @threads.
function _step_slices!(Tn, T, Q, c, kx, ky, kz, ls)
    nx, ny, nz = size(T)
    @inbounds for l in ls, j in 1:ny, i in 1:nx
        Tc   = T[i, j, l]
        flux = Q[i, j, l]
        i < nx && (flux += kx[i, j, l]   * (T[i+1, j, l] - Tc))
        i > 1  && (flux += kx[i-1, j, l] * (T[i-1, j, l] - Tc))
        j < ny && (flux += ky[i, j, l]   * (T[i, j+1, l] - Tc))
        j > 1  && (flux += ky[i, j-1, l] * (T[i, j-1, l] - Tc))
        l < nz && (flux += kz[i, j, l]   * (T[i, j, l+1] - Tc))
        l > 1  && (flux += kz[i, j, l-1] * (T[i, j, l-1] - Tc))
        Tn[i, j, l] = Tc + c[i, j, l] * flux
    end
end

# ── Stability ─────────────────────────────────────────────────────────────────

"""
    compute_stable_dt(heat_params, grid_params; safety=0.9) -> Float64

Return `safety` times the largest explicit-Euler Δt that cannot overshoot: the
minimum over all cells of VHC divided by the sum of the cell's face conductances.
"""
compute_stable_dt(heat_params::Config.HeatParams, grid_params::Config.GridParams; safety::Float64 = 0.9) =
    compute_stable_dt(heat_params.VHC, face_conductances(heat_params.k, grid_params)...; safety)

function compute_stable_dt(VHC, kx, ky, kz; safety::Float64 = 0.9)
    S = zeros(size(VHC))
    @views begin
        S[1:end-1, :, :] .+= kx;  S[2:end, :, :] .+= kx
        S[:, 1:end-1, :] .+= ky;  S[:, 2:end, :] .+= ky
        S[:, :, 1:end-1] .+= kz;  S[:, :, 2:end] .+= kz
    end
    return safety * minimum(VHC ./ S)
end

# ── Parallel kernel (CPU) ─────────────────────────────────────────────────────
#
# We partition the nz z-slices into n_workers contiguous chunks and spawn one
# Task per chunk.  Threads.@spawn is safe to call from inside another spawned
# task (unlike @threads).  No locks needed: each task writes to a disjoint
# range of Tn[:,:,ls].

function heat_step!(Tn::Array{Float64,3},
                    T::Array{Float64,3},
                    Q::Array{Float64,3},
                    c::Array{Float64,3},
                    kx, ky, kz,
                    backend::CPUBackend)
    nz = size(T, 3)
    @sync for ls in Iterators.partition(1:nz, cld(nz, backend.n_workers))
        Threads.@spawn _step_slices!(Tn, T, Q, c, kx, ky, kz, ls)
    end
end

# ── Phase solver ──────────────────────────────────────────────────────────────

"""
    solve_heat_phase(T_in, Q_source, heat_params, grid_params, t_phase;
                     n_update=0, update_cb=nothing, backend=cpu_backend()) -> T

Advance the temperature field over `t_phase` seconds with source `Q_source`.

`backend` selects the compute backend.  Defaults to `cpu_backend()`, which
reads Sys.CPU_THREADS at call time and logs the worker count once per phase.
Pass a future GPU backend here — nothing else in this function needs to change.
"""
function solve_heat_phase(T_in::Array{Float64,3},
                          Q_source::Array{Float64,3},
                          heat_params::Config.HeatParams,
                          grid_params::Config.GridParams,
                          t_phase::Float64;
                          n_update::Int  = 0,
                          update_cb      = nothing,
                          backend        = cpu_backend())

    T  = copy(T_in)
    Tn = similar(T)

    kx, ky, kz = face_conductances(heat_params.k, grid_params)

    Δt        = compute_stable_dt(heat_params.VHC, kx, ky, kz)
    num_steps = ceil(Int, t_phase / Δt)
    Δt        = t_phase / num_steps
    c         = Δt ./ heat_params.VHC

    @info "Heat phase: t_phase=$(t_phase) s, " *
          "Δt=$(round(Δt; sigdigits=4)) s, " *
          "steps=$num_steps, " *
          "workers=$(backend.n_workers)"

    update_steps = Set{Int}()
    if n_update > 0 && update_cb !== nothing
        for i in 1:n_update
            push!(update_steps, round(Int, i * num_steps / n_update))
        end
    end

    t = 0.0
    for step in 1:num_steps
        heat_step!(Tn, T, Q_source, c, kx, ky, kz, backend)
        T, Tn = Tn, T
        t += Δt
        if step in update_steps
            update_cb(T, t)
        end
    end

    return T
end

end # module HeatSolver