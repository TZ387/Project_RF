# HeatSolver.jl
#
# Explicit finite-difference solver for VHC · ∂T/∂t = ∇·(k∇T) + Q with insulated boundaries
# (README, Section 1.2).
#
# During the live plot the solver runs inside a Threads.@spawn task, where Threads.@threads
# deadlocks, so heat_step! spawns one task per chunk of z-slices instead. With CUDABackend()
# the arrays move to the GPU and every cell is updated in its own GPU thread.

module HeatSolver

using CUDA
using ..Config

export compute_stable_dt, solve_heat_phase

# ── Backend abstraction ───────────────────────────────────────────────────────

struct CPUBackend end
struct CUDABackend end

Base.show(io::IO, ::CPUBackend) = print(io, Threads.nthreads(), " CPU threads")
Base.show(io::IO, ::CUDABackend) = print(io, "GPU")

device_array(::CPUBackend, A) = A
device_array(::CUDABackend, A) = CuArray(A)

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

# Temperature of cell (i, j, l) after one explicit Euler step:  T + c·(∇·(k∇T) + Q),  c = Δt/VHC.
# Shared by the CPU loop and the GPU kernel.
Base.@propagate_inbounds function stepped_temperature(T, Q, c, kx, ky, kz, i, j, l)
    nx, ny, nz = size(T)
    Tc   = T[i, j, l]
    flux = Q[i, j, l]
    i < nx && (flux += kx[i, j, l]   * (T[i+1, j, l] - Tc))
    i > 1  && (flux += kx[i-1, j, l] * (T[i-1, j, l] - Tc))
    j < ny && (flux += ky[i, j, l]   * (T[i, j+1, l] - Tc))
    j > 1  && (flux += ky[i, j-1, l] * (T[i, j-1, l] - Tc))
    l < nz && (flux += kz[i, j, l]   * (T[i, j, l+1] - Tc))
    l > 1  && (flux += kz[i, j, l-1] * (T[i, j, l-1] - Tc))
    return Tc + c[i, j, l] * flux
end

# One step on the z-slices ls. Extracted so it can be called from any task without nesting @threads.
function _step_slices!(Tn, T, Q, c, kx, ky, kz, ls)
    nx, ny, _ = size(T)
    @inbounds for l in ls, j in 1:ny, i in 1:nx
        Tn[i, j, l] = stepped_temperature(T, Q, c, kx, ky, kz, i, j, l)
    end
end

function heat_kernel!(Tn, T, Q, c, kx, ky, kz)
    n = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if n <= length(T)
        i, j, l = Tuple(CartesianIndices(T)[n])
        @inbounds Tn[i, j, l] = stepped_temperature(T, Q, c, kx, ky, kz, i, j, l)
    end
    return
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
# One task per chunk of z-slices; each writes its own slices of Tn, so no locks are needed.

function heat_step!(Tn::Array{Float64,3},
                    T::Array{Float64,3},
                    Q::Array{Float64,3},
                    c::Array{Float64,3},
                    kx, ky, kz,
                    ::CPUBackend)
    nz = size(T, 3)
    @sync for ls in Iterators.partition(1:nz, cld(nz, Threads.nthreads()))
        Threads.@spawn _step_slices!(Tn, T, Q, c, kx, ky, kz, ls)
    end
end

heat_step!(Tn::CuArray, T, Q, c, kx, ky, kz, ::CUDABackend) =
    @cuda threads = 256 blocks = cld(length(T), 256) heat_kernel!(Tn, T, Q, c, kx, ky, kz)

# ── Phase solver ──────────────────────────────────────────────────────────────

"""
    solve_heat_phase(T_in, Q_source, heat_params, grid_params, t_phase;
                     n_update=0, update_cb=nothing, backend=CPUBackend()) -> T

Advance the temperature field over `t_phase` seconds with source `Q_source`.

`backend` selects the compute backend: `CPUBackend()` (the default), which uses
all Julia threads, or `CUDABackend()`. The returned field is always a CPU array;
`update_cb` receives the field on the backend's device.
"""
function solve_heat_phase(T_in::Array{Float64,3},
                          Q_source::Array{Float64,3},
                          heat_params::Config.HeatParams,
                          grid_params::Config.GridParams,
                          t_phase::Float64;
                          n_update::Int  = 0,
                          update_cb      = nothing,
                          backend        = CPUBackend())

    T  = device_array(backend, copy(T_in))
    Tn = similar(T)

    kx, ky, kz = face_conductances(heat_params.k, grid_params)

    Δt        = compute_stable_dt(heat_params.VHC, kx, ky, kz)
    num_steps = ceil(Int, t_phase / Δt)
    Δt        = t_phase / num_steps
    c         = Δt ./ heat_params.VHC

    Q, c, kx, ky, kz = map(A -> device_array(backend, A), (Q_source, c, kx, ky, kz))

    @info "Heat phase: t_phase=$(t_phase) s, " *
          "Δt=$(round(Δt; sigdigits=4)) s, " *
          "steps=$num_steps, " *
          "on $backend"

    update_steps = Set{Int}()
    if n_update > 0 && update_cb !== nothing
        for i in 1:n_update
            push!(update_steps, round(Int, i * num_steps / n_update))
        end
    end

    t = 0.0
    for step in 1:num_steps
        heat_step!(Tn, T, Q, c, kx, ky, kz, backend)
        T, Tn = Tn, T
        t += Δt
        if step in update_steps
            update_cb(T, t)
        end
    end

    return Array(T)
end

# ── Schedule solver ───────────────────────────────────────────────────────────

# Run every phase of the schedule in turn, with rf_factor·Qel + Q_extra as the source.
# update_cb(T, t, label), if given, is called n_update times per phase.
function solve_heat_schedule(Qel, heat_params, grid_params; update_cb = nothing, backend = CPUBackend())
    T         = heat_params.T_initial
    t_elapsed = 0.0

    for (phase_idx, phase) in enumerate(heat_params.schedule)
        Q     = phase.rf_factor .* Qel .+ phase.Q_extra
        label = iszero(Q) ? "cooling" : "heating"
        @info "Phase $phase_idx/$(length(heat_params.schedule)): $label for $(phase.duration) s"

        t_off = t_elapsed   # captured by value for this phase's closure
        cb    = update_cb === nothing ? nothing :
                (T_curr, t_local) -> update_cb(T_curr, t_off + t_local, label)

        T = solve_heat_phase(T, Q, heat_params, grid_params, phase.duration;
                             n_update = heat_params.n_update, update_cb = cb, backend)

        @info "Phase $phase_idx complete.  Peak T = $(round(maximum(T); digits=2)) °C"
        t_elapsed += phase.duration
    end

    return T
end

end # module HeatSolver