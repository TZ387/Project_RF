module RunSimulation

using CUDA
using ..GridSetup
using ..RFSolver
using ..HeatSolver
using ..Timelapse

export run_simulation, run_heat_simulation

check_gpu(use_gpu) = use_gpu && !CUDA.functional() && error("use_gpu = true, but CUDA.jl found no usable NVIDIA GPU")

function run_simulation(grid_params, rf_params, boundary_conditions; use_gpu::Bool = false)

    check_gpu(use_gpu)

    n_cells = grid_params.nx * grid_params.ny * grid_params.nz
    @info "Setting up grid ($(grid_params.nx)×$(grid_params.ny)×$(grid_params.nz), $n_cells cells)..."
    grid = GridSetup.setup_grid(grid_params)

    @info "Solving RF problem$(use_gpu ? " on the GPU" : "")..."
    V_dof, dh = solve_rf(grid, rf_params, grid_params, boundary_conditions; use_gpu)
    V = convert_V(V_dof, dh, grid_params)

    @info "Computing electric field and thermal dissipation..."
    Qel, E_mag, E_vec = calculate_fields(V, rf_params, grid_params)

    return grid, V_dof, Qel, E_mag, E_vec, V
end

function run_heat_simulation(Qel, grid_params, heat_params;
                        create_timelapse::Bool = false, use_gpu::Bool = false)

    check_gpu(use_gpu)
    backend = use_gpu ? HeatSolver.CUDABackend() : HeatSolver.CPUBackend()

    if create_timelapse
        @info "Running heat simulation with live plot..."
        return Timelapse.run_heat_timelapse(Qel, heat_params, grid_params; backend)
    end

    @info "Running heat simulation (no live plot)..."
    return HeatSolver.solve_heat_schedule(Qel, heat_params, grid_params; backend)
end

end  # module RunSimulation