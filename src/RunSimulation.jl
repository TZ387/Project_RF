module RunSimulation

using ..GridSetup
using ..RFSolver
using ..HeatSolver
using ..Timelapse
using ..Plotting

export run_simulation, run_heat_simulation

function run_simulation(grid_params, rf_params, boundary_conditions)

    n_cells = grid_params.nx * grid_params.ny * grid_params.nz
    @info "Setting up grid ($(grid_params.nx)×$(grid_params.ny)×$(grid_params.nz), $n_cells cells)..."
    grid = GridSetup.setup_grid(grid_params)

    @info "Solving RF problem..."
    V_dof, dh, cellValues = solve_rf(grid, rf_params, grid_params, boundary_conditions)

    @info "Computing electric field and thermal dissipation..."
    Qel, E_mag, E_vec = calculate_fields(cellValues, dh, V_dof, rf_params.sigma, grid_params)
    V = convert_V(V_dof, dh, grid_params)

    return grid, V_dof, Qel, E_mag, E_vec, V
end

function run_heat_simulation(Qel, grid_params, heat_params;
                        create_timelapse::Bool = false)

    if create_timelapse
        @info "Running heat simulation with live plot..."
        return Timelapse.run_heat_timelapse(Qel, heat_params, grid_params)
    end

    @info "Running heat simulation (no live plot)..."
    return HeatSolver.solve_heat_schedule(Qel, heat_params, grid_params)
end

end  # module RunSimulation