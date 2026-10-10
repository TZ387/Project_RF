module prrf

include(joinpath(@__DIR__, "Config.jl"))
include(joinpath(@__DIR__, "GridSetup.jl"))
include(joinpath(@__DIR__, "RFSolver.jl"))
include(joinpath(@__DIR__, "HeatSolver.jl"))
include(joinpath(@__DIR__, "Plotting.jl"))
include(joinpath(@__DIR__, "Timelapse.jl"))
include(joinpath(@__DIR__, "RunSimulation.jl"))
include(joinpath(@__DIR__, "SimulationIO.jl"))

using .Config
using .GridSetup
using .RFSolver
using .HeatSolver
using .Timelapse
using .Plotting
using .RunSimulation
using .SimulationIO

export GridParams, RFParams, HeatParams, setup_material_properties, create_coordinate_grids
export setup_grid
export solve_rf, calculate_fields, convert_V
export compute_stable_dt, solve_heat_phase
export run_heat_timelapse
export plot_slices, plot_field_directions, plot_graphs
export run_simulation, run_heat_simulation
export save_simulation, load_simulation

end