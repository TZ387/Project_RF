module prrf

include("Config.jl")
include("GridSetup.jl")
include("RFSolver.jl")
include("HeatSolver.jl")
include("Plotting.jl")
include("Timelapse.jl")
include("RunSimulation.jl")
include("SimulationIO.jl")

using .Config
using .GridSetup
using .RFSolver
using .HeatSolver
using .Timelapse
using .Plotting
using .RunSimulation
using .SimulationIO

export GridParams, RFParams, HeatPhase, HeatParams, setup_material_properties, create_coordinate_grids
export setup_grid
export solve_rf, calculate_fields, convert_V
export compute_stable_dt, solve_heat_phase
export run_heat_timelapse
export plot_slices, plot_field_directions, plot_graphs
export run_simulation, run_heat_simulation
export save_simulation, load_simulation

end