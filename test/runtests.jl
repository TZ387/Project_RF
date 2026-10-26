using prrf
using Logging
using Test

disable_logging(Logging.Info)  # hide the solvers' progress messages

const ε0 = 8.854e-12  # permittivity of free space [F/m]

@testset "prrf" begin
    include("grid.jl")
    include("rf_solver.jl")
    include("heat_solver.jl")
    include("simulation_io.jl")
    include("gpu.jl")
    include("examples.jl")
end
