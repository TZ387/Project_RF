using prrf
using Logging
using Test

disable_logging(Logging.Info)  # hide the solvers' progress messages

@testset "prrf" begin
    include("grid.jl")
    include("rf_solver.jl")
    include("heat_solver.jl")
    include("simulation_io.jl")
    include("gpu.jl")
    include("examples.jl")
end
