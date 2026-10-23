# GridSetup.jl
module GridSetup

using Ferrite
using ..Config

export setup_grid

# Define the domain and resolution using a parameter structure
function setup_grid(grid_params::Config.GridParams)
    lx, ly, lz = grid_params.lx, grid_params.ly, grid_params.lz
    nx, ny, nz = grid_params.nx, grid_params.ny, grid_params.nz
    
    P1 = Tensor{1,3,Float64}((-lx/2, -ly/2, 0))
    P2 = Tensor{1,3,Float64}((lx/2, ly/2, lz))
    nels = (nx, ny, nz)
    
    grid = generate_grid(Hexahedron, nels, P1, P2)
    # z is depth into the tissue, so the z = 0 face is the top
    grid.facetsets["top"], grid.facetsets["bottom"] = grid.facetsets["bottom"], grid.facetsets["top"]
    return grid
end

end # module GridSetup
