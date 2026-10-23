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
    # Exact grid-line coordinates; Ferrite interpolates them from the corners, which adds rounding noise
    xs, ys, zs = range(-lx/2, lx/2, nx + 1), range(-ly/2, ly/2, ny + 1), range(0, lz, nz + 1)
    grid.nodes .= vec([Node(Vec((x, y, z))) for x in xs, y in ys, z in zs])
    return grid
end

end # module GridSetup
