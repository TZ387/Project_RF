# Include the main module - this loads all submodules
using prrf

# Geometry definition function
function geometryDefinition(grid_params::GridParams)
    # Get coordinate grids from Config module (implementation details hidden)
    X, Y, Z = create_coordinate_grids(grid_params)
    # Initialize every cell with material 1, skin
    M = ones(Int, grid_params.nx, grid_params.ny, grid_params.nz)

    # Apply conditions using absolute coordinates [m]
    M[Z .> 0.002] .= 2     # Fat (2 mm depth)
    M[Z .> 0.022] .= 3    # Muscle (22 mm depth)
    
    return M
end

# Define materials and their properties
function define_material_properties()
    materials = Dict{Int, NamedTuple}()
    
    # Relative permittivities; epsilon_re of the tissues at 1 MHz from [3] in the README
    materials[1] = (name = "Skin",   sigma = 0.22,  epsilon_re = 1833, epsilon_im = 0, VHC = 4.18e6, k = 0.6)
    materials[2] = (name = "Fat",    sigma = 0.025, epsilon_re = 27.2, epsilon_im = 0, VHC = 1.0e6, k = 0.5)
    materials[3] = (name = "Muscle", sigma = 0.5,   epsilon_re = 1836, epsilon_im = 0, VHC = 1.0e6, k = 0.5)
    
    return materials
end

function main()
    # Flags for use
    use_gpu = false            # Solve on an NVIDIA GPU (needs CUDA)

    # Define grid parameters
    grid_params = GridParams(
        0.09,    # lx = Length of the simulation domain in x-direction [m]
        0.09,    # ly = Length of the simulation domain in y-direction [m]
        0.1,     # lz = Length of the simulation domain in z-direction [m]
        180,     # nx = Number of elements in the x-direction
        180,     # ny = Number of elements in the y-direction
        200      # nz = Number of elements in the z-direction
    )

    # Load materials and geometry
    materials = define_material_properties()
    material_indices = geometryDefinition(grid_params)

    # Setup material properties (delegated to Config.jl)
    sigma, epsilon_re, epsilon_im, VHC, k = setup_material_properties(material_indices, materials)

    # Load RF parameters
    rf_params = RFParams(
        sigma,      # Electrical conductivity matrix [S/m]
        epsilon_re, # Real part of the relative permittivity ε'/ε0
        epsilon_im, # Imaginary part of the relative permittivity ε''/ε0
        2π * 1e6,   # ω = Angular frequency of the RF signal [rad/s], f = 1 MHz
    )


    boundary_conditions = Dict(
        "bottom" => (x, t) -> 0.0,
        "top" => (x, t) -> begin  # Eq. (7) of Franco et al. 2010, README [5]
            L = 0.02  # Half of the length of the electrode in meters
            a = 1.28  # Correlation constant
            b = -2.25e-3  # Correlation constant
            l = 16  # Total number of coil turns
            P = 2000.0  # Applied RF power [W]
            I = 1.0  # Impedance of tissues [Ω]

            if abs(x[2]) <= L && abs(x[1]) <= L
                return (a * (l * x[2] / L)^2 + b) * sqrt(P * I)
            else
                return nothing  # Insulating skin outside the applicator, as in [5]
            end
        end
    )

    # Run the simulation
    grid, V_dof, Qel, E_mag, E_vec, V = run_simulation(
        grid_params, rf_params, boundary_conditions; use_gpu = use_gpu);

    # Save RF results
    save_simulation("Example6_Franco.h5", grid_params, material_indices;
        Qel   = Qel,
        E_mag = E_mag,
        V = V,
        V_dof     = V_dof,
    )

    plot_graphs(material_indices, grid_params, Qel, E_mag, E_vec, V, "Images/Example6_Franco")

    return grid, V_dof, Qel, E_mag, E_vec, V
end

# Call the main function and return values for potential inspection
grid, V_dof, Qel, E_mag, E_vec, V = main();