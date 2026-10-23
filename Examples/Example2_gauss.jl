# Include the main module - this loads all submodules
using prrf

# Geometry definition function
function geometryDefinition(grid_params::GridParams)
    # Get coordinate grids from Config module (implementation details hidden)
    X, Y, Z = create_coordinate_grids(grid_params)
    # Initialize every cell with material 1
    M = ones(Int, grid_params.nx, grid_params.ny, grid_params.nz)
    
    # Apply conditions using absolute coordinates [m]
    M[Z .> 0.0005] .= 2    # Skin (0.5 mm depth)
    M[Z .> 0.002] .= 3     # Fat (2 mm depth)  
    M[Z .> 0.0055] .= 4    # Muscle (5.5 mm depth)
    
    return M
end

# Define materials and their properties
function define_material_properties()
    materials = Dict{Int, NamedTuple}()
    
    epsilon_0 = 8.854e-12  # Permittivity of free space [F/m]
    materials[1] = (name = "Gel", sigma = 0.3, epsilon_im = 0*epsilon_0, VHC = 1.0e6, k = 0.025)
    materials[2] = (name = "Skin", sigma = 0.22, epsilon_im = 0*epsilon_0, VHC = 4.18e6, k = 0.6)
    materials[3] = (name = "Fat", sigma = 0.025, epsilon_im = 0*epsilon_0, VHC = 1.0e6, k = 0.5)
    materials[4] = (name = "Muscle", sigma = 0.5, epsilon_im = 0*epsilon_0, VHC = 1.0e6, k = 0.5)
    
    return materials
end

function main()
    # Flags for use
    use_gpu = false            # Solve on an NVIDIA GPU (needs CUDA)

    # Define grid parameters
    grid_params = GridParams(
        0.3,    # lx = Length of the simulation domain in x-direction [m]
        0.3,    # ly = Length of the simulation domain in y-direction [m]
        0.0105, # lz = Length of the simulation domain in z-direction [m]
        100,    # nx = Number of elements in the x-direction
        100,    # ny = Number of elements in the y-direction
        210     # nz = Number of elements in the z-direction
    )

    # Load materials and geometry
    materials = define_material_properties()
    material_indices = geometryDefinition(grid_params)

    # Setup material properties (delegated to Config.jl)
    sigma, epsilon_im, VHC, k = setup_material_properties(material_indices, materials)

    # Load RF parameters
    rf_params = RFParams(
        sigma,   # Electrical conductivity matrix [S/m]
        epsilon_im, # Imaginary part of the permittivity ε'' [F/m]
        1e6,     # ω = Angular frequency of the RF signal [rad/s]
    )

    # Define boundary conditions as Gaussian functions
    A = 10.0   # Amplitude of the Gaussian (e.g., 10 V)
    mu = 0.005  # x-coordinate of the Gaussian's centre [m]
    sigma2 = 0.03  # Standard deviation (controls the width of the Gaussian)

    boundary_conditions = Dict(
        "bottom" => (x, t) -> 0.0,
        #"left" => (x, t) -> 0.0,
        #"right" => (x, t) -> 0.0,
        #"front" => (x, t) -> 0.0,
        #"back" => (x, t) -> 0.0,
        "top" => (x, t) -> A * exp(-((x[1] - mu)^2 + (x[2] - 0.001)^2) / (2 * sigma2^2))
    )

    # Run the simulation
    grid, V_dof, Qel, E_mag, E_vec, V = run_simulation(
        grid_params, rf_params, boundary_conditions; use_gpu = use_gpu);

    plot_graphs(material_indices, grid_params, Qel, E_mag, E_vec, V)

    return grid, V_dof, Qel, E_mag, E_vec, V
end

# Call the main function and return values for potential inspection
grid, V_dof, Qel, E_mag, E_vec, V = main();