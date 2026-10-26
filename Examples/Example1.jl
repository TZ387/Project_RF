# Example1.jl
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
    
    # Relative permittivities; epsilon_re of the tissues at 1 MHz from [3] in the README, the gel taken as water-like
    materials[1] = (name = "Gel",    sigma = 0.3,   epsilon_re = 80,   epsilon_im = 0, VHC = 1.0e6, k = 0.025)
    materials[2] = (name = "Skin",   sigma = 0.22,  epsilon_re = 1833, epsilon_im = 0, VHC = 4.18e6, k = 0.6)
    materials[3] = (name = "Fat",    sigma = 0.025, epsilon_re = 27.2, epsilon_im = 0, VHC = 1.0e6, k = 0.5)
    materials[4] = (name = "Muscle", sigma = 0.5,   epsilon_re = 1836, epsilon_im = 0, VHC = 1.0e6, k = 0.5)
    
    return materials
end

function main()
    # Flags for use
    create_timelapse = true    # Show live-updating heat plot during simulation
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
    materials       = define_material_properties()
    material_indices = geometryDefinition(grid_params)

    # Setup material properties (delegated to Config.jl)
    sigma, epsilon_re, epsilon_im, VHC, k = setup_material_properties(material_indices, materials)

    # Load RF parameters
    rf_params = RFParams(
        sigma,       # Electrical conductivity matrix [S/m]
        epsilon_re,  # Real part of the relative permittivity ε'/ε0
        epsilon_im,  # Imaginary part of the relative permittivity ε''/ε0
        1e6,         # ω = Angular frequency of the RF signal [rad/s]
    )

    # Set initial temperature
    T_initial = fill(37.0, grid_params.nx, grid_params.ny, grid_params.nz)  # Initial temperature [°C]

    # Load heat parameters
    heat_params = HeatParams(
        [                       # schedule = ordered list of (:on/:off, duration [s]) phases
            (:on,  3.0),        # heat for 3 s
            (:off, 6.0),        # cool for 6 s
            (:on,  3.0),        # heat again for 3 s
        ],
        2,                      # n_update  = Number of plot refreshes per phase
        T_initial,              # T_initial = Initial temperature [°C]
        VHC,                    # Volumetric heat capacity matrix [J/(m³·K)]
        k;                      # Thermal conductivity matrix [W/(m·K)]
        T_plot_min = 37.0,      # lower limit for live heat plot [°C]  — nothing = auto
        T_plot_max = 37.5,      # upper limit for live heat plot [°C]  — nothing = auto
    )

    # Define boundary conditions as functions
    boundary_conditions = Dict(
        "top"    => (x, t) -> 10.0,
        "bottom" => (x, t) -> 0.0
        # Add other boundaries here as needed
    )

    # Run the simulation
    grid, V_dof, Qel, E_mag, E_vec, V = run_simulation(
        grid_params, rf_params, boundary_conditions; use_gpu = use_gpu);

    # Save RF results
    save_simulation("Example1.h5", grid_params, material_indices;
        Qel=Qel, E_mag=E_mag, V=V)

    plot_graphs(material_indices, grid_params, Qel, E_mag, E_vec, V)

    T_final = run_heat_simulation(Qel, grid_params, heat_params; create_timelapse = create_timelapse, use_gpu = use_gpu)

    return grid, V_dof, Qel, E_mag, E_vec, V, T_final
end

# Call the main function and return values for potential inspection
grid, V_dof, Qel, E_mag, E_vec, V, T_final = main();