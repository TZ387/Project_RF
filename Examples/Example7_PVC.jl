# Include the main module - this loads all submodules
using prrf
using GLMakie

# Geometry definition function
function geometryDefinition(grid_params::GridParams)
    # Get coordinate grids from Config module (implementation details hidden)
    X, Y, Z = create_coordinate_grids(grid_params)
    # Initialize every cell with material 1
    M = ones(Int, grid_params.nx, grid_params.ny, grid_params.nz)
    
    # Apply conditions using absolute coordinates [m]
    M[Z .> 0.001] .= 2    # Skin (1 mm depth)
    M[Z .> 0.003] .= 3     # Fat (3 mm depth)  
    M[Z .> 0.008] .= 4    # Muscle (8 mm depth)
    
    return M
end

# Define materials and their properties
function define_material_properties()
    materials = Dict{Int, NamedTuple}()
    
    # Relative permittivities; epsilon_re of the tissues at 1 MHz from [3] in the README
    # PVC at 1 MHz: ε'r = 4.0 and tan δ = 0.14 (Cardarelli, Materials Handbook, 2018), so sigma = ωε0ε'r tan δ
    materials[1] = (name = "PVC", sigma = 3.1e-5, epsilon_re = 4.0, epsilon_im = 0, VHC = 1.0e6, k = 0.025)
    materials[2] = (name = "Skin", sigma = 0.22, epsilon_re = 1833, epsilon_im = 0, VHC = 4.18e6, k = 0.6)
    materials[3] = (name = "Fat", sigma = 0.025, epsilon_re = 27.2, epsilon_im = 0, VHC = 1.0e6, k = 0.5)
    materials[4] = (name = "Muscle", sigma = 0.5, epsilon_re = 1836, epsilon_im = 0, VHC = 1.0e6, k = 0.5)
    
    return materials
end

function main()
    # Flags for use
    use_gpu = false            # Solve on an NVIDIA GPU (needs CUDA)

    # Define grid parameters
    grid_params = GridParams(
        0.3,    # lx = Length of the simulation domain in x-direction [m]
        0.3,    # ly = Length of the simulation domain in y-direction [m]
        0.0155, # lz = Length of the simulation domain in z-direction [m]
        100,    # nx = Number of elements in the x-direction
        100,    # ny = Number of elements in the y-direction
        310     # nz = Number of elements in the z-direction
    )

    # Load materials and geometry
    materials = define_material_properties()
    material_indices = geometryDefinition(grid_params)

    # Setup material properties (delegated to Config.jl)
    sigma, epsilon_re, epsilon_im, VHC, k = setup_material_properties(material_indices, materials)

    # Load RF parameters
    rf_params = RFParams(
        sigma,   # Electrical conductivity matrix [S/m]
        epsilon_re, # Real part of the relative permittivity ε'/ε0
        epsilon_im, # Imaginary part of the relative permittivity ε''/ε0
        2π * 1e6, # ω = Angular frequency of the RF signal [rad/s], f = 1 MHz
    )


    # Parameters for the circular electrode
    x_center = 0.005  # X-coordinate of the center of the electrode
    y_center = 0.001  # Y-coordinate of the center of the electrode
    radius = 0.025   # Radius of the circular electrode
    V0 = 500.0       # Constant potential of the circular electrode

    boundary_conditions = Dict(
        "bottom" => (x, t) -> 0.0,
        #"left" => (x, t) -> 0.0,
        #"right" => (x, t) -> 0.0,
        #"front" => (x, t) -> 0.0,
        #"back" => (x, t) -> 0.0,
        "top" => (x, t) -> begin
        r = sqrt((x[1] - x_center)^2 + (x[2] - y_center)^2)
        if r <= radius
            return V0  # Constant potential inside circular region
        else
            return nothing  # Insulating outside the circular region, where the PVC faces air
        end
        end
    )

    # Run the simulation
    grid, V_dof, Qel, E_mag, E_vec, V = run_simulation(
        grid_params, rf_params, boundary_conditions; use_gpu = use_gpu);
    plot_graphs(material_indices, grid_params, Qel, E_mag, E_vec, V)

    
    bio_mask = material_indices .!= 1
    Qel_bio = Qel .* bio_mask

    fig5 = plot_slices(Qel_bio, grid_params, title = "Qel in Biological Tissue Only [W/m³]")
    window5 = display(GLMakie.Screen(), fig5)
    
    return grid, V_dof, Qel, E_mag, E_vec, V
end

# Call the main function and return values for potential inspection
grid, V_dof, Qel, E_mag, E_vec, V = main();