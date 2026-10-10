# Config.jl

module Config

# Export the structures and functions to make them available outside this module
export RFParams, HeatParams, HeatSchedule, GridParams, setup_material_properties, create_coordinate_grids

# Parameters for RF problem
struct RFParams
    sigma::Array{Float64, 3}  # Electrical conductivity matrix [S/m]
    epsilon_im::Array{Float64, 3}  # Permittivity matrix [F/m]
    ω::Float64  # (Circular) Frequency of the RF signal [Hz]
end

# Ordered sequence of heating/cooling phases.
# Each entry is a (Symbol, Float64) tuple where the symbol is :on or :off
# and the float is the duration of that phase in seconds.
# Example: [(:on, 30.0), (:off, 60.0), (:on, 15.0)]
const HeatSchedule = Vector{Tuple{Symbol, Float64}}

# Parameters for heat problem
struct HeatParams
    schedule  :: HeatSchedule        # Ordered sequence of (:on/:off, duration [s]) phases
    n_update  :: Int                 # Number of plot updates per phase
    T_initial :: Array{Float64, 3}   # Initial temperature [°C]
    VHC       :: Array{Float64, 3}   # Volumetric heat capacity [J/(m³·K)]  (VHC = rho * c)
    k         :: Array{Float64, 3}   # Thermal conductivity [W/(m·K)]
    # Optional fixed colour-scale limits for the live heat plot.
    # When both are set, the colorbar stays fixed throughout the simulation.
    # When either is nothing, limits are derived automatically from the data.
    T_plot_min :: Union{Float64, Nothing}
    T_plot_max :: Union{Float64, Nothing}
end

# Convenience constructor — T_plot_min/T_plot_max default to nothing so all
# existing call sites that pass only the original five arguments still work.
function HeatParams(schedule, n_update, T_initial, VHC, k;
                    T_plot_min::Union{Float64,Nothing} = nothing,
                    T_plot_max::Union{Float64,Nothing} = nothing)
    HeatParams(schedule, n_update, T_initial, VHC, k, T_plot_min, T_plot_max)
end

# Parameters for grid setup
struct GridParams
    lx::Float64  # Length of the simulation domain in x-direction [m]
    ly::Float64  # Length of the simulation domain in y-direction [m]
    lz::Float64  # Length of the simulation domain in z-direction [m]
    nx::Int  # Number of elements in the x-direction
    ny::Int  # Number of elements in the y-direction
    nz::Int  # Number of elements in the z-direction
end

# Function to create 3D coordinate grids from grid parameters
function create_coordinate_grids(grid_params::GridParams)
    # Create 3D coordinate grids - contains actual coordinates in meters
    # Use (index - 0.5) to get cell center coordinates
    X = repeat(reshape((1:grid_params.nx) .- 0.5, grid_params.nx, 1, 1) .* (grid_params.lx / grid_params.nx), 1, grid_params.ny, grid_params.nz)
    Y = repeat(reshape((1:grid_params.ny) .- 0.5, 1, grid_params.ny, 1) .* (grid_params.ly / grid_params.ny), grid_params.nx, 1, grid_params.nz)
    Z = repeat(reshape((1:grid_params.nz) .- 0.5, 1, 1, grid_params.nz) .* (grid_params.lz / grid_params.nz), grid_params.nx, grid_params.ny, 1)
    
    return X, Y, Z
end

# Function to setup material properties matrices based on the geometry
function setup_material_properties(material_indices, materials)
    sigma, epsilon_im, VHC, k = (zeros(size(material_indices)) for _ in 1:4)

    for i in eachindex(material_indices)
        mat = materials[material_indices[i]]
        sigma[i]      = mat[:sigma]
        epsilon_im[i] = mat[:epsilon_im]
        VHC[i]        = mat[:VHC]
        k[i]          = mat[:k]
    end
    
    return sigma, epsilon_im, VHC, k
end

end # module Config