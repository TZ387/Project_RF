# Config.jl

module Config

# Export the structures and functions to make them available outside this module
export RFParams, HeatPhase, HeatParams, GridParams, setup_material_properties, create_coordinate_grids

const epsilon_0 = 8.854e-12  # Permittivity of free space [F/m]

# Parameters for RF problem
struct RFParams
    sigma::Array{Float64, 3}  # Electrical conductivity matrix [S/m]
    epsilon_re::Array{Float64, 3}  # Real part of the relative permittivity, ε'/ε0
    epsilon_im::Array{Float64, 3}  # Imaginary part of the relative permittivity, ε''/ε0
    ω::Float64  # Angular frequency of the RF signal [rad/s], ω = 2πf
end

# One phase of the heat schedule: for `duration` seconds the heat source is rf_factor·Qel + Q_extra.
# rf_factor and Q_extra are numbers (uniform) or (nx, ny, nz) arrays (one value per cell), 0 when left out.
# Example: HeatPhase(duration = 30.0, rf_factor = 0.5)
struct HeatPhase
    duration  :: Float64                                # [s]
    rf_factor :: Union{Real, AbstractArray{<:Real, 3}}  # Factor on the RF power Qel
    Q_extra   :: Union{Real, AbstractArray{<:Real, 3}}  # Additional heat source [W/m³]
end

HeatPhase(; duration, rf_factor = 0.0, Q_extra = 0.0) = HeatPhase(duration, rf_factor, Q_extra)

# Parameters for heat problem
struct HeatParams
    schedule  :: Vector{HeatPhase}   # Ordered sequence of heating/cooling phases
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

# T_plot_min and T_plot_max are optional keywords
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
    # Use (index - 0.5) to get cell center coordinates; x and y are centred like the Ferrite grid
    X = repeat(reshape((1:grid_params.nx) .- 0.5, grid_params.nx, 1, 1) .* (grid_params.lx / grid_params.nx) .- grid_params.lx / 2, 1, grid_params.ny, grid_params.nz)
    Y = repeat(reshape((1:grid_params.ny) .- 0.5, 1, grid_params.ny, 1) .* (grid_params.ly / grid_params.ny) .- grid_params.ly / 2, grid_params.nx, 1, grid_params.nz)
    Z = repeat(reshape((1:grid_params.nz) .- 0.5, 1, 1, grid_params.nz) .* (grid_params.lz / grid_params.nz), grid_params.nx, grid_params.ny, 1)
    
    return X, Y, Z
end

# Function to setup material properties matrices based on the geometry
function setup_material_properties(material_indices, materials)
    sigma, epsilon_re, epsilon_im, VHC, k = (zeros(size(material_indices)) for _ in 1:5)

    for i in eachindex(material_indices)
        mat = materials[material_indices[i]]
        sigma[i]      = mat[:sigma]
        epsilon_re[i] = mat[:epsilon_re]
        epsilon_im[i] = mat[:epsilon_im]
        VHC[i]        = mat[:VHC]
        k[i]          = mat[:k]
    end
    
    return sigma, epsilon_re, epsilon_im, VHC, k
end

end # module Config