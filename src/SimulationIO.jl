module SimulationIO

using HDF5
using ..Config

export save_simulation, load_simulation

# Save grid parameters, material indices, and any additional fields to an HDF5 file
function save_simulation(filepath::String,
                         grid_params::Config.GridParams,
                         material_indices;
                         kwargs...)
    h5open(filepath, "w") do f
        _save_grid_params(f, grid_params)
        _write_compressed(f, "material_indices", material_indices, _autochunk(size(material_indices)))

        if !isempty(kwargs)
            g = create_group(f, "data")
            attrs(g)["description"] = "Simulation output fields"
            for (k, v) in kwargs
                _write_field(g, string(k), v)
            end
        end
    end
    println("Simulation saved → $filepath")
end

# Load a previously saved simulation, returning grid_params, material_indices, and a NamedTuple of data fields
function load_simulation(filepath::String)
    h5open(filepath, "r") do f
        grid_params      = _load_grid_params(f)
        material_indices = read(f["material_indices"])

        data = if haskey(f, "data")
            _load_group(f["data"])
        else
            NamedTuple()
        end

        return grid_params, material_indices, data
    end
end

# Store grid parameters as individual scalar datasets under a dedicated group
function _save_grid_params(fid, gp::Config.GridParams)
    g = create_group(fid, "grid_params")
    for fname in (:lx, :ly, :lz, :nx, :ny, :nz)
        g[string(fname)] = getfield(gp, fname)
    end
end

# Reconstruct GridParams from the saved scalar datasets
function _load_grid_params(fid)
    g = fid["grid_params"]
    Config.GridParams(
        read(g["lx"]), read(g["ly"]), read(g["lz"]),
        read(g["nx"]), read(g["ny"]), read(g["nz"])
    )
end

# Dispatch on element type to pick the appropriate HDF5 storage strategy
function _write_field(group, name::String, v)
    if v isa AbstractArray{<:Complex}
        # Complex arrays are split into real and imag subgroups, marked with an attribute for loading
        sg = create_group(group, name)
        attrs(sg)["complex"] = 1
        chunk = _autochunk(size(v))
        _write_compressed(sg, "real", Float64.(real.(v)), chunk)
        _write_compressed(sg, "imag", Float64.(imag.(v)), chunk)
    elseif v isa AbstractArray{<:Real}
        _write_compressed(group, name, Float64.(v), _autochunk(size(v)))
    elseif v isa Number
        group[name] = v
    elseif v isa AbstractString
        group[name] = v
    else
        @warn "SimulationIO: field '$name' has unsupported element type $(eltype(v)); storing as string."
        group[name] = string(v)
    end
end

# Write an array with shuffle filter and moderate deflate compression
function _write_compressed(group, name::String, data::Array, chunk)
    ds = create_dataset(
        group, name, datatype(data), dataspace(data);
        chunk   = chunk,
        shuffle = (),
        deflate = 3
    )
    write_dataset(ds, datatype(data), data)
end

# Choose chunk size targeting ~32K elements, scaled to array dimensionality
_autochunk(sz::Tuple) = length(sz) in 1:3 ? min.(sz, round(Int, 32_768^(1 / length(sz)))) : sz

# Recursively load a group, reconstructing complex arrays from real/imag subgroups
function _load_group(group)
    pairs_vec = Pair{Symbol, Any}[]
    for key in keys(group)
        obj = group[key]
        sym = Symbol(key)
        if obj isa HDF5.Group
            # Check for the complex marker attribute set during saving
            if haskey(attrs(obj), "complex") && read(attrs(obj)["complex"]) == 1
                re  = read(obj["real"])
                im_ = read(obj["imag"])
                push!(pairs_vec, sym => complex.(re, im_))
            else
                push!(pairs_vec, sym => _load_group(obj))
            end
        else
            push!(pairs_vec, sym => read(obj))
        end
    end
    return NamedTuple(pairs_vec)
end

end # module SimulationIO