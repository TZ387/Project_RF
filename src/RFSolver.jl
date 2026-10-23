module RFSolver

using CUDA
using Ferrite
using LinearAlgebra
using IterativeSolvers
using LinearMaps
using StaticArrays
using ..Config

export solve_rf, calculate_fields, convert_V

function cell_index_to_xyz(cell_number, nx_max, ny_max)
    cell_number -= 1
    x = (cell_number % (nx_max * ny_max)) % nx_max + 1
    y = div((cell_number % (nx_max * ny_max)), nx_max) + 1
    z = div(cell_number, nx_max * ny_max) + 1
    return x, y, z
end

# Jacobi (diagonal) preconditioner — stores 1/d[i] directly.
struct JacobiPrecond{V <: AbstractVector}
    d_inv::V
end
LinearAlgebra.ldiv!(y::AbstractVector, P::JacobiPrecond, x::AbstractVector) = (y .= x .* P.d_inv)
LinearAlgebra.ldiv!(P::JacobiPrecond, x::AbstractVector) = (x .*= P.d_inv)
Base.:\(P::JacobiPrecond, x::AbstractVector) = x .* P.d_inv

# Contribution c * Ke_ref * v_local of cell cn and the cell's DOFs, as stack-allocated
# static vectors. Shared by the CPU and GPU matvec.
Base.@propagate_inbounds function cell_product(v, cell_dofs, conductivities, Ke_ref, cn)
    dofs = ntuple(i -> cell_dofs[i, cn], Val(8))
    return dofs, conductivities[cn] * (Ke_ref * SVector(map(i -> v[i], dofs)))
end

# Adds every cell's contribution to y. On the CPU, z-layers of equal parity share no DOFs,
# so each half runs in parallel; on the GPU, each cell is a thread and adds atomically.
function add_cells!(y, v, cell_dofs, conductivities, Ke_ref, n_layer)
    for parity in 1:2
        Threads.@threads for iz in parity:2:length(conductivities) ÷ n_layer
            @inbounds for cn in (iz - 1) * n_layer + 1 : iz * n_layer
                dofs, lr = cell_product(v, cell_dofs, conductivities, Ke_ref, cn)
                for i in 1:8
                    y[dofs[i]] += lr[i]
                end
            end
        end
    end
end

add_cells!(y::CuArray, v, cell_dofs, conductivities, Ke_ref, _) =
    @cuda threads = 256 blocks = cld(length(conductivities), 256) add_cells_kernel!(y, v, cell_dofs, conductivities, Ke_ref)

function add_cells_kernel!(y, v, cell_dofs, conductivities, Ke_ref)
    cn = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if cn <= length(conductivities)
        dofs, lr = @inbounds cell_product(v, cell_dofs, conductivities, Ke_ref, cn)
        for i in 1:8
            @inbounds CUDA.@atomic y[dofs[i]] += lr[i]
        end
    end
    return
end

# Solve the RF Laplace problem
#   div[(sigma + omega*epsilon_im) * grad(V)] = 0
# using matrix-free preconditioned CG.
#
# K is NEVER assembled or stored — memory is O(n).
#
# Speed: on a uniform structured hex grid (from generate_grid) every element
# has identical geometry, so  Ke(cell) = conductivity(cell) * Ke_ref.
# Ke_ref is computed once from the first element (512 bytes).
# Each matvec is then: for each cell, y_local += c * Ke_ref * v_local —
# an 8x8 dense matvec scaled by a scalar, with no quadrature at runtime.
# This is 8x fewer FLOPs per iteration than recomputing quadrature each time,
# and lower memory bandwidth than a sparse matvec on an explicit K.
#
# With use_gpu = true the CG iterations run on the GPU; set-up and result stay on the CPU.
function solve_rf(grid, rf_params::Config.RFParams, grid_params::Config.GridParams, boundary_conditions;
                  use_gpu::Bool = false)

    sigma     = rf_params.sigma
    eps_im    = rf_params.epsilon_im
    ω         = rf_params.ω

    basis      = Lagrange{Ferrite.RefHexahedron, 1}()
    quad       = QuadratureRule{Ferrite.RefHexahedron}(2)
    cellvalues = CellValues(quad, basis)

    dh = DofHandler(grid)
    add!(dh, :V_dof, basis)
    close!(dh)

    ch = ConstraintHandler(dh)
    for (boundary_name, condition) in boundary_conditions
        # Facets where the condition returns nothing stay insulating; Ferrite accepts f(x) as well as f(x, t)
        f = hasmethod(condition, Tuple{Vec{3, Float64}, Float64}) ? condition : (x, _) -> condition(x)
        prescribed = filter(getfacetset(grid, boundary_name)) do (cell, facet)
            all(n -> f(get_node_coordinate(grid, n), 0.0) !== nothing, Ferrite.facets(getcells(grid, cell))[facet])
        end
        add!(ch, Dirichlet(:V_dof, prescribed, condition))
    end
    close!(ch)

    n_dofs      = ndofs(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    nqp         = getnquadpoints(cellvalues)
    nx_max      = grid_params.nx
    ny_max      = grid_params.ny
    n_cells     = nx_max * ny_max * grid_params.nz

    # ── Precompute Ke_ref ────────────────────────────────────────────────────
    # All elements share the same geometry (uniform hex mesh), so the
    # geometric part of the stiffness matrix is identical across elements.
    # We compute it once from the first element with unit conductivity.
    # Stored as SMatrix so the 8x8 mul! in the matvec is fully unrolled
    # and SIMD-vectorized by the compiler — no BLAS overhead for tiny matrices.
    Ke_ref_mut = zeros(n_basefuncs, n_basefuncs)
    let fc = first(CellIterator(dh))
        reinit!(cellvalues, fc)
        for qp in 1:nqp
            dV = getdetJdV(cellvalues, qp)
            for i in 1:n_basefuncs
                gi = shape_gradient(cellvalues, qp, i)
                for j in 1:n_basefuncs
                    gj = shape_gradient(cellvalues, qp, j)
                    Ke_ref_mut[i, j] += (gi ⋅ gj) * dV
                end
            end
        end
    end
    Ke_ref = SMatrix{8,8}(Ke_ref_mut)  # immutable static matrix

    # ── Cache conductivities and DOF indices per cell ────────────────────────
    # One Float64 + 8 Int32 per cell.
    conductivities = Vector{Float64}(undef, n_cells)
    cell_dofs_all  = Matrix{Int32}(undef, n_basefuncs, n_cells)
    buf            = zeros(Int, n_basefuncs)

    for (cn, cell) in enumerate(CellIterator(dh))
        celldofs!(buf, dh, cn)
        cell_dofs_all[:, cn] .= buf
        ix, iy, iz = cell_index_to_xyz(cn, nx_max, ny_max)
        conductivities[cn] = sigma[ix, iy, iz] + ω * eps_im[ix, iy, iz]
    end

    # ── Build RHS f and Jacobi diagonal d ───────────────────────────────────
    f  = zeros(n_dofs)
    d  = zeros(n_dofs)
    Kd = diag(Ke_ref)  # SVector{8} diagonal

    for cn in 1:n_cells
        c    = conductivities[cn]
        dofs = @view cell_dofs_all[:, cn]
        for i in 1:n_basefuncs
            d[dofs[i]] += c * Kd[i]
        end
    end

    apply!(f, ch)
    for dof in ch.prescribed_dofs
        d[dof] = 1.0
    end

    # ── Matrix-free matvec y = K_eff * v ────────────────────────────────────
    to_device(A) = use_gpu ? CuArray(A) : A
    cell_dofs, cell_conductivities, dirichlet_dofs = to_device.((cell_dofs_all, conductivities, ch.prescribed_dofs))

    function matvec!(y, v)
        fill!(y, 0.0)
        add_cells!(y, v, cell_dofs, cell_conductivities, Ke_ref, nx_max * ny_max)
        view(y, dirichlet_dofs) .= view(v, dirichlet_dofs)  # identity rows
        return y
    end

    K_mf = LinearMap(matvec!, n_dofs; ismutating=true, issymmetric=true, isposdef=true)

    # ── Jacobi-preconditioned CG ─────────────────────────────────────────────
    P = JacobiPrecond(to_device(1.0 ./ d))

    V_dof = zeros(n_dofs)
    apply!(V_dof, ch)  # warm-start from Dirichlet values

    V_dof, history = cg!(to_device(V_dof), K_mf, to_device(f); Pl=P, reltol=1e-8, maxiter=3000,
                          initially_zero=false, log=true)
    V_dof = Array(V_dof)

    if !history.isconverged
        @warn "CG did not converge after $(history.iters) iterations " *
              "(residual=$(history[:resnorm][end]))"
    else
        @info "CG converged in $(history.iters) iterations"
    end

    apply!(V_dof, ch)
    return V_dof, dh, cellvalues
end

# Calculate electric field magnitude E [V/m], power density Q_el [W/m^3] from
# conduction and dielectric losses, and electric field vector E_vec [V/m] with
# shape (3, nx, ny, nz).
# E = -grad(V), so each component is the negated gradient of the voltage field.
# E_vec[1/2/3, x, y, z] gives the x/y/z component; (3, nx, ny, nz) layout keeps
# the three components of each cell contiguous in memory (column-major).
function calculate_fields(cellvalues::CellValues, dh::DofHandler, V_dof::AbstractVector{T}, rf_params::Config.RFParams, grid_params) where T
    n         = getnbasefunctions(cellvalues)
    cell_dofs = zeros(Int, n)
    nqp       = getnquadpoints(cellvalues)
    sigma_eff = rf_params.sigma .+ rf_params.ω .* rf_params.epsilon_im
    Q_el      = similar(sigma_eff)
    E_mag     = similar(sigma_eff)
    E_vec     = zeros(T, 3, grid_params.nx, grid_params.ny, grid_params.nz)

    for (cell_num, cell) in enumerate(CellIterator(dh))
        celldofs!(cell_dofs, dh, cell_num)
        ae = V_dof[cell_dofs]
        reinit!(cellvalues, cell)
        x, y, z = cell_index_to_xyz(cell_num, grid_params.nx, grid_params.ny)

        gx, gy, gz = zero(T), zero(T), zero(T)
        for qp in 1:nqp
            g   = function_gradient(cellvalues, qp, ae)
            gx += g[1]
            gy += g[2]
            gz += g[3]
        end
        gx /= nqp
        gy /= nqp
        gz /= nqp

        Q_el[x, y, z]    = 0.5 * sigma_eff[x, y, z] * (gx^2 + gy^2 + gz^2)
        E_mag[x, y, z]   = sqrt(gx^2 + gy^2 + gz^2)
        E_vec[1, x, y, z] = -gx
        E_vec[2, x, y, z] = -gy
        E_vec[3, x, y, z] = -gz
    end
    return Q_el, E_mag, E_vec
end

# DOFs are not numbered in node order, so map them onto the grid nodes first.
function convert_V(V_dof, dh, grid_params)
    V = evaluate_at_grid_nodes(dh, V_dof, :V_dof)
    return reshape(V, grid_params.nx + 1, grid_params.ny + 1, grid_params.nz + 1)
end

end # module RFSolver