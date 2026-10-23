module RFSolver

using CUDA
using Ferrite
using LinearAlgebra
using IterativeSolvers
using LinearMaps
using StaticArrays
using ..Config

export solve_rf, calculate_fields, convert_V

# Jacobi (diagonal) preconditioner — stores 1/d[i] directly.
struct JacobiPrecond{V <: AbstractVector}
    d_inv::V
end
LinearAlgebra.ldiv!(y::AbstractVector, P::JacobiPrecond, x::AbstractVector) = (y .= x .* P.d_inv)

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

# Solves ∇·[(σ + ωε'')∇V] = 0 with Jacobi-preconditioned CG. K is never assembled: on the
# uniform grid each cell's stiffness matrix is its conductivity times one reference matrix.
# With use_gpu = true the CG iterations run on the GPU; set-up and result stay on the CPU.
function solve_rf(grid, rf_params::Config.RFParams, grid_params::Config.GridParams, boundary_conditions;
                  use_gpu::Bool = false)

    basis      = Lagrange{Ferrite.RefHexahedron, 1}()
    quad       = QuadratureRule{Ferrite.RefHexahedron}(2)
    cellvalues = CellValues(quad, basis)

    dh = DofHandler(grid)
    add!(dh, :V_dof, basis)
    close!(dh)

    ch = ConstraintHandler(dh)
    for (boundary_name, condition) in boundary_conditions
        # Facets where the condition returns nothing stay insulating; Ferrite accepts f(x) as well as f(x, t)
        value = hasmethod(condition, Tuple{Vec{3, Float64}, Float64}) ? condition : (x, _) -> condition(x)
        prescribed = filter(getfacetset(grid, boundary_name)) do (cell, facet)
            all(n -> value(get_node_coordinate(grid, n), 0.0) !== nothing, Ferrite.facets(getcells(grid, cell))[facet])
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

    # ── Ke_ref: stiffness matrix of a cell with unit conductivity ───────────
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
    Ke_ref = SMatrix{8,8}(Ke_ref_mut)

    # ── Cache conductivities and DOF indices per cell ────────────────────────
    # Cells are numbered x fastest, like the (nx, ny, nz) material arrays.
    conductivities = vec(rf_params.sigma .+ rf_params.ω .* rf_params.epsilon_im)
    cell_dofs_all  = Matrix{Int32}(undef, n_basefuncs, n_cells)
    buf            = zeros(Int, n_basefuncs)
    for cn in 1:n_cells
        celldofs!(buf, dh, cn)
        cell_dofs_all[:, cn] .= buf
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

    V_dof, history = cg!(to_device(V_dof), K_mf, to_device(f); Pl=P, reltol=1e-8, maxiter=10_000,
                          initially_zero=false, log=true)
    V_dof = Array(V_dof)

    if !history.isconverged
        @warn "CG did not converge after $(history.iters) iterations " *
              "(residual=$(history[:resnorm][end]))"
    else
        @info "CG converged in $(history.iters) iterations"
    end

    apply!(V_dof, ch)
    return V_dof, dh
end

# E = -∇V [V/m] in every cell from the nodal potential V (nx+1, ny+1, nz+1). For trilinear
# elements the cell's mean gradient is the mean difference along its four parallel edges.
# Returns the power density Q_el = ½(σ + ωε'')|E|² [W/m³], |E| and E_vec with shape (3, nx, ny, nz).
function calculate_fields(V, rf_params::Config.RFParams, grid_params)
    C(a, b, c) = @view V[1+a:end-1+a, 1+b:end-1+b, 1+c:end-1+c]  # corner (a, b, c) of every cell
    dx, dy, dz = grid_params.lx / grid_params.nx, grid_params.ly / grid_params.ny, grid_params.lz / grid_params.nz
    Ex = (C(0,0,0) .+ C(0,1,0) .+ C(0,0,1) .+ C(0,1,1) .- C(1,0,0) .- C(1,1,0) .- C(1,0,1) .- C(1,1,1)) ./ 4dx
    Ey = (C(0,0,0) .+ C(1,0,0) .+ C(0,0,1) .+ C(1,0,1) .- C(0,1,0) .- C(1,1,0) .- C(0,1,1) .- C(1,1,1)) ./ 4dy
    Ez = (C(0,0,0) .+ C(1,0,0) .+ C(0,1,0) .+ C(1,1,0) .- C(0,0,1) .- C(1,0,1) .- C(0,1,1) .- C(1,1,1)) ./ 4dz
    E_mag = sqrt.(Ex .^ 2 .+ Ey .^ 2 .+ Ez .^ 2)
    Q_el  = 0.5 .* (rf_params.sigma .+ rf_params.ω .* rf_params.epsilon_im) .* E_mag .^ 2
    return Q_el, E_mag, permutedims(cat(Ex, Ey, Ez; dims = 4), (4, 1, 2, 3))
end

# DOFs are not numbered in node order, so copy each cell's DOF values to its nodes.
function convert_V(V_dof, dh, grid_params)
    grid, dofs = Ferrite.get_grid(dh), zeros(Int, 8)
    V = similar(V_dof, getnnodes(grid))
    for cn in 1:getncells(grid)
        celldofs!(dofs, dh, cn)
        for (i, node) in enumerate(getcells(grid, cn).nodes)
            V[node] = V_dof[dofs[i]]
        end
    end
    return reshape(V, grid_params.nx + 1, grid_params.ny + 1, grid_params.nz + 1)
end

end # module RFSolver