module RFSolver

using CUDA
using Ferrite
using LinearAlgebra
using StaticArrays
using ..Config

export solve_rf, calculate_fields, convert_V

# Contribution c * Ke_ref * v_local of cell cn and the cell's DOFs, as stack-allocated
# static vectors. Shared by the CPU and GPU matvec.
Base.@propagate_inbounds function cell_product(v, cell_dofs, conductivities, Ke_ref, cn)
    dofs = ntuple(i -> cell_dofs[i, cn], Val(8))
    return dofs, conductivities[cn] * (Ke_ref * SVector(map(i -> v[i], dofs)))
end

# Adds every cell's contribution to y. On the CPU, z-layers of equal parity share no DOFs,
# so each half runs in parallel; on the GPU, each cell is a thread and adds atomically, to the
# real and imaginary parts separately, as CUDA has no complex atomics.
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
    @cuda threads = 256 blocks = cld(length(conductivities), 256) add_cells_kernel!(reinterpret(Float64, y), v, cell_dofs, conductivities, Ke_ref)

# y holds the real and imaginary part of each entry in turn
function add_cells_kernel!(y, v, cell_dofs, conductivities, Ke_ref)
    cn = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if cn <= length(conductivities)
        dofs, lr = @inbounds cell_product(v, cell_dofs, conductivities, Ke_ref, cn)
        for i in 1:8
            @inbounds CUDA.@atomic y[2dofs[i] - 1] += real(lr[i])
            @inbounds CUDA.@atomic y[2dofs[i]] += imag(lr[i])
        end
    end
    return
end

# ── Jacobi-preconditioned COCG ────────────────────────────────────────────────
# COCG is CG for complex symmetric systems: the products uᵀv are not conjugated (van der Vorst
# and Melissen 1990). Stops when ‖r‖ ≤ reltol·‖r₀‖, as IterativeSolvers.cg! did. The vector work
# takes three passes per iteration: threaded loops on the CPU, broadcasts on the GPU.

# Runs f on one chunk of 1:n per thread and returns the chunk results
on_chunks(f, n) = fetch.([Threads.@spawn f(c) for c in Iterators.partition(1:n, cld(n, Threads.nthreads()))])

# u = d_inv .* r .+ β .* u
function cg_direction!(u, r, d_inv, β)
    on_chunks(length(u)) do chunk
        @inbounds for i in chunk
            u[i] = d_inv[i] * r[i] + β * u[i]
        end
    end
end
cg_direction!(u::CuArray, r, d_inv, β) = (u .= d_inv .* r .+ β .* u)

# x += α u and r -= α c; returns ‖r‖² and rᵀ(d_inv .* r)
function cg_step!(x, r, u, c, d_inv, α)
    sums = on_chunks(length(x)) do chunk
        rr, rz = 0.0, zero(eltype(r))
        @inbounds for i in chunk
            x[i] += α * u[i]
            r[i] -= α * c[i]
            rr += abs2(r[i])
            rz += d_inv[i] * r[i]^2
        end
        rr, rz
    end
    return sum(first, sums), sum(last, sums)
end
function cg_step!(x::CuArray, r, u, c, d_inv, α)
    x .+= α .* u
    r .-= α .* c
    return sum(abs2, r), mapreduce((ri, di) -> di * ri^2, +, r, d_inv)
end

# Solves A x = b from the given x, with A applied by matvec!(y, v) and d_inv = 1 ./ diag(A).
# Returns the number of iterations, the final ‖r‖ and whether it converged.
function cocg!(x, matvec!, b, d_inv; reltol, maxiter)
    c = matvec!(similar(x), x)
    r = b .- c
    u = zero(x)
    rr, ρ = cg_step!(x, r, u, c, d_inv, 0.0)  # sums of the initial residual; x and r unchanged
    tol, ρ_prev, iterations = reltol * sqrt(rr), one(ρ), 0
    while iterations < maxiter && sqrt(rr) > tol
        cg_direction!(u, r, d_inv, ρ / ρ_prev)
        matvec!(c, u)
        α = ρ / (transpose(u) * c)
        ρ_prev = ρ
        rr, ρ = cg_step!(x, r, u, c, d_inv, α)
        iterations += 1
    end
    return iterations, sqrt(rr), sqrt(rr) <= tol
end

# Solves ∇·[(σ + ωε'' − iωε')∇V] = 0 for the complex amplitude V with Jacobi-preconditioned
# COCG. K is never assembled: on the uniform grid each cell's stiffness matrix is its complex
# conductivity times one reference matrix.
# With use_gpu = true the COCG iterations run on the GPU; set-up and result stay on the CPU.
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
    ωε0            = rf_params.ω * Config.epsilon_0
    conductivities = vec(complex.(rf_params.sigma .+ ωε0 .* rf_params.epsilon_im, .-ωε0 .* rf_params.epsilon_re))
    cell_dofs_all  = Matrix{Int32}(undef, n_basefuncs, n_cells)
    buf            = zeros(Int, n_basefuncs)
    for cn in 1:n_cells
        celldofs!(buf, dh, cn)
        cell_dofs_all[:, cn] .= buf
    end

    # ── Build RHS f and Jacobi diagonal d ───────────────────────────────────
    f  = zeros(ComplexF64, n_dofs)
    d  = zeros(ComplexF64, n_dofs)
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

    # ── Jacobi-preconditioned COCG ───────────────────────────────────────────
    V_dof = zeros(ComplexF64, n_dofs)
    apply!(V_dof, ch)  # warm-start from Dirichlet values
    V_dof = to_device(V_dof)

    iterations, residual, converged = cocg!(V_dof, matvec!, to_device(f), to_device(1.0 ./ d);
                                            reltol = 1e-8, maxiter = 10_000)
    V_dof = Array(V_dof)

    if converged
        @info "COCG converged in $iterations iterations"
    else
        @warn "COCG did not converge after $iterations iterations (residual=$residual)"
    end

    apply!(V_dof, ch)
    return V_dof, dh
end

# E = -∇V [V/m] in every cell from the nodal potential V (nx+1, ny+1, nz+1). Returns Q_el, E_mag
# and E_vec. E_vec, with shape (3, nx, ny, nz), is the cell's mean field: for trilinear elements,
# the mean difference along its four parallel edges. E_mag is the root mean square of |E| over the
# cell, with |E|² = |Ex|² + |Ey|² + |Ez|² as E is complex, so Q_el = ½(σ + ωε'')E_mag² [W/m³] is
# the cell's mean power density and adds up to the power the electrodes deliver. The displacement
# current ωε' deposits no heat.
function calculate_fields(V, rf_params::Config.RFParams, grid_params)
    C(a, b, c) = @view V[1+a:end-1+a, 1+b:end-1+b, 1+c:end-1+c]  # corner (a, b, c) of every cell
    dx, dy, dz = grid_params.lx / grid_params.nx, grid_params.ly / grid_params.ny, grid_params.lz / grid_params.nz
    Ex = (C(0,0,0) .+ C(0,1,0) .+ C(0,0,1) .+ C(0,1,1) .- C(1,0,0) .- C(1,1,0) .- C(1,0,1) .- C(1,1,1)) ./ 4dx
    Ey = (C(0,0,0) .+ C(1,0,0) .+ C(0,0,1) .+ C(1,0,1) .- C(0,1,0) .- C(1,1,0) .- C(0,1,1) .- C(1,1,1)) ./ 4dy
    Ez = (C(0,0,0) .+ C(1,0,0) .+ C(0,1,0) .+ C(1,1,0) .- C(0,0,1) .- C(1,0,1) .- C(0,1,1) .- C(1,1,1)) ./ 4dz
    # Variance of a field component over the cell, times the spacing squared, from the differences
    # along four parallel edges at positions (0,0), (1,0), (0,1), (1,1) across the cell
    variance(d00, d10, d01, d11) = (abs2(d00 - d10 + d01 - d11) + abs2(d00 + d10 - d01 - d11)) / 48 + abs2(d00 - d10 - d01 + d11) / 144
    E_mag = sqrt.(abs2.(Ex) .+ abs2.(Ey) .+ abs2.(Ez) .+
                  variance.(C(0,0,0) .- C(1,0,0), C(0,1,0) .- C(1,1,0), C(0,0,1) .- C(1,0,1), C(0,1,1) .- C(1,1,1)) ./ dx^2 .+
                  variance.(C(0,0,0) .- C(0,1,0), C(1,0,0) .- C(1,1,0), C(0,0,1) .- C(0,1,1), C(1,0,1) .- C(1,1,1)) ./ dy^2 .+
                  variance.(C(0,0,0) .- C(0,0,1), C(1,0,0) .- C(1,0,1), C(0,1,0) .- C(0,1,1), C(1,1,0) .- C(1,1,1)) ./ dz^2)
    Q_el  = 0.5 .* (rf_params.sigma .+ rf_params.ω * Config.epsilon_0 .* rf_params.epsilon_im) .* E_mag .^ 2
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