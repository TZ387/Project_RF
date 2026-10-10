# prrf

A program for RF heating simulation in biological tissue. Given an arbitrary electrode geometry and boundary conditions, it computes the electric potential, electric field, and ohmic heat dissipation in the tissue, and then uses that dissipation as a source term to evolve the temperature field over time.

## Requirements

- [Julia](https://julialang.org/) ≥ 1.10
- Dependencies are listed in `Project.toml` and managed by Julia's built-in package manager. To install them, open a Julia REPL in the project directory and run:

```julia
using Pkg
Pkg.instantiate()
```

## 1. Theoretical basis

### 1.1 RF electric field

The RF heating procedure uses two or more electrodes to apply alternating electrical current to tissue — one electrode applies an alternating electrical potential, while the other acts as an electrical ground.

The governing equation for the electric potential in the tissue is:

$$\nabla[(\sigma - i\omega\epsilon)(\nabla V)]=0$$

Considering $\epsilon = \epsilon' + i\epsilon''$ and that the real part $\epsilon'$ only introduces a phase shift that does not affect RF heating at typical frequencies in the range of $10^6$ s$^{-1}$, this simplifies to:

$$\nabla[(\sigma + \omega\epsilon'')(\nabla V)]=0$$

Here, $\sigma$ [S/m] is electrical conductivity, $\omega$ [1/s] is the angular frequency, $V$ [V] is the electric potential, and $\epsilon$ [F/m] is the electrical permittivity of the material.

Note that $\epsilon$ is the product of the vacuum permittivity $\epsilon_0 \approx 8.85 \times 10^{-12}$ F/m and the relative permittivity $\epsilon_r$:

$$D = \epsilon E = \epsilon_0 \epsilon_r E$$

Once $V$ is known, the electric field is $\mathbf{E} = -\nabla V$ [V/m], and the ohmic power density (volumetric heat source) deposited in the tissue is:

$$Q_{el} = \frac{1}{2} \sigma |\mathbf{E}|^2 \quad \text{[W/m}^3\text{]}$$

### 1.2 Heat equation

The temperature field $T$ is evolved using the classic heat equation with $Q_{el}$ as a volumetric source:

$$\text{VHC}(\mathbf{x})\, \frac{\partial T}{\partial t} = \nabla \cdot [k(\mathbf{x})\, \nabla T] + Q(\mathbf{x},t)$$

where VHC $= \rho c$ [J/(m³·K)] is the volumetric heat capacity and $k$ [W/(m·K)] is the thermal conductivity, both spatially varying according to the tissue geometry.

The simulation follows an arbitrary **heating schedule** — an ordered list of `(:on, duration)` and `(:off, duration)` phases, for example:

```julia
schedule = [(:on, 30.0), (:off, 60.0), (:on, 15.0)]
```

During `:on` phases $Q = Q_{el}$; during `:off` phases $Q = 0$.

The time step $\Delta t$ is chosen automatically to satisfy the explicit-Euler von-Neumann stability criterion:

$$\Delta t \leq \frac{\text{VHC}}{2\, k \left(\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} + \frac{1}{\Delta z^2}\right)}$$

evaluated over all cells, with a safety factor of 0.45.

### 1.3 CPU parallelisation

The Laplacian and Euler update kernels are parallelised (if this is enabled, see section Enabling Multiple Threads below) by partitioning the grid's z-dimension into chunks, one per worker thread. The worker count is set automatically to $\min(N_\text{Julia threads},\ \lfloor N_\text{logical cores} / 2 \rfloor)$, excluding hyperthreads which share the FPU and provide no benefit for dense arithmetic. The active count is printed at the start of each phase.

The matrix-vector product of the RF solver is parallelised over alternating z-layers and uses all Julia threads.

## 2. Usage

The workflow consists of two steps that can be run independently.

### Step 1 — RF simulation

Run `run_simulation` to obtain the electric field results and the ohmic heat source `Qel`:

```julia
grid, V_dof, Qel, E_mag, E_vec, V = run_simulation(grid_params, rf_params, boundary_conditions)
```

### Step 2 — Heat simulation

Pass `Qel` to `run_heat_simulation` to evolve the temperature field:

```julia
T_final = run_heat_simulation(Qel, grid_params, heat_params)
```

Set `create_timelapse = true` to display a live-updating cross-section plot of the temperature field as the simulation runs:

```julia
T_final = run_heat_simulation(Qel, grid_params, heat_params; create_timelapse = true)
```

The plot refreshes `n_update` times per schedule phase. The colour scale is updated automatically to the current field range at each refresh.

### Saving plots

To save plots to the `Images` subfolder, pass a filename to `plot_graphs`:

```julia
plot_graphs(material_indices, grid_params, Qel, E_mag, E_vec, V, "Example6_Franco")
```

### HDF5 Data Handling

Simulation results can be stored with `save_simulation` in `.h5` files using HDF5.jl. This format is efficient, portable, and well-suited for large numerical datasets. The data structure is similar to what you would see in the MATLAB workspace, making it intuitive to explore.

You can inspect these files using HDFView:

- <https://www.hdfgroup.org/download-hdfview/>

### Enabling multiple threads

By default Julia starts with a single thread, so the parallel kernels described in Section 1.3 have no effect. To enable them, Julia must be launched with multiple threads.

**VS Code** — add to `.vscode/settings.json` in the project folder:

```json
{
    "julia.additionalArgs": ["--threads", "auto"]
}
```

**Command line / other environments** — pass the flag directly or set the environment variable:

```bash
julia --threads auto example1.jl
# or
export JULIA_NUM_THREADS=auto
```

You can confirm the thread count inside Julia with `Threads.nthreads()`.

## 3. Literature

[1] Quasi-Static Electromagnetic Dosimetry: From Basic Principles to Examples of Applications

[2] Quasi-Static Approximation Error of Electric Field Analysis for Transcranial Current Stimulation

[3] Numerical Study of Hyper‐Thermic Laser Lipolysis With 1,064 nm Nd:YAG Laser in Human Subjects
