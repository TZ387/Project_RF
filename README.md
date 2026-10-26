# prrf

A program for RF heating simulation in biological tissue. Given an arbitrary electrode geometry and boundary conditions, it computes the electric potential, electric field, and heat dissipation in the tissue, and then uses that dissipation as a source term to evolve the temperature field over time.

## Requirements

- [Julia](https://julialang.org/) ≥ 1.12
- Optional: an NVIDIA GPU with a current driver, for `use_gpu = true`
- Dependencies are listed in `Project.toml` and managed by Julia's built-in package manager. To install them, open a Julia REPL in the project directory and run:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

To run the tests:

```julia
Pkg.test()
```

## 1. Theoretical basis

### 1.1 RF electric field

The RF heating procedure uses two or more electrodes to apply alternating electrical current to tissue — one electrode applies an alternating electrical potential, while the other acts as an electrical ground.

At typical RF heating frequencies of around 1 MHz the field in the tissue can be treated as quasi-static [1], and the governing equation for the electric potential is:

$$\nabla \cdot [(\sigma - i\omega\epsilon)\nabla V]=0$$

With $\epsilon = \epsilon' + i\epsilon''$ this is:

$$\nabla \cdot [(\sigma + \omega\epsilon'' - i\omega\epsilon')\nabla V]=0$$

where $\sigma + \omega\epsilon''$ carries the conduction current and dielectric loss, and $\omega\epsilon'$ the displacement current. The displacement current is not neglected: in tissue at MHz frequencies it is a fraction of the conduction current, and in insulators, such as a dielectric layer between electrode and skin (`Example7_PVC.jl`, `Example8_teflon.jl`), it dominates and makes the layer act as a capacitor. Reference [2] shows that the quasi-static approximation holds up to the MHz range, and that neglecting $\epsilon'$ would be the larger source of error.

Here, $\sigma$ [S/m] is electrical conductivity, $\omega = 2\pi f$ [rad/s] is the angular frequency, $V$ [V] is the electric potential, and $\epsilon$ [F/m] is the electrical permittivity of the material.

$V$ is therefore a complex amplitude (phasor): the potential at time $t$ is $\mathrm{Re}[V e^{-i\omega t}]$, so $|V|$ is its amplitude and $\arg V$ its phase. The same holds for $\mathbf{E}$.

Note that $\epsilon$ is the product of the vacuum permittivity $\epsilon_0 \approx 8.85 \times 10^{-12}$ F/m and the relative permittivity $\epsilon_r$:

$$D = \epsilon E = \epsilon_0 \epsilon_r E$$

The code takes relative permittivities: `epsilon_re` is $\epsilon'/\epsilon_0$ and `epsilon_im` is $\epsilon''/\epsilon_0$.

Tissue conductivities and relative permittivities at a given frequency can be looked up in [3]. The conductivity listed there already includes the dielectric loss $\omega\epsilon''$, so use it as `sigma` with `epsilon_im = 0`, and the listed relative permittivity as `epsilon_re`, as the examples do.

Once $V$ is known, the electric field is $\mathbf{E} = -\nabla V$ [V/m], and the power density (volumetric heat source) deposited in the tissue by conduction and dielectric losses is:

$$Q_{el} = \frac{1}{2} (\sigma + \omega\epsilon'') |\mathbf{E}|^2 \quad \text{[W/m}^3\text{]}, \qquad |\mathbf{E}|^2 = |E_x|^2 + |E_y|^2 + |E_z|^2$$

The displacement current deposits no heat; $\epsilon'$ only changes how the field is distributed. $V$ and $\mathbf{E}$ are peak amplitudes, so $|\mathbf{E}|$ is $\sqrt{2}$ times the RMS field; with RMS values the factor $\frac{1}{2}$ drops out.

The potential is computed with linear finite elements on a uniform hexahedral grid, using a matrix-free COCG solver [6], the variant of the conjugate gradient method for complex symmetric systems.

### 1.2 Heat equation

The temperature field $T$ is evolved using the classic heat equation with $Q_{el}$ as a volumetric source. This is the bioheat equation used in [4] without the blood-perfusion term:

$$\text{VHC}(\mathbf{x})\, \frac{\partial T}{\partial t} = \nabla \cdot [k(\mathbf{x})\, \nabla T] + Q(\mathbf{x},t)$$

where VHC $= \rho c$ [J/(m³·K)] is the volumetric heat capacity and $k$ [W/(m·K)] is the thermal conductivity, both spatially varying according to the tissue geometry. All outer boundaries are insulated (zero heat flux).

The simulation follows an arbitrary **heating schedule** — an ordered list of `(:on, duration)` and `(:off, duration)` phases, for example:

```julia
schedule = [(:on, 30.0), (:off, 60.0), (:on, 15.0)]
```

During `:on` phases $Q = Q_{el}$; during `:off` phases $Q = 0$.

The equation is solved with finite differences and explicit Euler time stepping. The time step $\Delta t$ is chosen automatically to satisfy the stability criterion:

$$\Delta t \leq \frac{\text{VHC}}{\sum_{f} k_f / \Delta_f^2}$$

evaluated over all cells, with a safety factor of 0.9. The sum runs over the faces of a cell, $k_f$ is the harmonic mean of the conductivities of the two cells sharing face $f$, and $\Delta_f$ is the grid spacing across it.

### 1.3 Parallelisation

The heat update kernel is parallelised (if this is enabled, see section Enabling Multiple Threads below) by partitioning the grid's z-dimension into chunks, one per Julia thread. The thread count is printed at the start of each phase.

The RF solver's matrix-vector product is parallelised over alternating z-layers, and the COCG vector updates over chunks of the vectors; both use all Julia threads.

With `use_gpu = true` (see Usage) the heat update and the COCG iterations of the RF solve run on an NVIDIA GPU through [CUDA.jl](https://cuda.juliagpu.org/), in double precision like the CPU code, with one GPU thread per cell in the heat update and the matrix-vector product. On an RTX 2060 SUPER compared with eight CPU threads, the heat simulation on Example 1's grid is about ten times faster, and `run_simulation` on Example 6's grid about six times. The first GPU run in a Julia session takes about 15 s longer, while CUDA starts and compiles the kernels.

## 2. Usage

The scripts in `Examples/` are complete set-ups meant to be copied and changed; `Example1.jl` runs the whole workflow, and `Example6_Franco.jl` uses the electrode potential of [5]. Each example has a `use_gpu` flag at the start of `main` (Section 1.3).

The workflow consists of two steps that can be run independently.

### Step 1 — RF simulation

Run `run_simulation` to obtain the electric field results and the heat source `Qel`:

```julia
grid, V_dof, Qel, E_mag, E_vec, V = run_simulation(grid_params, rf_params, boundary_conditions)
```

`boundary_conditions` assigns a potential to faces of the domain (`"top"`, `"bottom"`, `"left"`, `"right"`, `"front"`, `"back"`); faces that are not listed, and parts of a face where the function returns `nothing`, are insulating. Each entry is a function of the position `x`, which is centred in x and y and runs in z, the depth into the tissue, from 0 at the `"top"` face to `lz` at the `"bottom"` face:

```julia
boundary_conditions = Dict(
    "top"    => (x, t) -> 10.0,   # electrode at 10 V
    "bottom" => (x, t) -> 0.0,    # ground
)
```

`Qel` and `E_mag` are cell arrays of size `(nx, ny, nz)`, `E_vec` is `(3, nx, ny, nz)`, and `V` is the potential at the grid nodes, `(nx+1, ny+1, nz+1)`. `V` and `E_vec` are complex amplitudes (Section 1.1): `abs.(V)` is the amplitude of the potential and `angle.(V)` its phase. `plot_graphs` shows $|V|$, and in each cell the direction of the field at the instant it is strongest there.

Pass `use_gpu = true` to solve on an NVIDIA GPU (Section 1.3); the results are returned as ordinary arrays either way.

### Step 2 — Heat simulation

Pass `Qel` to `run_heat_simulation` to evolve the temperature field:

```julia
T_final = run_heat_simulation(Qel, grid_params, heat_params)
```

Set `create_timelapse = true` to display a live-updating cross-section plot of the temperature field as the simulation runs:

```julia
T_final = run_heat_simulation(Qel, grid_params, heat_params; create_timelapse = true)
```

The plot refreshes `n_update` times per schedule phase. The colour scale is updated automatically to the current field range at each refresh, unless `T_plot_min` and `T_plot_max` are set in `HeatParams`.

Set `use_gpu = true` to run the time stepping on an NVIDIA GPU (Section 1.3); it works with or without the live plot.

### Saving plots

To save the plots, pass a path prefix to `plot_graphs`; this writes `Images/Example6_Franco_a.png` to `Images/Example6_Franco_e.png`:

```julia
plot_graphs(material_indices, grid_params, Qel, E_mag, E_vec, V, "Images/Example6_Franco")
```

### HDF5 Data Handling

Simulation results can be stored in `.h5` files with `save_simulation` and read back with `load_simulation`, for example to run the heat simulation later without repeating the RF solve:

```julia
save_simulation("Example1.h5", grid_params, material_indices; Qel=Qel, E_mag=E_mag, V=V)
grid_params, material_indices, data = load_simulation("Example1.h5")   # data.Qel, data.E_mag, data.V
```

Complex arrays such as `V` are stored as their real and imaginary parts. The files can be inspected with [HDFView](https://www.hdfgroup.org/download-hdfview/).

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
julia --project=. --threads auto Examples/Example1.jl
# or
export JULIA_NUM_THREADS=auto
```

You can confirm the thread count inside Julia with `Threads.nthreads()`.

## 3. Literature

[1] D. Andreuccetti and N. Zoppetti, "Quasi-Static Electromagnetic Dosimetry: From Basic Principles to Examples of Applications", *Int. J. Occup. Saf. Ergon.* 12(2), 201–215 (2006). [doi:10.1080/10803548.2006.11076682](https://doi.org/10.1080/10803548.2006.11076682)

[2] G. Gaugain, L. Quéguiner, M. Bikson, R. Sauleau, M. Zhadobov, J. Modolo and D. Nikolayev, "Quasi-static approximation error of electric field analysis for transcranial current stimulation", *J. Neural Eng.* 20, 016027 (2023). [doi:10.1088/1741-2552/acb14d](https://doi.org/10.1088/1741-2552/acb14d)

[3] D. Andreuccetti, R. Fossi and C. Petrucci, "An Internet resource for the calculation of the dielectric properties of body tissues in the frequency range 10 Hz – 100 GHz", IFAC-CNR, Florence (1997). Based on data published by C. Gabriel et al. in 1996. <http://niremf.ifac.cnr.it/tissprop/>

[4] M. Milanic, B. Tasic Muc, N. Lukac and M. Lukac, "Numerical Study of Hyper‐Thermic Laser Lipolysis With 1,064 nm Nd:YAG Laser in Human Subjects", *Lasers Surg. Med.* 51(10), 897–909 (2019). [doi:10.1002/lsm.23124](https://doi.org/10.1002/lsm.23124)

[5] W. Franco, A. Kothare, S. J. Ronan, R. C. Grekin and T. H. McCalmont, "Hyperthermic Injury to Adipocyte Cells by Selective Heating of Subcutaneous Fat With a Novel Radiofrequency Device: Feasibility Studies", *Lasers Surg. Med.* 42, 361–370 (2010). [doi:10.1002/lsm.20925](https://doi.org/10.1002/lsm.20925)

[6] H. A. van der Vorst and J. B. M. Melissen, "A Petrov–Galerkin type method for solving $Ax = b$, where $A$ is symmetric complex", *IEEE Trans. Magn.* 26(2), 706–708 (1990). [doi:10.1109/20.106415](https://doi.org/10.1109/20.106415)
