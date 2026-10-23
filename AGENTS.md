# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project

`prrf` is a Julia package (Julia ≥ 1.12) for simulating RF heating in biological tissue. It solves for the electric potential on a 3D voxel grid, derives the electric field and the heat source `Qel`, and then evolves the temperature field over time. Theory and usage are in [README.md](README.md).

## Layout

- `src/prrf.jl` — top-level module. Includes the submodules and re-exports the public API; a new public function must be exported from its submodule and from here.
- `src/Config.jl` — `GridParams`, `RFParams`, `HeatParams`, material property setup, coordinate grids.
- `src/GridSetup.jl` — Ferrite hexahedral grid.
- `src/RFSolver.jl` — RF potential solve (matrix-free Jacobi-preconditioned CG, on the CPU or a CUDA GPU) and field calculation.
- `src/HeatSolver.jl` — explicit finite-difference heat solver, CPU-parallel or on a CUDA GPU.
- `src/Timelapse.jl` — live GLMakie plot of the heat simulation.
- `src/Plotting.jl` — slice plots and the E-field direction plot.
- `src/RunSimulation.jl` — the two entry points, `run_simulation` and `run_heat_simulation`.
- `src/SimulationIO.jl` — HDF5 save and load.
- `Examples/` — runnable scripts for end users (see below).
- `test/` — test suite, entry point `runtests.jl`. `test/Project.toml` is part of a Pkg workspace and shares the root `Manifest.toml`.

## Less is more

The codebase is meant to stay lean. Prefer the solution with the least code, as long as it stays readable.

- **New features.** Add only the code the feature needs. No boilerplate, and no options or abstractions for cases that do not exist yet.
- **Existing code.** If you see something that can be written more efficiently and with less code, take the opportunity. Make it its own commit, separate from the task you were given. The repetition between example scripts is the exception (see below).
- **Comments.** Keep them few and short. Comment only what the code does not make obvious; no long explanatory blocks, and no comments that restate the code.

## Examples are deliberately simple

The scripts in `Examples/` are meant for end users to open and change settings in: geometry, materials, grid size, boundary conditions, heat schedule. They are kept simple on purpose.

- Keep each example a flat, self-contained script with its values inline and commented.
- Do not refactor them: no shared helper file to remove the repetition between examples, no config files or command-line arguments, no extra abstraction or validation.
- When a change in `src/` alters the public API, update every affected example in the same commit, and change only what the API change requires.

## Git

- Commit locally only. Never push or otherwise write to the remote (no `git push`, no pull requests). The maintainer pushes to GitHub.
- One commit is one specific thing. Do not combine unrelated fixes or changes; split them into separate commits. A single change that touches many files, such as a rename across `src/` and all examples, is still one commit.
- Keep commit messages short. A simple commit gets one short subject line and no body. Add a body only when the change needs explaining, such as a solver or performance change.
- Follow the existing history: imperative mood, capitalised, no prefix tags, no trailing full stop. For example, `Rename V_new as V` or `Add License`.

## Running

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'    # install dependencies
julia --project=. -e 'using prrf'                      # check that the package loads
julia --project=. -e 'using Pkg; Pkg.test()'           # run the tests
julia --project=. --threads auto Examples/Example1.jl  # run an example
```

- Run the tests after changing `src/` or `Examples/`. They take about a minute. There is no CI, and none should be added; the tests are run locally.
- `test/examples.jl` runs every example with its solve, plots and file output replaced by stubs, then solves the example's set-up on a coarse grid. It relies on each example defining `geometryDefinition` and `define_material_properties` and calling `run_simulation`, as the existing ones do.
- The first load precompiles GLMakie and can take several minutes.
- The examples open GLMakie windows, so they need a display, and they solve grids of 2 to 6.5 million cells. They are not a quick check.

## Things to know before changing the code

- **Array shapes.** Cell fields (`sigma`, `Qel`, `E_mag`, `T`, `material_indices`) are `(nx, ny, nz)`. The potential `V` is nodal, `(nx+1, ny+1, nz+1)`. `E_vec` is `(3, nx, ny, nz)`.
- **Coordinates.** The Ferrite grid, the `x` passed to boundary-condition functions, the cell centres from `create_coordinate_grids` and the plots are all centred in x and y (`[-l/2, l/2]`) and run `[0, lz]` in z. z is depth, so `setup_grid` swaps Ferrite's `"top"` and `"bottom"` face names to make `"top"` the z = 0 face.
- **Uniform grid assumption.** `solve_rf` computes one reference element stiffness matrix and reuses it for every cell, `calculate_fields` takes E from edge differences of the nodal V, and `convert_V` relies on the nodes being numbered x fastest. All three are only valid for the uniform hex grid that `setup_grid` produces.
- **Threading.** The heat solver runs inside a `Threads.@spawn` task, where `Threads.@threads` deadlocks, and GLMakie observables may only be written from the main thread. Read the header comments in `src/HeatSolver.jl` and `src/Timelapse.jl` before changing either.
- **GPU.** `use_gpu = true` runs the heat time stepping and the RF solver's CG iterations with CUDA.jl, in Float64 like the CPU code. The CPU and GPU paths share the per-cell code (`stepped_temperature`, `cell_product`); the GPU matvec adds cell contributions with atomics. `test/gpu.jl` compares GPU with CPU results and is skipped when no usable GPU is present.
- **Ignored files.** `*.h5`, `Images/`, `Results/`, `.vscode/`, `prompt.txt` and `benchmark_solvers.jl` are git-ignored output or local scratch. Do not force-add them.
- **Dependencies.** `Manifest.toml` is committed and also covers `test/Project.toml`, and every dependency in `Project.toml` has a `[compat]` entry. Change them only when the task is a dependency change.
