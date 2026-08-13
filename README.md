# Navier–Stokes-Based Shape Optimization of Axisymmetric Bodies

A Python framework for CFD-based shape optimization of axisymmetric bodies using the steady incompressible Navier–Stokes equations.

## Video Demo

[Watch the video demo](https://youtu.be/jKckAemR8v4)

## Description

This project implements a computational framework for the shape optimization of axisymmetric bodies using the steady incompressible Navier–Stokes equations.

The objective is to identify body geometries that minimize the drag coefficient while satisfying geometric constraints.

The complete workflow is implemented programmatically in Python, including:

- geometric parameterization.
- automatic mesh generation.
- finite element solution of the Navier–Stokes equations.
- direct computation of pressure and viscous drag.
- global optimization using Differential Evolution.
- geometry and optimization-history visualization.
- numerical result export.

The implementation uses FEniCSx/DOLFINx, Gmsh, PETSc, MPI, and SciPy.

---

## Features

The program performs the following tasks:

1. Parameterizes an axisymmetric body using one of several nose models.
2. Constructs the complete body profile, consisting of the nose and cylindrical sections.
3. Checks geometric admissibility before running the CFD solver.
4. Generates an unstructured computational mesh using Gmsh.
5. Solves the steady incompressible Navier–Stokes equations using the finite element method.
6. Computes pressure and viscous drag directly from the stress tensor.
7. Optimizes the geometry using Differential Evolution.
8. Tracks the geometry and drag during the optimization.
9. Saves the optimized geometry and numerical results.

---

## Mathematical Model

The flow is modeled as:

- incompressible.
- laminar.
- steady.
- axisymmetric.

The steady incompressible Navier–Stokes equations are

$$
\rho(\mathbf{u}\cdot\nabla) \mathbf{u} = -\nabla p + \mu\nabla^2\mathbf{u}
$$

together with the continuity equation

$$
\nabla\cdot\mathbf{u}=0.
$$

The implementation uses an axisymmetric formulation, reducing the computational cost compared with a full three-dimensional simulation while retaining the relevant physics for bodies of revolution.

---

## Shape Parameterization

Several nose parameterizations are available:

- Tangent Ogive.
- Modified Power Series.
- Quintic Bézier.
- 2C-NS model, included for comparison.

The nose geometry is parameterized as a function of the axial coordinate,

$$
r = r(x),
$$

with geometric constraints imposed to ensure an admissible transition between the nose and cylindrical body.

For the optimized parameterizations, the design variables are the corresponding shape coefficients or control variables.

---

## Geometric Constraints

Candidate geometries are checked before the CFD simulation.

The geometric constraints include:

- positive radius.
- non-negative profile slope.
- monotonic radial growth.
- absence of self-intersections.
- continuity at the nose-cylinder junction.
- zero slope at the nose-cylinder junction when required by the selected parameterization.

Invalid geometries receive a penalty in the optimization objective and are discarded before unnecessary CFD calculations are performed.

---

## CFD Solver

The CFD solver is implemented using FEniCSx/DOLFINx.

The finite element formulation uses second-order elements for velocity and first-order elements for pressure.

The computational domain includes the body surface, inlet, outlet, far-field boundary, and axis of symmetry.

Boundary conditions include:

- prescribed inlet/free-stream velocity.
- far-field condition.
- no-slip condition at the body wall.
- symmetry condition at the axis.
- zero-gauge pressure at the outlet.

To improve nonlinear convergence, a continuation (homotopy) procedure is used. The simulation initially solves the Stokes problem and progressively increases the convective contribution until the full Navier–Stokes equations are reached.

---

## Drag Computation

The drag force is obtained directly by integrating the fluid stress tensor over the body surface.

The solver separately evaluates:

- pressure drag.
- viscous drag.
- total drag.
- total drag coefficient.

The drag coefficient is computed using the reference area and free-stream dynamic pressure.

No external CFD software or auxiliary wall-function post-processing is required.

---

## Optimization

The shape optimization is performed using the Differential Evolution algorithm implemented in SciPy.

The objective function is

$$
J =
C_D
+
P_{\mathrm{geometry}}
+
\lambda\|\mathbf{a}\|^2,
$$

where:

- $C_D$ is the total drag coefficient.
- $P_{\mathrm{geometry}}$ is the geometric penalty.
- $\mathbf{a}$ represents the shape design variables.
- $\lambda$ is the regularization coefficient.

Each candidate geometry is evaluated through a complete CFD simulation.

Consequently, the optimization is directly coupled to the Navier–Stokes solver.

---

## Project Structure

Although the implementation is contained in a single Python file, the code is organized into logical components:

- geometry and flow definitions.
- shape parameterizations.
- geometric constraints.
- mesh generation.
- axisymmetric operators.
- Navier–Stokes solver.
- drag computation.
- optimization objective.
- optimization monitoring.
- Differential Evolution optimization.
- result export.
- main execution.

---

## Installation

The complete CFD environment requires Linux or WSL and is managed using **Miniforge/Mamba**.

The recommended installation method is to create the environment from the provided `environment.yml` file.

## 1. Install system prerequisites

On Ubuntu or WSL:

```bash
sudo apt update
sudo apt upgrade -y

sudo apt install -y wget git build-essential
```

## 2. Install Miniforge

Download Miniforge:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O miniforge.sh
```

Install it:

```bash
bash miniforge.sh
```

Restart the shell or load the Conda configuration:

```bash
source ~/.bashrc
```

Verify the installation:

```bash
conda --version
mamba --version
```

## 3. Create the environment

The complete CFD environment is specified in `environment.yml`.

Create it with:

```bash
mamba env create -f environment.yml
```

Activate the environment:

```bash
mamba activate optenv
```

## 4. Verify the installation

Run:

```bash
python - <<'PY'
import dolfinx
import ufl
import basix
import gmsh

from mpi4py import MPI
from petsc4py import PETSc

print("DOLFINx:", dolfinx.__version__)
print("UFL:", ufl.__version__)
print("Basix:", basix.__version__)
print("Gmsh:", gmsh.__version__)
print("MPI:", MPI.Get_version())
print("PETSc:", PETSc.Sys.getVersion())

print("Installation successful.")
PY
```

If the final message is printed without errors, the CFD environment is ready.

### Python dependencies

The general Python dependencies are also listed in `requirements.txt`.

For an existing compatible Python environment, they can be installed with:

```bash
pip install -r requirements.txt
```

For the complete DOLFINx/PETSc/MPI environment, use `environment.yml` instead.

---

## Usage

Run the project with:

```bash
python project.py --mode <mode>
```

where `<mode>` can be one of:

| Mode | Description |
| --- | --- |
| `profile`     | Generates and plots the current axisymmetric body geometry. No mesh generation or CFD simulation is performed.
| `evaluate`    | Builds the computational mesh, solves the Navier–Stokes equations for the current geometry, computes the drag, and saves the results.
| `optimize`    | Performs shape optimization using Differential Evolution. Each candidate geometry is evaluated through a complete Navier–Stokes simulation.

### Examples

Generate the body profile:

```bash
python project.py --mode profile
```

Evaluate the current geometry:

```bash
python project.py --mode evaluate
```

Run the complete optimization:

```bash
python project.py --mode optimize
```

---

## Configuring the Problem

The simulation is configured through three dataclasses located near the beginning of the source code:

```python
geometry = Geometry(...)
flow = Flow(...)
settings = CFDSettings(...)
```

These objects define the body geometry, flow conditions, and CFD solver settings.

The optimization parameters are configured separately in the optimization functions.

---

## Geometry Parameters

The `Geometry` class defines the axisymmetric body.

| Parameter | Description |
| --- | --- |
| `radius`          | Body radius.
| `nose_length`     | Length of the nose section.
| `cylinder_length` | Length of the cylindrical section.
| `nose_points`     | Number of discretization points along the nose profile.
| `cylinder_points` | Number of discretization points along the cylindrical section.
| `nose_model`      | Nose parameterization model.
| `exponent`        | Exponent used by the Modified Power Series model.
| `h`, `n`          | Parameters of the 2C-NS model.
| `control_points`  | Control variables used by the Quintic Bézier parameterization.

### Available nose models

The program currently supports:

```python
NoseShape.TANGENT_OGIVE
NoseShape.MODIFIED_POWER_SERIES
NoseShape.MODEL_2C_NS
NoseShape.QBEZIER
```

The parameterization can be selected by changing:

```python
geometry.nose_model
```

---

## Flow Parameters

The `Flow` class specifies the free-stream conditions.

| Parameter | Description |
| --- | --- |
| `density`             | Fluid density.
| `dynamic_viscosity`   | Dynamic viscosity.
| `velocity`            | Free-stream velocity.
| `pressure`            | Reference pressure.

These parameters determine the Reynolds number and dynamic pressure used in the drag calculation.

---

## CFD Solver Parameters

The `CFDSettings` class controls the computational domain, mesh generation, nonlinear solver, and continuation strategy.

## Computational Domain

| Parameter | Description |
| --- | --- |
| `upstream_body_lengths`   | Inlet distance measured in body lengths.
| `downstream_body_lengths` | Outlet distance measured in body lengths.
| `farfield_body_radii`     | Distance between the body axis and the far-field boundary.

## Mesh Generation

| Parameter | Description |
| --- | --- |
| `wall_samples`            | Number of spline points used to represent the body in Gmsh.
| `mesh_size_wall_ratio`    | Characteristic mesh size near the body wall.
| `mesh_size_far_ratio`     | Characteristic mesh size in the far field.
| `gmsh_terminal`           | Enables Gmsh terminal output when set to `True`.

## Nonlinear Solver

| Parameter | Description |
| --- | --- |
| `snes_relative_tolerance` | Relative convergence tolerance.
| `snes_absolute_tolerance` | Absolute convergence tolerance.
| `snes_max_iterations`     | Maximum number of SNES iterations per continuation stage.
| `use_backtracking`        | Enables Newton backtracking line search.
| `monitor_solver`          | Prints PETSc convergence information.

---

## Continuation (Homotopy)

The nonlinear solver starts with the Stokes equations and gradually introduces the convective term.

This behavior is controlled by:

```python
continuation_factors
```

For example:

```python
(0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
```

causes the solver to progressively increase the strength of the convective term until the full Navier–Stokes equations are reached.

---

## Optimization Parameters

The optimization is performed by the `optimize_shape()` function.

| Parameter | Description |
| --- | --- |
| `bounds`          | Lower and upper limits for each design variable.
| `seed`            | Random seed used by Differential Evolution.
| `max_iterations`  | Maximum number of Differential Evolution generations.
| `population_size` | Population-size multiplier used by Differential Evolution.
| `tolerance`       | Convergence tolerance.
| `polish`          | Performs a final local optimization if enabled.
| `verbose`         | Prints optimization progress.

---

## Objective Function Parameters

The objective function can be customized using:

| Parameter | Description |
| --- | --- |
| `penalty_weight`          | Weight applied to geometric constraint violations.
| `invalid_geometry_cost`   | Cost assigned to invalid geometries.
| `cfd_failure_cost`        | Cost assigned when the CFD solver fails to converge.
| `regularization`          | Regularization coefficient applied to the design variables.

The objective function is:

$$
J =
C_D
+
P_{\mathrm{geometry}}
+
\lambda\|\mathbf{a}\|^2.
$$

---

## Output Files

Depending on the selected mode, the program generates files such as:

| File | Description |
| --- | --- |
| `optimized_profile.csv`   | Coordinates of the optimized body profile.
| `optimized_profile.png`   | Plot of the final optimized geometry.
| `results.json`            | Geometry, flow conditions, optimization results, and drag coefficients.
| `drag_history.png`        | Evolution of the drag coefficient during optimization.
| `profile_XXXXX.png`       | Geometry corresponding to an individual optimization evaluation.

The output directory is created automatically if it does not already exist.

---

## Possible Applications

The framework is based on generic axisymmetric geometries and can therefore be used to investigate the aerodynamic or hydrodynamic design of bodies such as:

- aerospace vehicles.
- underwater vehicles.
- projectiles.
- capsules.
- pressure vessels.
- streamlined industrial components.

The optimization framework is not restricted to a particular vehicle geometry.

---

## Future Improvements

Potential extensions include:

- compressible flow.
- turbulence models.
- transient simulations.
- adjoint-based optimization.
- multi-objective optimization.
- three-dimensional geometries.
- parallel optimization.
- higher-order shape parameterizations.

---

## Motivation

This project combines computational fluid dynamics, numerical optimization, and scientific computing into a single reproducible framework.

Rather than relying on commercial CFD software, the workflow implements the geometry generation, mesh generation, Navier–Stokes solution, drag computation, and optimization programmatically, providing a transparent and extensible framework for CFD-based shape optimization.

---

## License

Copyright (c) 2026 Thales Coelho Leite Fava

This project is licensed under the MIT License. See the `LICENSE` file for details.
