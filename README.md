# Navier–Stokes-Based Shape Optimization of Axisymmetric Bodies

#### Video Demo: https://youtu.be/jKckAemR8v4

#### Description

This project implements a computational framework for the shape optimization of axisymmetric bodies using the steady incompressible Navier–Stokes equations.

The objective is to identify body geometries that minimize the drag coefficient while satisfying geometric constraints.

The complete workflow is implemented programmatically in Python, including:

- geometric parameterization;
- automatic mesh generation;
- finite element solution of the Navier–Stokes equations;
- direct computation of pressure and viscous drag;
- global optimization using Differential Evolution;
- geometry and optimization-history visualization;
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

- incompressible;
- laminar;
- steady;
- axisymmetric.

The steady incompressible Navier–Stokes equations are

$$\rho(\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \mu\nabla^2\mathbf{u},$$

together with the continuity equation

$$\nabla\cdot\mathbf{u}=0.$$

The implementation uses an axisymmetric formulation, reducing the computational cost compared with a full three-dimensional simulation while retaining the relevant physics for bodies of revolution.

---

## Shape Parameterization

Several nose parameterizations are available:

- Tangent Ogive;
- Modified Power Series;
- Quintic Bézier;
- 2C-NS model, included for comparison.

The nose geometry is parameterized as a function of the axial coordinate,

$$r = r(x),$$

with geometric constraints imposed to ensure an admissible transition between the nose and cylindrical body.

For the optimized parameterizations, the design variables are the corresponding shape coefficients or control variables.

---

## Geometric Constraints

Candidate geometries are checked before the CFD simulation.

The geometric constraints include:

- positive radius;
- non-negative profile slope;
- monotonic radial growth;
- absence of self-intersections;
- continuity at the nose-cylinder junction;
- zero slope at the nose-cylinder junction when required by the selected parameterization.

Invalid geometries receive a penalty in the optimization objective and are discarded before unnecessary CFD calculations are performed.

---

## CFD Solver

The CFD solver is implemented using FEniCSx/DOLFINx.

The finite element formulation uses second-order elements for velocity and first-order elements for pressure.

The computational domain includes the body surface, inlet, outlet, far-field boundary, and axis of symmetry.

Boundary conditions include:

- prescribed inlet/free-stream velocity;
- far-field condition;
- no-slip condition at the body wall;
- symmetry condition at the axis;
- zero-gauge pressure at the outlet.

To improve nonlinear convergence, a continuation (homotopy) procedure is used. The simulation initially solves the Stokes problem and progressively increases the convective contribution until the full Navier–Stokes equations are reached.

---

## Drag Computation

The drag force is obtained directly by integrating the fluid stress tensor over the body surface.

The solver separately evaluates:

- pressure drag;
- viscous drag;
- total drag;
- total drag coefficient.

The drag coefficient is computed using the reference area and free-stream dynamic pressure.

No external CFD software or auxiliary wall-function post-processing is required.

---

## Optimization

The shape optimization is performed using the Differential Evolution algorithm implemented in SciPy.

The objective function is

$$J = C_D + P_{\mathrm{geometry}} + \lambda\|\mathbf{a}\|^2,$$

where:

- $C_D$ is the total drag coefficient;
- $P_{\mathrm{geometry}}$ is the geometric penalty;
- $\mathbf{a}$ represents the shape design variables;
- $\lambda$ is the regularization coefficient.

Each candidate geometry is evaluated through a complete CFD simulation.

Consequently, the optimization is directly coupled to the Navier–Stokes solver.

---

## Project Structure

Although the implementation is contained in a single Python file, the code is organized into logical components:

- geometry and flow definitions;
- shape parameterizations;
- geometric constraints;
- mesh generation;
- axisymmetric operators;
- Navier–Stokes solver;
- drag computation;
- optimization objective;
- optimization monitoring;
- Differential Evolution optimization;
- result export;
- main execution.

---

## Installation

The complete CFD environment requires Linux or WSL and is managed using **Miniforge/Mamba**.

The recommended installation method is to create the environment from the provided `environment.yml` file.

### 1. Install system prerequisites

On Ubuntu or WSL:

```bash
sudo apt update
sudo apt upgrade -y

sudo apt install -y wget git build-essential
