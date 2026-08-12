#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Navier-Stokes-Based Shape Optimization of Axisymmetric Bodies

A self-contained framework for CFD-based shape optimization using the
steady incompressible axisymmetric Navier-Stokes equations and the
finite element method.

Features
--------
- Geometry parameterization
- Automatic mesh generation with Gmsh
- Axisymmetric Navier-Stokes solver (FEniCSx)
- Drag computation
- Differential Evolution optimization
- Automatic result export

Author
------
Thales Coelho Leite Fava

Copyright (c) 2026 Thales Coelho Leite Fava

License
-------
MIT License
See the LICENSE file in the project root for details.
"""

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

import json
import uuid
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from enum import Enum, auto
from typing import NamedTuple
from dataclasses import asdict, dataclass
from scipy.optimize import differential_evolution

# =============================================================================
# Saving results and monitoring convergence
# =============================================================================


class OptimizationMonitor:
    """Global counter of optimization steps."""

    def __init__(self, output_directory: Path) -> None:
        self.counter = 0
        self.output_directory = output_directory

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.drag_history = []

    def advance(self):

        self.counter += 1

        return self.counter

    def plot_geometry(
        self,
        coefficients: np.ndarray,
        geometry: Geometry,
    ) -> None:
        """Save geometry drawing during optimization."""

        self.advance()

        (
            x,
            r,
            _,
            _,
            _,
        ) = body_profile(
            coefficients,
            geometry,
        )

        plt.figure(figsize=(9.0, 4.5))

        plt.plot(
            x,
            r,
            linewidth=2,
        )

        plt.plot(
            x,
            -r,
            linewidth=2,
        )

        plt.xlabel("x [m]")
        plt.ylabel("r [m]")
        plt.axis("equal")
        plt.grid(True)
        plt.title(f"Evaluation {self.counter}\n" f"Coefficients = {coefficients}")
        plt.tight_layout()
        filename = self.output_directory / f"profile_{self.counter:05d}.png"
        plt.savefig(
            filename,
            dpi=180,
        )
        plt.close()

    def plot_drag_history(self):

        plt.figure(figsize=(9.0, 4.5))

        plt.plot(
            self.drag_history,
            "-o",
            linewidth=2,
            markersize=4,
        )

        plt.xlabel("Evaluation")
        plt.ylabel(r"$C_D$")
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            self.output_directory / "drag_history.png",
            dpi=180,
        )

        plt.close()


def save_profile_figure(
    coefficients: np.ndarray,
    geometry: Geometry,
    output_path: Path,
) -> None:
    """Save optimized geometry drawing."""
    (
        x,
        r,
        _,
        _,
        _,
    ) = body_profile(
        coefficients,
        geometry,
    )

    plt.figure(figsize=(9.0, 4.5))

    plt.plot(
        x,
        r,
        label="Upper profile",
    )

    plt.plot(
        x,
        -r,
        label="Lower profile",
    )

    plt.axvline(
        geometry.nose_length,
        linestyle="--",
        label="Nose-cylinder junction",
    )

    plt.xlabel("x [m]")
    plt.ylabel("r [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=180,
    )
    plt.close()


def save_results(
    coefficients: np.ndarray,
    geometry: Geometry,
    flow: Flow,
    settings: CFDSettings,
    output_directory: Path,
    optimization_result=None,
    run_cfd: bool = True,
) -> DragResult | None:
    """Save profile, figure, parameters, and optionally CFD results"""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        x,
        r,
        drdx,
        _,
        _,
    ) = body_profile(
        coefficients,
        geometry,
    )

    profile_path = output_directory / "optimized_profile.csv"

    np.savetxt(
        profile_path,
        np.column_stack(
            (
                x,
                r,
                drdx,
            )
        ),
        delimiter=",",
        header="x_m,r_m,drdx",
        comments="",
    )

    figure_path = output_directory / "optimized_profile.png"

    save_profile_figure(
        coefficients,
        geometry,
        figure_path,
    )

    drag_result: DragResult | None = None

    if run_cfd:
        drag_result = axisymmetric_navier_stokes_drag(
            coefficients,
            geometry,
            flow,
            settings,
        )

    geometry_data = asdict(geometry)
    geometry_data["nose_model"] = geometry.nose_model.value

    data = {
        "coefficients": {
            "a1": float(coefficients[0]),
            "a2": float(coefficients[1]),
        },
        "geometry": geometry_data,
        "flow": asdict(flow),
        "cfd_settings": asdict(settings),
        "drag": (drag_result._asdict() if drag_result is not None else None),
    }

    if optimization_result is not None:
        data["optimization"] = {
            "success": bool(optimization_result.success),
            "message": str(optimization_result.message),
            "objective": float(optimization_result.fun),
            "iterations": int(optimization_result.nit),
            "function_evaluations": int(optimization_result.nfev),
        }

    json_path = output_directory / "results.json"

    json_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return drag_result


# =============================================================================
# Geometry Parameterization
# =============================================================================


class NoseShape(Enum):
    """Selection of the nose parameterization model."""

    TANGENT_OGIVE = auto()
    MODIFIED_POWER_SERIES = auto()
    MODEL_2C_NS = auto()
    QBEZIER = auto()


@dataclass(frozen=True)
class Geometry:
    """Geometric parameters of nose and cylinder."""

    radius: float = 0.050
    nose_length: float = 0.200
    cylinder_length: float = 0.300
    nose_points: int = 501
    cylinder_points: int = 301

    nose_model: NoseShape = NoseShape.QBEZIER

    exponent: float = 1.0

    h: float = 0.2
    n: float = 0.5

    control_points = np.array(
        [
            0.20 * nose_length,
            0.08 * radius,
            0.40 * nose_length,
            0.30 * radius,
            0.60 * nose_length,
            0.65 * radius,
            0.80 * nose_length,
        ]
    )

    def validate(self) -> None:
        """Check whether geometry parameters are admissible."""
        if self.radius <= 0.0:
            raise ValueError("geometry.radius should be positive.")

        if self.cylinder_length < 0.0:
            raise ValueError("geometry.cylinder_length should be positive.")

        if self.nose_points < 10:
            raise ValueError("geometry.nose_points should be at least 10.")

        if self.cylinder_points < 2:
            raise ValueError("geometry.cylinder_points should be at least 2.")

    @property
    def body_length(self) -> float:
        """Total length of nose and cylinder."""
        return self.nose_length + self.cylinder_length

    @property
    def reference_area(self) -> float:
        """Frontal reference area pi R²."""
        return np.pi * self.radius**2


def model_2c_ns(
    x: np.ndarray,
    radius: float,
    nose_length: float,
    h: float,
    n: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    2C-NS model of PhD Thesis

    Bertoldo, G., "OTIMIZAÇÃO AERODINÂMICA DE NEWTON COM BASE NAS EQUAÇÕES DE NAVIER-STOKES", 2014.

    r(x) = R [h^(1/n) + (1-h^(1/n)) x/L]^n

    This model does not ensure tangency between nose and cylinder at x=L.
    """
    if not (0.0 < h < 1.0):
        raise ValueError("In the 2C-NS model, use 0 < h < 1.")

    if not (0.0 < n <= 1.0):
        raise ValueError("In the 2C-NS model, use 0 < n <= 1.")

    x = np.asarray(x, dtype=float)
    xi = x / nose_length

    a = h ** (1.0 / n)
    base = a + (1.0 - a) * xi

    radius_values = radius * base**n
    slope_values = radius * n * (1.0 - a) / nose_length * base ** (n - 1.0)

    return radius_values, slope_values


def tangent_ogive(
    x: np.ndarray,
    radius: float,
    nose_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Circular tangent ogive.

    rho_o = (L² + R²)/(2R)

    r_TO(x) =
        sqrt(rho_o² - (L-x)²) + R - rho_o

    Satisfied conditions:
        r_TO(0) = 0
        r_TO(L) = R
        r_TO'(L) = 0
    """
    x = np.asarray(x, dtype=float)

    if radius <= 0.0:
        raise ValueError("radius should be positive.")

    if nose_length <= radius:
        raise ValueError("nose_length should be larger than radius.")

    rho_ogive = (nose_length**2 + radius**2) / (2.0 * radius)

    radicand = rho_ogive**2 - (nose_length - x) ** 2

    if np.any(radicand < -1.0e-12):
        raise ValueError("Coordinates outside valid interval for tangent ogive.")

    root = np.sqrt(np.maximum(radicand, 0.0))

    radius_values = root + radius - rho_ogive

    slope_values = (nose_length - x) / np.maximum(root, 1.0e-15)

    return radius_values, slope_values


def modified_power_series(
    x: np.ndarray,
    radius: float,
    nose_length: float,
    exponent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Modified power series.

    r(x) = R [1 - (1 - x/L)^n]

    where

        R = cylinder radius
        L = nose length
        n = exponent

    Satisfied conditions:

        r(0) = 0
        r(L) = R
        dr/dx(L) = 0      (for n > 1)
    """

    if radius <= 0.0:
        raise ValueError("radius should be positive.")

    if nose_length <= 0.0:
        raise ValueError("nose_length should be positive.")

    if exponent <= 1.0:
        warnings.warn("To ensure tangency, use n > 1.", UserWarning)

    x = np.asarray(
        x,
        dtype=float,
    )

    xi = x / nose_length

    radius_values = radius * (1.0 - (1.0 - xi) ** exponent)

    slope_values = radius * exponent / nose_length * (1.0 - xi) ** (exponent - 1.0)

    return (
        radius_values,
        slope_values,
    )


def quintic_bezier(
    x: np.ndarray,
    radius: float,
    nose_length: float,
    control_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quintic Bézier parameterization.

    Design variables

        control_points =

        [
            P1x,
            P1r,

            P2x,
            P2r,

            P3x,
            P3r,

            P4x,
        ]

    Boundary conditions

        P0 = (0,0)

        P4 = (P4x,R)

        P5 = (L,R)

    Satisfied conditions:

        r(0)=0
        r(L)=R
        dr/dx(L)=0
    """

    if radius <= 0:
        raise ValueError("radius should be positive.")

    if nose_length <= 0:
        raise ValueError("nose_length should be positive.")

    cp = np.asarray(
        control_points,
        dtype=float,
    )

    if cp.size != 7:
        raise ValueError("Expected seven design variables.")

    x = np.asarray(
        x,
        dtype=float,
    )

    t = x / nose_length

    omt = 1.0 - t

    P1x, P1r, P2x, P2r, P3x, P3r, P4x = cp

    Px = np.array(
        [
            0.0,
            P1x,
            P2x,
            P3x,
            P4x,
            nose_length,
        ]
    )

    Pr = np.array(
        [
            0.0,
            P1r,
            P2r,
            P3r,
            radius,
            radius,
        ]
    )

    B0 = omt**5
    B1 = 5 * omt**4 * t
    B2 = 10 * omt**3 * t**2
    B3 = 10 * omt**2 * t**3
    B4 = 5 * omt * t**4
    B5 = t**5

    x_values = (
        B0 * Px[0] + B1 * Px[1] + B2 * Px[2] + B3 * Px[3] + B4 * Px[4] + B5 * Px[5]
    )

    radius_values = (
        B0 * Pr[0] + B1 * Pr[1] + B2 * Pr[2] + B3 * Pr[3] + B4 * Pr[4] + B5 * Pr[5]
    )

    C0 = omt**4
    C1 = 4 * omt**3 * t
    C2 = 6 * omt**2 * t**2
    C3 = 4 * omt * t**3
    C4 = t**4

    dxdt = 5.0 * (
        C0 * (Px[1] - Px[0])
        + C1 * (Px[2] - Px[1])
        + C2 * (Px[3] - Px[2])
        + C3 * (Px[4] - Px[3])
        + C4 * (Px[5] - Px[4])
    )

    drdt = 5.0 * (
        C0 * (Pr[1] - Pr[0])
        + C1 * (Pr[2] - Pr[1])
        + C2 * (Pr[3] - Pr[2])
        + C3 * (Pr[4] - Pr[3])
        + C4 * (Pr[5] - Pr[4])
    )

    slope_values = drdt / np.maximum(dxdt, 1.0e-14)

    return (
        radius_values,
        slope_values,
    )


def nose_parameterization(
    x: np.ndarray,
    geometry: Geometry,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute r(x) and dr/dx according to the selected
    nose parameterization.
    """

    if geometry.nose_model == NoseShape.TANGENT_OGIVE:

        return tangent_ogive(
            x=x,
            radius=geometry.radius,
            nose_length=geometry.nose_length,
        )

    elif geometry.nose_model == NoseShape.MODIFIED_POWER_SERIES:

        return modified_power_series(
            x=x,
            radius=geometry.radius,
            nose_length=geometry.nose_length,
            exponent=geometry.exponent,
        )

    elif geometry.nose_model == NoseShape.MODEL_2C_NS:

        return model_2c_ns(
            x=x,
            radius=geometry.radius,
            nose_length=geometry.nose_length,
            h=geometry.h,
            n=geometry.n,
        )

    elif geometry.nose_model == NoseShape.QBEZIER:

        return quintic_bezier(
            x=x,
            radius=geometry.radius,
            nose_length=geometry.nose_length,
            control_points=geometry.control_points,
        )

    raise ValueError(f"Unsupported nose model: {geometry.nose_model}")


def shape_basis(
    xi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shape functions and derivatives with respect to xi=x/L

    phi_1 = xi²(1-xi)²

    phi_2 = xi²(1-xi)²(2xi-1)

    Satisfied conditions:
        phi_k(0) = phi_k(1) = 0
        phi_k'(0) = phi_k'(1) = 0
    """
    xi = np.asarray(xi, dtype=float)

    phi_1 = xi**2 * (1.0 - xi) ** 2

    dphi_1_dxi = 2.0 * xi * (1.0 - xi) ** 2 - 2.0 * xi**2 * (1.0 - xi)

    phi_2 = phi_1 * (2.0 * xi - 1.0)

    dphi_2_dxi = dphi_1_dxi * (2.0 * xi - 1.0) + 2.0 * phi_1

    phi = np.vstack((phi_1, phi_2))

    dphi_dxi = np.vstack(
        (
            dphi_1_dxi,
            dphi_2_dxi,
        )
    )

    return phi, dphi_dxi


def geometry_penalty(
    r_nose: np.ndarray,
    drdx_nose: np.ndarray,
    geometry: Geometry,
    weight: float = 1.0e4,
    tolerance: float = 1.0e-10,
) -> float:
    """
    Penalize violations of conditions:

        0 <= r(x) <= R
        dr/dx >= 0

    Variables are made nondimensional so that radius and slope penalties have comparable scales.
    """
    r_hat = np.asarray(r_nose, dtype=float) / geometry.radius

    slope_hat = (
        np.asarray(drdx_nose, dtype=float) * geometry.nose_length / geometry.radius
    )

    lower_violation = np.maximum(
        -r_hat - tolerance,
        0.0,
    )

    upper_violation = np.maximum(
        r_hat - 1.0 - tolerance,
        0.0,
    )

    monotonicity_violation = np.maximum(
        -slope_hat - tolerance,
        0.0,
    )

    mean_square_violation = (
        np.mean(lower_violation**2)
        + np.mean(upper_violation**2)
        + np.mean(monotonicity_violation**2)
    )

    return float(weight * mean_square_violation)


def evaluate_geometry_penalty(
    coefficients: np.ndarray,
    geometry: Geometry,
    weight: float = 1.0e4,
) -> float:
    """Compute geometry penalty directly from the coefficients."""
    (
        _,
        _,
        _,
        r_nose,
        drdx_nose,
    ) = body_profile(
        coefficients,
        geometry,
    )

    return geometry_penalty(
        r_nose,
        drdx_nose,
        geometry,
        weight=weight,
    )


def body_profile(
    coefficients: np.ndarray,
    geometry: Geometry,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Build nose-cylinder profile.

    Nose:

        r(x;a) =
            r_TO(x)
            + R sum_k a_k phi_k(x/L_n)

    Cylinder:

        r(x) = R

    Outputs:
        x_total
        r_total
        drdx_total
        r_nose
        drdx_nose
    """
    geometry.validate()

    coefficients = np.asarray(
        coefficients,
        dtype=float,
    )

    if coefficients.shape != (2,):
        raise ValueError("This model uses exactly two coefficients: a1 e a2.")

    x_nose = np.linspace(
        0.0,
        geometry.nose_length,
        geometry.nose_points,
    )

    xi = x_nose / geometry.nose_length

    radius_base, slope_base = nose_parameterization(
        x_nose,
        geometry=geometry,
    )

    phi, dphi_dxi = shape_basis(xi)

    perturbation = coefficients @ phi

    perturbation_slope = coefficients @ dphi_dxi

    r_nose = radius_base + geometry.radius * perturbation

    drdx_nose = slope_base + geometry.radius / geometry.nose_length * perturbation_slope

    x_cylinder = np.linspace(
        geometry.nose_length,
        geometry.body_length,
        geometry.cylinder_points,
    )[1:]

    r_cylinder = np.full_like(
        x_cylinder,
        geometry.radius,
    )

    drdx_cylinder = np.zeros_like(
        x_cylinder,
    )

    x_total = np.concatenate(
        (
            x_nose,
            x_cylinder,
        )
    )

    r_total = np.concatenate(
        (
            r_nose,
            r_cylinder,
        )
    )

    drdx_total = np.concatenate(
        (
            drdx_nose,
            drdx_cylinder,
        )
    )

    return (
        x_total,
        r_total,
        drdx_total,
        r_nose,
        drdx_nose,
    )


# =============================================================================
# Flow Parameters
# =============================================================================


@dataclass(frozen=True)
class Flow:
    """Freestream conditions (SI Units)."""

    density: float = 1.225
    dynamic_viscosity: float = 1.81e-5
    velocity: float = 300.0
    pressure: float = 101325.0

    def validate(self) -> None:
        """Check whether flow properties are admissible."""
        if self.density <= 0.0:
            raise ValueError("flow.density should be positive.")

        if self.dynamic_viscosity <= 0.0:
            raise ValueError("flow.dynamic_viscosity should be positive.")

        if self.velocity <= 0.0:
            raise ValueError("flow.velocity should be positive.")

    @property
    def dynamic_pressure(self) -> float:
        """Freestream dynamic pressure."""
        return 0.5 * self.density * self.velocity**2


# =============================================================================
# Computational domain, mesh, and Navier-Stokes solver parameters
# =============================================================================


@dataclass(frozen=True)
class CFDSettings:
    """Domain, mesh, and nonlinear solver configurations."""

    upstream_body_lengths: float = 3.0
    downstream_body_lengths: float = 8.0
    farfield_body_radii: float = 10.0

    wall_samples: int = 120
    mesh_size_wall_ratio: float = 0.08
    mesh_size_far_ratio: float = 0.60

    snes_relative_tolerance: float = 1.0e-7
    snes_absolute_tolerance: float = 1.0e-8
    snes_max_iterations: int = 60
    use_backtracking: bool = True
    monitor_solver: bool = True

    continuation_factors: tuple[float, ...] = (
        0.0,
        0.02,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
        0.70,
        0.85,
        1.0,
    )

    gmsh_terminal: bool = False

    def validate(self) -> None:
        """Check whether CFD parameters are admissible"""
        if self.upstream_body_lengths <= 0.0:
            raise ValueError("upstream_body_lengths should be positive.")

        if self.downstream_body_lengths <= 0.0:
            raise ValueError("downstream_body_lengths should be positive.")

        if self.farfield_body_radii <= 1.0:
            raise ValueError("farfield_body_radii should be larger than 1.")

        if self.wall_samples < 20:
            raise ValueError("wall_samples should be at least 20.")

        if self.mesh_size_wall_ratio <= 0.0:
            raise ValueError("mesh_size_wall_ratio should be positive.")

        if self.mesh_size_far_ratio <= self.mesh_size_wall_ratio:
            raise ValueError(
                "mesh_size_far_ratio should be larger than mesh_size_wall_ratio."
            )

        if self.snes_relative_tolerance <= 0.0:
            raise ValueError("snes_relative_tolerance should be positive.")

        if self.snes_absolute_tolerance <= 0.0:
            raise ValueError("snes_absolute_tolerance should be positive.")

        if self.snes_max_iterations < 1:
            raise ValueError("snes_max_iterations should be at least one.")

        if len(self.continuation_factors) < 2:
            raise ValueError("continuation_factors should contain at least 0.0 e 1.0.")

        factors = np.asarray(self.continuation_factors, dtype=float)

        if not np.isclose(factors[0], 0.0):
            raise ValueError("The first continuation_factor should be 0.0 (Stokes).")

        if not np.isclose(factors[-1], 1.0):
            raise ValueError(
                "The last continuation_factor should be 1.0 (Full nonlinear term)."
            )

        if np.any(np.diff(factors) <= 0.0):
            raise ValueError("continuation_factors should be increasing.")


class DragResult(NamedTuple):
    """Integrated Navier-Stokes results."""

    cd_total: float
    cd_pressure: float
    cd_viscous: float

    force_total: float
    force_pressure: float
    force_viscous: float

    reynolds: float
    nonlinear_iterations: int


FLUID = 1
INLET = 2
OUTLET = 3
FARFIELD = 4
AXIS = 5
BODY = 6


def resample_profile_by_arclength(
    x: np.ndarray,
    r: np.ndarray,
    number_of_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample profile by arclength to better distribute the Gmsh
    spline points than a resampling based only on the original
    index.
    """
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)

    if x.ndim != 1 or r.ndim != 1:
        raise ValueError("x and r should be unidimensional vectors.")

    if x.shape != r.shape:
        raise ValueError("x and r should have the same length.")

    if x.size < 2:
        raise ValueError("The profile needs at lest two points.")

    if number_of_points < 2:
        raise ValueError("number_of_points should be at least 2.")

    dx = np.diff(x)
    dr = np.diff(r)

    arc_increment = np.sqrt(dx**2 + dr**2)

    arc_coordinate = np.concatenate(
        (
            np.array([0.0]),
            np.cumsum(arc_increment),
        )
    )

    total_arc = arc_coordinate[-1]

    if total_arc <= 0.0:
        raise ValueError("Profile arclength is zero.")

    normalized_arc = arc_coordinate / total_arc

    normalized_arc_new = np.linspace(
        0.0,
        1.0,
        number_of_points,
    )

    x_new = np.interp(
        normalized_arc_new,
        normalized_arc,
        x,
    )

    r_new = np.interp(
        normalized_arc_new,
        normalized_arc,
        r,
    )

    return x_new, r_new


def build_axisymmetric_mesh(
    x_body: np.ndarray,
    r_body: np.ndarray,
    geometry: Geometry,
    settings: CFDSettings,
):
    """
    Construct the meridional section of the computational fluid domain.

    Boundary traversal order:
        upstream symmetry axis
        -> nose and body surface
        -> rear base
        -> downstream symmetry axis
        -> outlet
        -> far-field boundary
        -> inlet
    """
    try:
        import gmsh
        from dolfinx.io import gmsh as gmshio
        from mpi4py import MPI
    except ImportError as error:
        raise ImportError(
            "Failed to import Gmsh, DOLFINx, or MPI. "
            "Please run this program in a FEniCSx environment."
        ) from error

    geometry.validate()
    settings.validate()

    communicator = MPI.COMM_WORLD
    model_rank = 0

    body_length = geometry.body_length

    x_inlet = -settings.upstream_body_lengths * body_length

    x_outlet = body_length + settings.downstream_body_lengths * body_length

    r_farfield = settings.farfield_body_radii * geometry.radius

    mesh_size_wall = settings.mesh_size_wall_ratio * geometry.radius

    mesh_size_far = settings.mesh_size_far_ratio * geometry.radius

    x_wall, r_wall = resample_profile_by_arclength(
        x_body,
        r_body,
        settings.wall_samples,
    )

    gmsh.initialize()

    try:
        gmsh.option.setNumber(
            "General.Terminal",
            int(settings.gmsh_terminal),
        )

        model_name = f"body_{uuid.uuid4().hex}"

        gmsh.model.add(model_name)

        if communicator.rank == model_rank:
            geometry_model = gmsh.model.geo

            point_inlet_axis = geometry_model.addPoint(
                x_inlet,
                0.0,
                0.0,
                mesh_size_far,
            )

            wall_points: list[int] = []

            for x_value, r_value in zip(
                x_wall,
                r_wall,
                strict=True,
            ):
                wall_points.append(
                    geometry_model.addPoint(
                        float(x_value),
                        float(r_value),
                        0.0,
                        mesh_size_wall,
                    )
                )

            point_tip = wall_points[0]
            point_tail_top = wall_points[-1]

            point_tail_axis = geometry_model.addPoint(
                body_length,
                0.0,
                0.0,
                mesh_size_wall,
            )

            point_outlet_axis = geometry_model.addPoint(
                x_outlet,
                0.0,
                0.0,
                mesh_size_far,
            )

            point_outlet_top = geometry_model.addPoint(
                x_outlet,
                r_farfield,
                0.0,
                mesh_size_far,
            )

            point_inlet_top = geometry_model.addPoint(
                x_inlet,
                r_farfield,
                0.0,
                mesh_size_far,
            )

            axis_upstream = geometry_model.addLine(
                point_inlet_axis,
                point_tip,
            )

            body_side = geometry_model.addSpline(
                wall_points,
            )

            body_base = geometry_model.addLine(
                point_tail_top,
                point_tail_axis,
            )

            axis_downstream = geometry_model.addLine(
                point_tail_axis,
                point_outlet_axis,
            )

            outlet = geometry_model.addLine(
                point_outlet_axis,
                point_outlet_top,
            )

            farfield = geometry_model.addLine(
                point_outlet_top,
                point_inlet_top,
            )

            inlet = geometry_model.addLine(
                point_inlet_top,
                point_inlet_axis,
            )

            curve_loop = geometry_model.addCurveLoop(
                [
                    axis_upstream,
                    body_side,
                    body_base,
                    axis_downstream,
                    outlet,
                    farfield,
                    inlet,
                ]
            )

            fluid_surface = geometry_model.addPlaneSurface([curve_loop])

            geometry_model.synchronize()

            gmsh.model.addPhysicalGroup(
                2,
                [fluid_surface],
                FLUID,
            )
            gmsh.model.setPhysicalName(
                2,
                FLUID,
                "Fluid",
            )

            gmsh.model.addPhysicalGroup(
                1,
                [inlet],
                INLET,
            )
            gmsh.model.setPhysicalName(
                1,
                INLET,
                "Inlet",
            )

            gmsh.model.addPhysicalGroup(
                1,
                [outlet],
                OUTLET,
            )
            gmsh.model.setPhysicalName(
                1,
                OUTLET,
                "Outlet",
            )

            gmsh.model.addPhysicalGroup(
                1,
                [farfield],
                FARFIELD,
            )
            gmsh.model.setPhysicalName(
                1,
                FARFIELD,
                "Farfield",
            )

            gmsh.model.addPhysicalGroup(
                1,
                [
                    axis_upstream,
                    axis_downstream,
                ],
                AXIS,
            )
            gmsh.model.setPhysicalName(
                1,
                AXIS,
                "Axis",
            )

            gmsh.model.addPhysicalGroup(
                1,
                [
                    body_side,
                    body_base,
                ],
                BODY,
            )
            gmsh.model.setPhysicalName(
                1,
                BODY,
                "Body",
            )

            distance_field = gmsh.model.mesh.field.add("Distance")

            gmsh.model.mesh.field.setNumbers(
                distance_field,
                "EdgesList",
                [
                    body_side,
                    body_base,
                ],
            )

            gmsh.model.mesh.field.setNumber(
                distance_field,
                "Sampling",
                120,
            )

            threshold_field = gmsh.model.mesh.field.add("Threshold")

            gmsh.model.mesh.field.setNumber(
                threshold_field,
                "IField",
                distance_field,
            )

            gmsh.model.mesh.field.setNumber(
                threshold_field,
                "LcMin",
                mesh_size_wall,
            )

            gmsh.model.mesh.field.setNumber(
                threshold_field,
                "LcMax",
                mesh_size_far,
            )

            gmsh.model.mesh.field.setNumber(
                threshold_field,
                "DistMin",
                2.0 * mesh_size_wall,
            )

            gmsh.model.mesh.field.setNumber(
                threshold_field,
                "DistMax",
                12.0 * mesh_size_wall,
            )

            gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

            gmsh.model.mesh.generate(2)

        mesh_data = gmshio.model_to_mesh(
            gmsh.model,
            communicator,
            model_rank,
            gdim=2,
        )

    finally:
        gmsh.finalize()

    domain = mesh_data.mesh
    facet_tags = mesh_data.facet_tags

    if facet_tags is None:
        raise RuntimeError("Gmsh did not return the boundary markers.")

    facet_tags.name = "Facet markers"

    return domain, facet_tags


# =============================================================================
# Navier-Stokes solver and drag calculation
# =============================================================================


def axisymmetric_divergence(
    velocity,
    radial_coordinate,
):
    """
    Axisymmetric divergence:

        div_axi(u) =
            du_x/dx
            + du_r/dr
            + u_r/r
    """
    return velocity[0].dx(0) + velocity[1].dx(1) + velocity[1] / radial_coordinate


def axisymmetric_strain(
    velocity,
    radial_coordinate,
):
    """
    Axisymmetric 3×3 strain-rate tensor in cylindrical coordinates:

        (x, r, θ)
    """
    import ufl

    epsilon_xx = velocity[0].dx(0)
    epsilon_rr = velocity[1].dx(1)
    epsilon_tt = velocity[1] / radial_coordinate

    epsilon_xr = 0.5 * (velocity[0].dx(1) + velocity[1].dx(0))

    return ufl.as_tensor(
        (
            (
                epsilon_xx,
                epsilon_xr,
                0.0,
            ),
            (
                epsilon_xr,
                epsilon_rr,
                0.0,
            ),
            (
                0.0,
                0.0,
                epsilon_tt,
            ),
        )
    )


def create_uniform_velocity_function(
    velocity_space,
    velocity_value: float,
):
    """Create a uniform vector-valued function (U_inf, 0)."""
    from dolfinx import fem
    from petsc4py import PETSc

    velocity_function = fem.Function(velocity_space)

    def expression(coordinates):
        values = np.zeros(
            (
                2,
                coordinates.shape[1],
            ),
            dtype=PETSc.ScalarType,
        )

        values[0] = PETSc.ScalarType(velocity_value)

        return values

    velocity_function.interpolate(expression)

    return velocity_function


def axisymmetric_navier_stokes_drag(
    coefficients: np.ndarray,
    geometry: Geometry,
    flow: Flow,
    settings: CFDSettings | None = None,
) -> DragResult:
    """
    Solve the steady Navier-Stokes equations and integrate the drag force.

    The computed pressure is gauge pressure, with p = 0 prescribed at the outlet.

    Force acting on the body:

        F_body =
                    - integral_Gamma sigma n_fluid dS

    In the axisymmetric formulation:

        dS = 2 pi r ds_meridional
    """
    try:
        import ufl
        from basix.ufl import element, mixed_element
        from dolfinx import fem
        from dolfinx.fem.petsc import NonlinearProblem
        from mpi4py import MPI
        from petsc4py import PETSc
    except ImportError as error:
        raise ImportError(
            "FEniCSx dependencies were not found. "
            "Please install DOLFINx, UFL, Basix, PETSc, mpi4py, and Gmsh."
        ) from error

    if settings is None:
        settings = CFDSettings()

    geometry.validate()
    flow.validate()
    settings.validate()

    coefficients = np.asarray(
        coefficients,
        dtype=float,
    )

    (
        x_body,
        r_body,
        _,
        r_nose,
        drdx_nose,
    ) = body_profile(
        coefficients,
        geometry,
    )

    penalty = geometry_penalty(
        r_nose,
        drdx_nose,
        geometry,
    )

    if penalty > 1.0e-8:
        raise ValueError(
            "Geometry violates geometry constraints. " f"Penalidade = {penalty:.6e}."
        )

    domain, facet_tags = build_axisymmetric_mesh(
        x_body,
        r_body,
        geometry,
        settings,
    )

    velocity_element = element(
        "Lagrange",
        domain.basix_cell(),
        2,
        shape=(2,),
    )

    pressure_element = element(
        "Lagrange",
        domain.basix_cell(),
        1,
    )

    mixed_space = fem.functionspace(
        domain,
        mixed_element(
            [
                velocity_element,
                pressure_element,
            ]
        ),
    )

    solution = fem.Function(
        mixed_space,
        name="velocity_pressure",
    )

    velocity, pressure = ufl.split(solution)

    test_velocity, test_pressure = ufl.TestFunctions(mixed_space)

    solution.sub(0).interpolate(
        lambda coordinates: np.vstack(
            (
                np.full(
                    coordinates.shape[1],
                    flow.velocity,
                    dtype=PETSc.ScalarType,
                ),
                np.zeros(
                    coordinates.shape[1],
                    dtype=PETSc.ScalarType,
                ),
            )
        )
    )

    facet_dimension = domain.topology.dim - 1

    velocity_space, _ = mixed_space.sub(0).collapse()

    radial_velocity_space, _ = mixed_space.sub(0).sub(1).collapse()

    pressure_space, _ = mixed_space.sub(1).collapse()

    velocity_infinity = create_uniform_velocity_function(
        velocity_space,
        flow.velocity,
    )

    velocity_zero = fem.Function(velocity_space)

    radial_velocity_zero = fem.Function(radial_velocity_space)

    pressure_zero = fem.Function(pressure_space)

    inlet_velocity_dofs = fem.locate_dofs_topological(
        (
            mixed_space.sub(0),
            velocity_space,
        ),
        facet_dimension,
        facet_tags.find(INLET),
    )

    farfield_velocity_dofs = fem.locate_dofs_topological(
        (
            mixed_space.sub(0),
            velocity_space,
        ),
        facet_dimension,
        facet_tags.find(FARFIELD),
    )

    body_velocity_dofs = fem.locate_dofs_topological(
        (
            mixed_space.sub(0),
            velocity_space,
        ),
        facet_dimension,
        facet_tags.find(BODY),
    )

    axis_radial_dofs = fem.locate_dofs_topological(
        (
            mixed_space.sub(0).sub(1),
            radial_velocity_space,
        ),
        facet_dimension,
        facet_tags.find(AXIS),
    )

    outlet_pressure_dofs = fem.locate_dofs_topological(
        (
            mixed_space.sub(1),
            pressure_space,
        ),
        facet_dimension,
        facet_tags.find(OUTLET),
    )

    boundary_conditions = [
        fem.dirichletbc(
            velocity_infinity,
            inlet_velocity_dofs,
            mixed_space.sub(0),
        ),
        fem.dirichletbc(
            velocity_infinity,
            farfield_velocity_dofs,
            mixed_space.sub(0),
        ),
        fem.dirichletbc(
            velocity_zero,
            body_velocity_dofs,
            mixed_space.sub(0),
        ),
        fem.dirichletbc(
            radial_velocity_zero,
            axis_radial_dofs,
            mixed_space.sub(0).sub(1),
        ),
        fem.dirichletbc(
            pressure_zero,
            outlet_pressure_dofs,
            mixed_space.sub(1),
        ),
    ]

    spatial_coordinate = ufl.SpatialCoordinate(domain)

    radial_coordinate = spatial_coordinate[1]

    density = fem.Constant(
        domain,
        PETSc.ScalarType(flow.density),
    )

    viscosity = fem.Constant(
        domain,
        PETSc.ScalarType(flow.dynamic_viscosity),
    )

    strain_velocity = axisymmetric_strain(
        velocity,
        radial_coordinate,
    )

    strain_test_velocity = axisymmetric_strain(
        test_velocity,
        radial_coordinate,
    )

    divergence_velocity = axisymmetric_divergence(
        velocity,
        radial_coordinate,
    )

    divergence_test_velocity = axisymmetric_divergence(
        test_velocity,
        radial_coordinate,
    )

    convective_acceleration = ufl.dot(
        ufl.grad(velocity),
        velocity,
    )

    convection_factor = fem.Constant(
        domain,
        PETSc.ScalarType(0.0),
    )

    residual = (
        convection_factor
        * density
        * ufl.inner(
            convective_acceleration,
            test_velocity,
        )
        * radial_coordinate
        * ufl.dx
        + 2.0
        * viscosity
        * ufl.inner(
            strain_velocity,
            strain_test_velocity,
        )
        * radial_coordinate
        * ufl.dx
        - pressure * divergence_test_velocity * radial_coordinate * ufl.dx
        + test_pressure * divergence_velocity * radial_coordinate * ufl.dx
    )

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": ("bt" if settings.use_backtracking else "basic"),
        "snes_rtol": (settings.snes_relative_tolerance),
        "snes_atol": (settings.snes_absolute_tolerance),
        "snes_max_it": (settings.snes_max_iterations),
        "snes_error_if_not_converged": False,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
    }

    if settings.monitor_solver:
        petsc_options.update(
            {
                "snes_monitor": None,
                "snes_converged_reason": None,
                "snes_linesearch_monitor": None,
                "ksp_converged_reason": None,
            }
        )

    nonlinear_problem = NonlinearProblem(
        residual,
        solution,
        bcs=boundary_conditions,
        petsc_options_prefix="body_axisymmetric_ns_",
        petsc_options=petsc_options,
    )

    total_nonlinear_iterations = 0

    for factor in settings.continuation_factors:
        convection_factor.value = PETSc.ScalarType(factor)

        if settings.monitor_solver:
            print(f"\n[continuation] convective factor = {factor:.3f}")

        nonlinear_problem.solve()

        convergence_reason = nonlinear_problem.solver.getConvergedReason()

        stage_iterations = nonlinear_problem.solver.getIterationNumber()

        total_nonlinear_iterations += int(stage_iterations)

        if settings.monitor_solver:
            print(
                "[continuation] SNES reason:",
                convergence_reason,
                "| iterations:",
                stage_iterations,
            )

        if convergence_reason <= 0:
            raise RuntimeError(
                "The nonlinear solver failed to converge during continuation. "
                f"Convective factor = {factor:.3f}; "
                f"PETSc SNES convergence reason = {convergence_reason}. "
                "See the snes_monitor output for the convergence history."
            )

        solution.x.scatter_forward()

    nonlinear_iterations = total_nonlinear_iterations

    velocity_solution, pressure_solution = ufl.split(solution)

    fluid_normal = ufl.FacetNormal(domain)

    axial_direction = ufl.as_vector(
        (
            1.0,
            0.0,
        )
    )

    meridional_strain = ufl.sym(ufl.grad(velocity_solution))

    viscous_stress = 2.0 * viscosity * meridional_strain

    pressure_stress = -pressure_solution * ufl.Identity(2)

    boundary_measure = ufl.Measure(
        "ds",
        domain=domain,
        subdomain_data=facet_tags,
    )

    pressure_force_form = fem.form(
        -2.0
        * np.pi
        * ufl.dot(
            ufl.dot(
                pressure_stress,
                fluid_normal,
            ),
            axial_direction,
        )
        * radial_coordinate
        * boundary_measure(BODY)
    )

    viscous_force_form = fem.form(
        -2.0
        * np.pi
        * ufl.dot(
            ufl.dot(
                viscous_stress,
                fluid_normal,
            ),
            axial_direction,
        )
        * radial_coordinate
        * boundary_measure(BODY)
    )

    pressure_force_local = fem.assemble_scalar(pressure_force_form)

    viscous_force_local = fem.assemble_scalar(viscous_force_form)

    pressure_force = domain.comm.allreduce(
        pressure_force_local,
        op=MPI.SUM,
    )

    viscous_force = domain.comm.allreduce(
        viscous_force_local,
        op=MPI.SUM,
    )

    total_force = pressure_force + viscous_force

    normalization_force = flow.dynamic_pressure * geometry.reference_area

    cd_pressure = pressure_force / normalization_force

    cd_viscous = viscous_force / normalization_force

    cd_total = cd_pressure + cd_viscous

    reynolds = (
        flow.density * flow.velocity * geometry.body_length / flow.dynamic_viscosity
    )

    return DragResult(
        cd_total=float(cd_total),
        cd_pressure=float(cd_pressure),
        cd_viscous=float(cd_viscous),
        force_total=float(total_force),
        force_pressure=float(pressure_force),
        force_viscous=float(viscous_force),
        reynolds=float(reynolds),
        nonlinear_iterations=int(nonlinear_iterations),
    )


# =============================================================================
# Object function and optimization
# =============================================================================


def navier_stokes_objective(
    coefficients: np.ndarray,
    geometry: Geometry,
    flow: Flow,
    settings: CFDSettings,
    monitor: OptimizationMonitor,
    penalty_weight: float = 1.0e4,
    invalid_geometry_cost: float = 1.0e3,
    cfd_failure_cost: float = 2.0e3,
    regularization: float = 1.0e-4,
    verbose: bool = False,
) -> float:
    """
    Objective function for Differential Evolution.

    Returns

        C_D
        + geometric penalty
        + coefficient regularization.

    Invalid geometries are rejected before mesh generation.

    CFD failures are assigned a large penalty cost instead of
    terminating the optimization.
    """
    coefficients = np.asarray(
        coefficients,
        dtype=float,
    )

    penalty = evaluate_geometry_penalty(
        coefficients,
        geometry,
        weight=penalty_weight,
    )

    if penalty > 1.0e-8:
        objective_value = invalid_geometry_cost + penalty

        if verbose:
            print(
                "Invalid geometry:",
                coefficients,
                "objective =",
                objective_value,
            )

        return float(objective_value)

    monitor.plot_geometry(
        coefficients,
        geometry,
    )

    try:
        drag_result = axisymmetric_navier_stokes_drag(
            coefficients,
            geometry,
            flow,
            settings,
        )

        monitor.drag_history.append(drag_result.cd_total)
        monitor.plot_drag_history()

    except Exception as error:
        objective_value = cfd_failure_cost + regularization * float(
            coefficients @ coefficients
        )

        if verbose:
            print(
                "CFD failed for",
                coefficients,
                ":",
                repr(error),
            )

        return float(objective_value)

    objective_value = (
        drag_result.cd_total
        + penalty
        + regularization * float(coefficients @ coefficients)
    )

    if verbose:
        print(
            "Coefficients:",
            coefficients,
            "CD:",
            drag_result.cd_total,
            "objectives:",
            objective_value,
        )

    return float(objective_value)


def optimize_shape(
    geometry: Geometry,
    flow: Flow,
    settings: CFDSettings,
    monitor: OptimizationMonitor,
    bounds: tuple[
        tuple[float, float],
        ...,
    ] = (
        (-0.50, 0.50),
        (-0.50, 0.50),
    ),
    seed: int = 7,
    max_iterations: int = 10,
    population_size: int = 5,
    tolerance: float = 1.0e-4,
    polish: bool = False,
    verbose: bool = True,
):
    """
    Global optimization using Differential Evolution.

    Warning
    -------
    Each function evaluation solves a complete Navier–Stokes problem.
    """
    geometry.validate()
    flow.validate()
    settings.validate()

    result = differential_evolution(
        func=lambda coefficients: (
            navier_stokes_objective(
                coefficients=coefficients,
                geometry=geometry,
                flow=flow,
                settings=settings,
                monitor=monitor,
                verbose=verbose,
            )
        ),
        bounds=bounds,
        strategy="best1bin",
        maxiter=max_iterations,
        popsize=population_size,
        tol=tolerance,
        polish=polish,
        seed=seed,
        updating="immediate",
        workers=1,
    )

    return result


# =============================================================================
# Execution
# =============================================================================


def create_default_problem() -> tuple[
    Geometry,
    Flow,
    CFDSettings,
]:
    """Creates a conservative initial configuration to test the solver."""
    geometry = Geometry(
        radius=0.050,
        nose_length=0.200,
        cylinder_length=0.300,
        nose_points=501,
        cylinder_points=301,
    )

    flow = Flow(
        density=1.225,
        dynamic_viscosity=1.81e-5,
        velocity=0.10,
        pressure=101325.0,
    )

    settings = CFDSettings(
        upstream_body_lengths=3.0,
        downstream_body_lengths=8.0,
        farfield_body_radii=10.0,
        wall_samples=120,
        mesh_size_wall_ratio=0.12,
        mesh_size_far_ratio=0.60,
        snes_relative_tolerance=1.0e-7,
        snes_absolute_tolerance=1.0e-8,
        snes_max_iterations=60,
        use_backtracking=True,
        monitor_solver=True,
        gmsh_terminal=False,
    )

    return geometry, flow, settings


def run_profile_only(
    output_directory: Path,
) -> None:
    """Generates only the base geometry, without importing FEniCSx."""
    geometry, flow, settings = create_default_problem()

    coefficients = np.array(
        [
            0.0,
            0.0,
        ]
    )

    save_results(
        coefficients=coefficients,
        geometry=geometry,
        flow=flow,
        settings=settings,
        output_directory=output_directory,
        run_cfd=False,
    )

    print(
        "Base-profile saved in:",
        output_directory.resolve(),
    )


def run_single_cfd_evaluation(
    output_directory: Path,
) -> None:
    """Runs a single CFD evaluation."""
    geometry, flow, settings = create_default_problem()

    coefficients = np.array(
        [
            0.0,
            0.0,
        ]
    )

    drag = save_results(
        coefficients=coefficients,
        geometry=geometry,
        flow=flow,
        settings=settings,
        output_directory=output_directory,
        run_cfd=True,
    )

    print("Coefficients:", coefficients)
    print("Reynolds:", drag.reynolds)
    print("Pressure CD:", drag.cd_pressure)
    print("Viscous CD:", drag.cd_viscous)
    print("Total CD:", drag.cd_total)
    print(
        "Iterações não lineares:",
        drag.nonlinear_iterations,
    )


def run_optimization(
    output_directory: Path,
) -> None:
    """Execute full optimization."""
    geometry, flow, settings = create_default_problem()

    monitor = OptimizationMonitor(
        output_directory,
    )

    result = optimize_shape(
        geometry=geometry,
        flow=flow,
        settings=settings,
        monitor=monitor,
        bounds=(
            (-0.50, 0.50),
            (-0.50, 0.50),
        ),
        seed=7,
        max_iterations=10,
        population_size=5,
        tolerance=1.0e-4,
        polish=False,
        verbose=True,
    )

    print("Optimal coefficients:", result.x)
    print("Object functions:", result.fun)
    print("Convergence:", result.success)
    print("Message:", result.message)

    drag = save_results(
        coefficients=result.x,
        geometry=geometry,
        flow=flow,
        settings=settings,
        output_directory=output_directory,
        optimization_result=result,
        run_cfd=True,
    )

    print("Total optimal CD:", drag.cd_total)
    print(
        "Results saved in:",
        output_directory.resolve(),
    )


def parse_arguments() -> argparse.Namespace:
    """Read command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Navier-Stokes-Based Shape Optimization of the Nose of Axisymmetric Bodies."
        )
    )

    parser.add_argument(
        "--mode",
        choices=(
            "profile",
            "evaluate",
            "optimize",
        ),
        default="profile",
        help=(
            "profile: generates only the geometry; "
            "evaluate: performs a single CFD evaluation; "
            "optimize: performs the optimization."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory.",
    )

    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    arguments = parse_arguments()

    if arguments.mode == "profile":
        run_profile_only(arguments.output)

    elif arguments.mode == "evaluate":
        run_single_cfd_evaluation(arguments.output)

    elif arguments.mode == "optimize":
        run_optimization(arguments.output)

    else:
        raise RuntimeError("Unknown execution mode.")


if __name__ == "__main__":
    main()
