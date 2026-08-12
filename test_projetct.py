import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import project


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def geometry():
    return project.Geometry(
        radius=0.05,
        nose_length=0.20,
        cylinder_length=0.30,
        nose_points=51,
        cylinder_points=31,
    )


@pytest.fixture
def flow():
    return project.Flow(
        density=1.225,
        dynamic_viscosity=1.81e-5,
        velocity=0.10,
        pressure=101325.0,
    )


@pytest.fixture
def settings():
    return project.CFDSettings(
        wall_samples=20,
        monitor_solver=False,
        gmsh_terminal=False,
    )


@pytest.fixture
def coefficients():
    return np.array([0.0, 0.0])


# =============================================================================
# OptimizationMonitor
# =============================================================================


def test_OptimizationMonitor_init(tmp_path):
    monitor = project.OptimizationMonitor(tmp_path)

    assert monitor.counter == 0
    assert monitor.drag_history == []
    assert monitor.output_directory == tmp_path
    assert tmp_path.exists()


def test_OptimizationMonitor_advance(tmp_path):
    monitor = project.OptimizationMonitor(tmp_path)

    assert monitor.advance() == 1
    assert monitor.counter == 1

    assert monitor.advance() == 2
    assert monitor.counter == 2


def test_OptimizationMonitor_plot_geometry(
    tmp_path,
    geometry,
    coefficients,
):
    monitor = project.OptimizationMonitor(tmp_path)

    monitor.plot_geometry(coefficients, geometry)

    assert monitor.counter == 1
    assert (tmp_path / "profile_00001.png").exists()


def test_OptimizationMonitor_plot_drag_history(tmp_path):
    monitor = project.OptimizationMonitor(tmp_path)

    monitor.drag_history = [0.5, 0.4, 0.35]

    monitor.plot_drag_history()

    assert (tmp_path / "drag_history.png").exists()


# =============================================================================
# save_profile_figure
# =============================================================================


def test_save_profile_figure(
    tmp_path,
    geometry,
    coefficients,
):
    output_path = tmp_path / "profile.png"

    project.save_profile_figure(
        coefficients,
        geometry,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


# =============================================================================
# save_results
# =============================================================================


def test_save_results_without_cfd(
    tmp_path,
    geometry,
    flow,
    settings,
    coefficients,
):
    result = project.save_results(
        coefficients=coefficients,
        geometry=geometry,
        flow=flow,
        settings=settings,
        output_directory=tmp_path,
        run_cfd=False,
    )

    assert result is None

    profile_path = tmp_path / "optimized_profile.csv"
    figure_path = tmp_path / "optimized_profile.png"
    json_path = tmp_path / "results.json"

    assert profile_path.exists()
    assert figure_path.exists()
    assert json_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))

    assert data["coefficients"]["a1"] == 0.0
    assert data["coefficients"]["a2"] == 0.0

    assert data["geometry"]["radius"] == geometry.radius
    assert data["geometry"]["nose_model"] == geometry.nose_model.value

    assert data["drag"] is None


def test_save_results_with_optimization_result(
    tmp_path,
    geometry,
    flow,
    settings,
    coefficients,
):
    optimization_result = SimpleNamespace(
        success=True,
        message="Optimization terminated successfully.",
        fun=0.123,
        nit=5,
        nfev=30,
    )

    project.save_results(
        coefficients=coefficients,
        geometry=geometry,
        flow=flow,
        settings=settings,
        output_directory=tmp_path,
        optimization_result=optimization_result,
        run_cfd=False,
    )

    data = json.loads(
        (tmp_path / "results.json").read_text(
            encoding="utf-8"
        )
    )

    assert data["optimization"]["success"] is True
    assert data["optimization"]["message"] == (
        "Optimization terminated successfully."
    )
    assert data["optimization"]["objective"] == 0.123
    assert data["optimization"]["iterations"] == 5
    assert data["optimization"]["function_evaluations"] == 30


def test_save_results_with_cfd(
    tmp_path,
    geometry,
    flow,
    settings,
    coefficients,
    monkeypatch,
):
    drag = project.DragResult(
        cd_total=0.50,
        cd_pressure=0.40,
        cd_viscous=0.10,
        force_total=1.0,
        force_pressure=0.8,
        force_viscous=0.2,
        reynolds=1000.0,
        nonlinear_iterations=5,
    )

    monkeypatch.setattr(
        project,
        "axisymmetric_navier_stokes_drag",
        lambda *args, **kwargs: drag,
    )

    result = project.save_results(
        coefficients=coefficients,
        geometry=geometry,
        flow=flow,
        settings=settings,
        output_directory=tmp_path,
        run_cfd=True,
    )

    assert result == drag

    data = json.loads(
        (tmp_path / "results.json").read_text(
            encoding="utf-8"
        )
    )

    assert data["drag"]["cd_total"] == 0.50
    assert data["drag"]["cd_pressure"] == 0.40
    assert data["drag"]["cd_viscous"] == 0.10


# =============================================================================
# Geometry
# =============================================================================


def test_Geometry_validate_valid(geometry):
    geometry.validate()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"radius": 0.0},
            "geometry.radius should be positive.",
        ),
        (
            {"radius": -1.0},
            "geometry.radius should be positive.",
        ),
        (
            {"cylinder_length": -1.0},
            "geometry.cylinder_length should be positive.",
        ),
        (
            {"nose_points": 9},
            "geometry.nose_points should be at least 10.",
        ),
        (
            {"cylinder_points": 1},
            "geometry.cylinder_points should be at least 2.",
        ),
    ],
)
def test_Geometry_validate_invalid(kwargs, message):
    geometry = project.Geometry(**kwargs)

    with pytest.raises(ValueError, match=message):
        geometry.validate()


def test_Geometry_body_length(geometry):
    assert geometry.body_length == pytest.approx(0.50)


def test_Geometry_reference_area(geometry):
    expected = np.pi * geometry.radius**2

    assert geometry.reference_area == pytest.approx(expected)


# =============================================================================
# NoseShape
# =============================================================================


def test_NoseShape_members():
    assert project.NoseShape.TANGENT_OGIVE
    assert project.NoseShape.MODIFIED_POWER_SERIES
    assert project.NoseShape.MODEL_2C_NS
    assert project.NoseShape.QBEZIER


# =============================================================================
# model_2c_ns
# =============================================================================


def test_model_2c_ns(geometry):
    x = np.linspace(
        0.0,
        geometry.nose_length,
        20,
    )

    r, drdx = project.model_2c_ns(
        x,
        geometry.radius,
        geometry.nose_length,
        geometry.h,
        geometry.n,
    )

    assert r.shape == x.shape
    assert drdx.shape == x.shape

    assert r[0] == pytest.approx(
        geometry.radius * geometry.h
    )

    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(drdx))


@pytest.mark.parametrize(
    "h",
    [-0.1, 0.0, 1.0, 1.1],
)
def test_model_2c_ns_invalid_h(geometry, h):
    x = np.linspace(0.0, geometry.nose_length, 10)

    with pytest.raises(ValueError):
        project.model_2c_ns(
            x,
            geometry.radius,
            geometry.nose_length,
            h,
            geometry.n,
        )


@pytest.mark.parametrize(
    "n",
    [-1.0, 0.0, 1.1],
)
def test_model_2c_ns_invalid_n(geometry, n):
    x = np.linspace(0.0, geometry.nose_length, 10)

    with pytest.raises(ValueError):
        project.model_2c_ns(
            x,
            geometry.radius,
            geometry.nose_length,
            geometry.h,
            n,
        )


# =============================================================================
# tangent_ogive
# =============================================================================


def test_tangent_ogive(geometry):
    x = np.linspace(
        0.0,
        geometry.nose_length,
        100,
    )

    r, drdx = project.tangent_ogive(
        x,
        geometry.radius,
        geometry.nose_length,
    )

    assert r.shape == x.shape
    assert drdx.shape == x.shape

    assert r[0] == pytest.approx(0.0)
    assert r[-1] == pytest.approx(geometry.radius)

    assert drdx[-1] == pytest.approx(0.0)

    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(drdx))


def test_tangent_ogive_invalid_radius(geometry):
    x = np.linspace(0.0, geometry.nose_length, 10)

    with pytest.raises(ValueError, match="radius"):
        project.tangent_ogive(
            x,
            0.0,
            geometry.nose_length,
        )


def test_tangent_ogive_invalid_length(geometry):
    x = np.linspace(0.0, geometry.radius, 10)

    with pytest.raises(ValueError):
        project.tangent_ogive(
            x,
            geometry.radius,
            geometry.radius,
        )


# =============================================================================
# modified_power_series
# =============================================================================


def test_modified_power_series(geometry):
    exponent = 2.0

    x = np.linspace(
        0.0,
        geometry.nose_length,
        100,
    )

    r, drdx = project.modified_power_series(
        x,
        geometry.radius,
        geometry.nose_length,
        exponent,
    )

    assert r.shape == x.shape
    assert drdx.shape == x.shape

    assert r[0] == pytest.approx(0.0)
    assert r[-1] == pytest.approx(geometry.radius)
    assert drdx[-1] == pytest.approx(0.0)

    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(drdx))


def test_modified_power_series_warning():
    x = np.linspace(0.0, 1.0, 10)

    with pytest.warns(UserWarning):
        project.modified_power_series(
            x,
            radius=1.0,
            nose_length=2.0,
            exponent=1.0,
        )


@pytest.mark.parametrize(
    "radius, nose_length, exponent",
    [
        (0.0, 1.0, 2.0),
        (-1.0, 1.0, 2.0),
        (1.0, 0.0, 2.0),
        (1.0, -1.0, 2.0),
    ],
)
def test_modified_power_series_invalid(
    radius,
    nose_length,
    exponent,
):
    x = np.linspace(0.0, 1.0, 10)

    with pytest.raises(ValueError):
        project.modified_power_series(
            x,
            radius,
            nose_length,
            exponent,
        )


# =============================================================================
# quintic_bezier
# =============================================================================


def test_quintic_bezier(geometry):
    x = np.linspace(
        0.0,
        geometry.nose_length,
        100,
    )

    r, drdx = project.quintic_bezier(
        x,
        geometry.radius,
        geometry.nose_length,
        geometry.control_points,
    )

    assert r.shape == x.shape
    assert drdx.shape == x.shape

    assert r[0] == pytest.approx(0.0)
    assert r[-1] == pytest.approx(geometry.radius)

    assert drdx[-1] == pytest.approx(0.0)

    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(drdx))


def test_quintic_bezier_invalid_control_points(geometry):
    x = np.linspace(0.0, geometry.nose_length, 10)

    with pytest.raises(
        ValueError,
        match="Expected seven design variables",
    ):
        project.quintic_bezier(
            x,
            geometry.radius,
            geometry.nose_length,
            np.zeros(6),
        )


@pytest.mark.parametrize(
    "radius, nose_length",
    [
        (0.0, 0.2),
        (-1.0, 0.2),
        (0.05, 0.0),
        (0.05, -1.0),
    ],
)
def test_quintic_bezier_invalid_geometry(
    geometry,
    radius,
    nose_length,
):
    x = np.linspace(0.0, 0.2, 10)

    with pytest.raises(ValueError):
        project.quintic_bezier(
            x,
            radius,
            nose_length,
            geometry.control_points,
        )


# =============================================================================
# nose_parameterization
# =============================================================================


@pytest.mark.parametrize(
    "nose_model",
    [
        project.NoseShape.TANGENT_OGIVE,
        project.NoseShape.MODIFIED_POWER_SERIES,
        project.NoseShape.MODEL_2C_NS,
        project.NoseShape.QBEZIER,
    ],
)
def test_nose_parameterization(nose_model, geometry):
    if nose_model == project.NoseShape.MODIFIED_POWER_SERIES:
        geometry = project.Geometry(
            radius=geometry.radius,
            nose_length=geometry.nose_length,
            cylinder_length=geometry.cylinder_length,
            nose_points=geometry.nose_points,
            cylinder_points=geometry.cylinder_points,
            nose_model=nose_model,
            exponent=2.0,
        )
    else:
        geometry = project.Geometry(
            radius=geometry.radius,
            nose_length=geometry.nose_length,
            cylinder_length=geometry.cylinder_length,
            nose_points=geometry.nose_points,
            cylinder_points=geometry.cylinder_points,
            nose_model=nose_model,
        )

    x = np.linspace(
        0.0,
        geometry.nose_length,
        20,
    )

    r, drdx = project.nose_parameterization(
        x,
        geometry,
    )

    assert r.shape == x.shape
    assert drdx.shape == x.shape
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(drdx))


def test_nose_parameterization_unsupported(geometry):
    invalid_geometry = SimpleNamespace(
        nose_model="invalid",
    )

    x = np.linspace(0.0, geometry.nose_length, 10)

    with pytest.raises(ValueError, match="Unsupported nose model"):
        project.nose_parameterization(
            x,
            invalid_geometry,
        )


# =============================================================================
# shape_basis
# =============================================================================


def test_shape_basis():
    xi = np.array(
        [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ]
    )

    phi, dphi = project.shape_basis(xi)

    assert phi.shape == (2, 5)
    assert dphi.shape == (2, 5)

    # Both basis functions vanish at the endpoints.
    assert np.allclose(phi[:, [0, -1]], 0.0)

    # Both derivatives vanish at the endpoints.
    assert np.allclose(dphi[:, [0, -1]], 0.0)


def test_shape_basis_known_values():
    xi = np.array([0.5])

    phi, dphi = project.shape_basis(xi)

    assert phi[0, 0] == pytest.approx(0.0625)
    assert phi[1, 0] == pytest.approx(0.0)


# =============================================================================
# geometry_penalty
# =============================================================================


def test_geometry_penalty_valid(geometry):
    r = np.linspace(
        0.0,
        geometry.radius,
        20,
    )

    drdx = np.ones(20)

    penalty = project.geometry_penalty(
        r,
        drdx,
        geometry,
    )

    assert penalty == pytest.approx(0.0)


@pytest.mark.parametrize(
    "r, drdx",
    [
        (
            np.array([-0.01, 0.02]),
            np.array([1.0, 1.0]),
        ),
        (
            np.array([0.02, 0.06]),
            np.array([1.0, 1.0]),
        ),
        (
            np.array([0.02, 0.03]),
            np.array([-1.0, 1.0]),
        ),
    ],
)
def test_geometry_penalty_invalid(
    geometry,
    r,
    drdx,
):
    penalty = project.geometry_penalty(
        r,
        drdx,
        geometry,
    )

    assert penalty > 0.0


def test_geometry_penalty_weight(geometry):
    r = np.array([-0.01, 0.02])
    drdx = np.array([1.0, 1.0])

    p1 = project.geometry_penalty(
        r,
        drdx,
        geometry,
        weight=1.0,
    )

    p2 = project.geometry_penalty(
        r,
        drdx,
        geometry,
        weight=2.0,
    )

    assert p2 == pytest.approx(2.0 * p1)


# =============================================================================
# evaluate_geometry_penalty
# =============================================================================


def test_evaluate_geometry_penalty(
    geometry,
    coefficients,
):
    penalty = project.evaluate_geometry_penalty(
        coefficients,
        geometry,
    )

    assert isinstance(penalty, float)
    assert penalty >= 0.0


def test_evaluate_geometry_penalty_matches_direct_calculation(
    geometry,
    coefficients,
):
    (
        _,
        _,
        _,
        r_nose,
        drdx_nose,
    ) = project.body_profile(
        coefficients,
        geometry,
    )

    expected = project.geometry_penalty(
        r_nose,
        drdx_nose,
        geometry,
    )

    actual = project.evaluate_geometry_penalty(
        coefficients,
        geometry,
    )

    assert actual == pytest.approx(expected)


# =============================================================================
# body_profile
# =============================================================================


def test_body_profile(
    geometry,
    coefficients,
):
    (
        x,
        r,
        drdx,
        r_nose,
        drdx_nose,
    ) = project.body_profile(
        coefficients,
        geometry,
    )

    assert len(x) == geometry.nose_points + geometry.cylinder_points - 1
    assert len(r) == len(x)
    assert len(drdx) == len(x)

    assert len(r_nose) == geometry.nose_points
    assert len(drdx_nose) == geometry.nose_points

    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(geometry.body_length)

    assert r[-1] == pytest.approx(geometry.radius)

    # Cylinder is constant radius.
    assert np.allclose(
        r[geometry.nose_points:],
        geometry.radius,
    )

    # Cylinder derivative is zero.
    assert np.allclose(
        drdx[geometry.nose_points:],
        0.0,
    )


@pytest.mark.parametrize(
    "coefficients",
    [
        np.array([0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([]),
    ],
)
def test_body_profile_invalid_coefficients(
    geometry,
    coefficients,
):
    with pytest.raises(
        ValueError,
        match="exactly two coefficients",
    ):
        project.body_profile(
            coefficients,
            geometry,
        )


# =============================================================================
# Flow
# =============================================================================


def test_Flow_validate_valid(flow):
    flow.validate()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"density": 0.0},
            "flow.density should be positive.",
        ),
        (
            {"density": -1.0},
            "flow.density should be positive.",
        ),
        (
            {"dynamic_viscosity": 0.0},
            "flow.dynamic_viscosity should be positive.",
        ),
        (
            {"velocity": 0.0},
            "flow.velocity should be positive.",
        ),
    ],
)
def test_Flow_validate_invalid(kwargs, message):
    flow = project.Flow(**kwargs)

    with pytest.raises(ValueError, match=message):
        flow.validate()


def test_Flow_dynamic_pressure(flow):
    expected = (
        0.5
        * flow.density
        * flow.velocity**2
    )

    assert flow.dynamic_pressure == pytest.approx(expected)


# =============================================================================
# CFDSettings
# =============================================================================


def test_CFDSettings_validate_valid(settings):
    settings.validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"upstream_body_lengths": 0.0},
        {"downstream_body_lengths": 0.0},
        {"farfield_body_radii": 1.0},
        {"wall_samples": 19},
        {"mesh_size_wall_ratio": 0.0},
        {
            "mesh_size_wall_ratio": 0.5,
            "mesh_size_far_ratio": 0.5,
        },
        {"snes_relative_tolerance": 0.0},
        {"snes_absolute_tolerance": 0.0},
        {"snes_max_iterations": 0},
        {"continuation_factors": (0.0,)},
        {"continuation_factors": (0.1, 1.0)},
        {"continuation_factors": (0.0, 0.5)},
        {"continuation_factors": (0.0, 0.5, 0.4, 1.0)},
    ],
)
def test_CFDSettings_validate_invalid(kwargs):
    settings = project.CFDSettings(**kwargs)

    with pytest.raises(ValueError):
        settings.validate()


# =============================================================================
# resample_profile_by_arclength
# =============================================================================


def test_resample_profile_by_arclength():
    x = np.array([0.0, 1.0, 2.0])
    r = np.array([0.0, 1.0, 0.0])

    x_new, r_new = project.resample_profile_by_arclength(
        x,
        r,
        5,
    )

    assert len(x_new) == 5
    assert len(r_new) == 5

    assert x_new[0] == pytest.approx(0.0)
    assert x_new[-1] == pytest.approx(2.0)

    assert r_new[0] == pytest.approx(0.0)
    assert r_new[-1] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "x, r, number_of_points",
    [
        (
            np.array([[0.0, 1.0]]),
            np.array([0.0, 1.0]),
            5,
        ),
        (
            np.array([0.0, 1.0]),
            np.array([[0.0, 1.0]]),
            5,
        ),
        (
            np.array([0.0, 1.0]),
            np.array([0.0]),
            5,
        ),
        (
            np.array([0.0]),
            np.array([0.0]),
            5,
        ),
        (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            1,
        ),
    ],
)
def test_resample_profile_by_arclength_invalid(
    x,
    r,
    number_of_points,
):
    with pytest.raises(ValueError):
        project.resample_profile_by_arclength(
            x,
            r,
            number_of_points,
        )


def test_resample_profile_by_arclength_zero_length():
    x = np.array([1.0, 1.0])
    r = np.array([2.0, 2.0])

    with pytest.raises(ValueError, match="arclength is zero"):
        project.resample_profile_by_arclength(
            x,
            r,
            5,
        )


# =============================================================================
# axisymmetric_divergence
# =============================================================================


def test_axisymmetric_divergence():
    velocity_x = MagicMock()
    velocity_r = MagicMock()

    velocity = [velocity_x, velocity_r]
    radial_coordinate = MagicMock()

    velocity_x.dx.return_value = 2.0
    velocity_r.dx.return_value = 3.0

    result = project.axisymmetric_divergence(
        velocity,
        radial_coordinate,
    )

    assert result == 2.0 + 3.0 + velocity_r / radial_coordinate


# =============================================================================
# axisymmetric_strain
# =============================================================================


def test_axisymmetric_strain_requires_ufl(monkeypatch):
    """
    The real implementation requires UFL.

    This test verifies that the function attempts to use UFL rather than
    testing the full symbolic tensor construction.
    """

    fake_ufl = MagicMock()

    monkeypatch.setitem(
        sys.modules,
        "ufl",
        fake_ufl,
    )

    velocity = [
        MagicMock(),
        MagicMock(),
    ]

    radial_coordinate = MagicMock()

    fake_ufl.as_tensor.return_value = "tensor"

    result = project.axisymmetric_strain(
        velocity,
        radial_coordinate,
    )

    assert result == "tensor"
    fake_ufl.as_tensor.assert_called_once()


# =============================================================================
# create_uniform_velocity_function
# =============================================================================


def test_create_uniform_velocity_function(monkeypatch):
    fake_fem = MagicMock()
    fake_petsc = MagicMock()

    class FakeFunction:
        def __init__(self, space):
            self.space = space
            self.expression = None

        def interpolate(self, expression):
            self.expression = expression

    fake_fem.Function.side_effect = FakeFunction
    fake_petsc.ScalarType = float

    fake_dolfinx = MagicMock()
    fake_dolfinx.fem = fake_fem

    monkeypatch.setitem(
        sys.modules,
        "dolfinx",
        fake_dolfinx,
    )

    fake_petsc4py = MagicMock()
    fake_petsc4py.PETSc = fake_petsc

    monkeypatch.setitem(
        sys.modules,
        "petsc4py",
        fake_petsc4py,
    )

    function = project.create_uniform_velocity_function(
        "velocity_space",
        3.5,
    )

    assert function.space == "velocity_space"

    coordinates = np.zeros((2, 4))
    values = function.expression(coordinates)

    assert values.shape == (2, 4)
    assert np.allclose(values[0], 3.5)
    assert np.allclose(values[1], 0.0)


# =============================================================================
# build_axisymmetric_mesh
# =============================================================================


def test_build_axisymmetric_mesh_missing_dependencies(
    monkeypatch,
    geometry,
    settings,
):
    """
    The function should provide a clear error when FEniCSx/Gmsh is absent.

    We intentionally block the imports used inside the function.
    """

    real_import = __builtins__["__import__"]

    def fake_import(name, *args, **kwargs):
        if name in {
            "gmsh",
            "dolfinx.io",
            "mpi4py",
        }:
            raise ImportError(name)

        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(
        "builtins.__import__",
        fake_import,
    )

    with pytest.raises(
        ImportError,
        match="Failed to import Gmsh",
    ):
        project.build_axisymmetric_mesh(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.1]),
            geometry,
            settings,
        )


# =============================================================================
# axisymmetric_navier_stokes_drag
# =============================================================================


def test_axisymmetric_navier_stokes_drag_missing_dependencies(
    monkeypatch,
    geometry,
    flow,
    settings,
):
    real_import = __builtins__["__import__"]

    def fake_import(name, *args, **kwargs):
        if name in {
            "ufl",
            "basix.ufl",
            "dolfinx",
            "dolfinx.fem.petsc",
            "mpi4py",
            "petsc4py",
        }:
            raise ImportError(name)

        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(
        "builtins.__import__",
        fake_import,
    )

    with pytest.raises(
        ImportError,
        match="FEniCSx dependencies were not found",
    ):
        project.axisymmetric_navier_stokes_drag(
            np.array([0.0, 0.0]),
            geometry,
            flow,
            settings,
        )


# =============================================================================
# navier_stokes_objective
# =============================================================================


def test_navier_stokes_objective_invalid_geometry(
    geometry,
    flow,
    settings,
    monkeypatch,
    tmp_path,
):
    monitor = project.OptimizationMonitor(tmp_path)

    # Force the geometry penalty to be positive.
    monkeypatch.setattr(
        project,
        "evaluate_geometry_penalty",
        lambda *args, **kwargs: 10.0,
    )

    result = project.navier_stokes_objective(
        coefficients=np.array([0.1, 0.2]),
        geometry=geometry,
        flow=flow,
        settings=settings,
        monitor=monitor,
        penalty_weight=100.0,
        invalid_geometry_cost=1000.0,
    )

    assert result == pytest.approx(1010.0)


def test_navier_stokes_objective_cfd_failure(
    geometry,
    flow,
    settings,
    monkeypatch,
    tmp_path,
):
    monitor = project.OptimizationMonitor(tmp_path)

    monkeypatch.setattr(
        project,
        "evaluate_geometry_penalty",
        lambda *args, **kwargs: 0.0,
    )

    def failing_cfd(*args, **kwargs):
        raise RuntimeError("CFD failed")

    monkeypatch.setattr(
        project,
        "axisymmetric_navier_stokes_drag",
        failing_cfd,
    )

    coefficients = np.array([0.1, 0.2])

    result = project.navier_stokes_objective(
        coefficients,
        geometry,
        flow,
        settings,
        monitor,
        cfd_failure_cost=2000.0,
        regularization=1.0e-4,
    )

    expected = (
        2000.0
        + 1.0e-4 * float(coefficients @ coefficients)
    )

    assert result == pytest.approx(expected)


# =============================================================================
# optimize_shape
# =============================================================================


def test_optimize_shape(
    geometry,
    flow,
    settings,
    monkeypatch,
    tmp_path,
):
    monitor = project.OptimizationMonitor(tmp_path)

    expected_result = SimpleNamespace(
        x=np.array([0.1, 0.2]),
        fun=0.3,
    )

    captured = {}

    def fake_differential_evolution(**kwargs):
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr(
        project,
        "differential_evolution",
        fake_differential_evolution,
    )

    result = project.optimize_shape(
        geometry=geometry,
        flow=flow,
        settings=settings,
        monitor=monitor,
        max_iterations=3,
        population_size=2,
        tolerance=1.0e-3,
        polish=False,
        verbose=False,
    )

    assert result is expected_result

    assert captured["maxiter"] == 3
    assert captured["popsize"] == 2
    assert captured["tol"] == 1.0e-3
    assert captured["polish"] is False
    assert captured["seed"] == 7
    assert captured["workers"] == 1


# =============================================================================
# create_default_problem
# =============================================================================


def test_create_default_problem():
    geometry, flow, settings = project.create_default_problem()

    assert isinstance(geometry, project.Geometry)
    assert isinstance(flow, project.Flow)
    assert isinstance(settings, project.CFDSettings)

    assert geometry.radius == pytest.approx(0.05)
    assert geometry.nose_length == pytest.approx(0.20)
    assert geometry.cylinder_length == pytest.approx(0.30)

    assert flow.velocity == pytest.approx(0.10)

    geometry.validate()
    flow.validate()
    settings.validate()


# =============================================================================
# run_profile_only
# =============================================================================


def test_run_profile_only(
    tmp_path,
    monkeypatch,
):
    called = {}

    def fake_save_results(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(
        project,
        "save_results",
        fake_save_results,
    )

    project.run_profile_only(tmp_path)

    assert called["output_directory"] == tmp_path
    assert called["run_cfd"] is False

    np.testing.assert_array_equal(
        called["coefficients"],
        np.array([0.0, 0.0]),
    )


# =============================================================================
# run_single_cfd_evaluation
# =============================================================================


def test_run_single_cfd_evaluation(
    tmp_path,
    monkeypatch,
    capsys,
):
    drag = project.DragResult(
        cd_total=0.5,
        cd_pressure=0.4,
        cd_viscous=0.1,
        force_total=1.0,
        force_pressure=0.8,
        force_viscous=0.2,
        reynolds=1000.0,
        nonlinear_iterations=10,
    )

    monkeypatch.setattr(
        project,
        "save_results",
        lambda **kwargs: drag,
    )

    project.run_single_cfd_evaluation(tmp_path)

    output = capsys.readouterr().out

    assert "Coefficients:" in output
    assert "Reynolds:" in output
    assert "Pressure CD:" in output
    assert "Viscous CD:" in output
    assert "Total CD:" in output


# =============================================================================
# run_optimization
# =============================================================================


def test_run_optimization(
    tmp_path,
    monkeypatch,
    capsys,
):
    optimization_result = SimpleNamespace(
        x=np.array([0.1, 0.2]),
        fun=0.3,
        success=True,
        message="success",
    )

    drag = project.DragResult(
        cd_total=0.3,
        cd_pressure=0.2,
        cd_viscous=0.1,
        force_total=1.0,
        force_pressure=0.7,
        force_viscous=0.3,
        reynolds=1000.0,
        nonlinear_iterations=5,
    )

    monkeypatch.setattr(
        project,
        "optimize_shape",
        lambda **kwargs: optimization_result,
    )

    monkeypatch.setattr(
        project,
        "save_results",
        lambda **kwargs: drag,
    )

    project.run_optimization(tmp_path)

    output = capsys.readouterr().out

    assert "Optimal coefficients:" in output
    assert "Object functions:" in output
    assert "Convergence:" in output
    assert "Total optimal CD:" in output


# =============================================================================
# parse_arguments
# =============================================================================


@pytest.mark.parametrize(
    "args, mode",
    [
        ([], "profile"),
        (["--mode", "profile"], "profile"),
        (["--mode", "evaluate"], "evaluate"),
        (["--mode", "optimize"], "optimize"),
    ],
)
def test_parse_arguments(monkeypatch, args, mode):
    monkeypatch.setattr(
        sys,
        "argv",
        ["project.py"] + args,
    )

    arguments = project.parse_arguments()

    assert arguments.mode == mode


def test_parse_arguments_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "project.py",
            "--mode",
            "profile",
            "--output",
            str(tmp_path),
        ],
    )

    arguments = project.parse_arguments()

    assert arguments.mode == "profile"
    assert arguments.output == tmp_path


def test_parse_arguments_invalid_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "project.py",
            "--mode",
            "invalid",
        ],
    )

    with pytest.raises(SystemExit):
        project.parse_arguments()


# =============================================================================
# main
# =============================================================================


@pytest.mark.parametrize(
    "mode, function_name",
    [
        ("profile", "run_profile_only"),
        ("evaluate", "run_single_cfd_evaluation"),
        ("optimize", "run_optimization"),
    ],
)
def test_main_dispatch(
    monkeypatch,
    tmp_path,
    mode,
    function_name,
):
    monkeypatch.setattr(
        project,
        "parse_arguments",
        lambda: SimpleNamespace(
            mode=mode,
            output=tmp_path,
        ),
    )

    called = {}

    def fake_function(output):
        called["output"] = output

    monkeypatch.setattr(
        project,
        function_name,
        fake_function,
    )

    project.main()

    assert called["output"] == tmp_path
