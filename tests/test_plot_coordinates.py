#!/usr/bin/env python3
"""Manufactured checks for the Boozer/PEST plotting data path."""

import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from cumes_plot.coordinate_figures import (
    _periodic_indices,
    render_coordinate_figures,
)
from cumes_plot.coordinates import (
    COORDINATE_CONVENTION,
    FAMILY_NAMES,
    FOURIER_CONVENTION,
    MAGIC,
    PERIOD,
    load_boozer,
    make_boozer_grid,
    make_pest_grid,
)
from cumes_plot.equilibrium import field_lines
from cumes_plot.output_paths import figure_path, resolve_output_base
from cumes_plot.render_3d import add_pyvista_field_lines
from plot_equilibrium import FIGURE_PARAMETERS


def _string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<i", len(encoded)) + encoded


def _manufactured_file(path):
    source_ns = 4
    first_surface = 1
    surfaces = source_ns - first_surface
    ntheta, nzeta = 12, 6
    mode_m = np.array([0, 1], dtype="<i4")
    mode_n = np.array([0, 0], dtype="<i4")
    s = np.array([1.0 / 3.0, 2.0 / 3.0, 1.0], dtype="<f8")
    iota = np.array([0.0, 0.45, 0.60], dtype="<f8")
    radius = np.array([0.15, 0.30, 0.45])
    families = {name: np.zeros((surfaces, 2), dtype="<f8")
                for name in FAMILY_NAMES}
    families["rmncc"][:, 0] = 1.5
    families["rmncc"][:, 1] = radius
    families["zmnsc"][:, 1] = radius
    families["numnsc"][:, 1] = 0.08
    theta = PERIOD * np.arange(ntheta) / ntheta
    b = np.empty((surfaces, nzeta, ntheta), dtype="<f8")
    for surface in range(surfaces):
        b[surface] = 1.0 + 0.1 * surface + 0.2 * np.cos(theta)[None, :]
    sqrtg = -1.0 / (b * b)
    b2j00 = -np.ones(surfaces, dtype="<f8")

    header = [
        8, source_ns, ntheta, nzeta, 2, 0, 5, first_surface,
        ntheta, nzeta, 1, 0, 4,
    ]
    payload = bytearray(MAGIC)
    payload += struct.pack("<i", 2)
    payload += _string(COORDINATE_CONVENTION)
    payload += _string(FOURIER_CONVENTION)
    payload += _string("manufactured.json")
    payload += struct.pack("<13i", *header)
    payload += struct.pack("<d", 1.0e-12)
    for array in (s, iota, mode_m, mode_n):
        payload += array.tobytes()
    for name in FAMILY_NAMES:
        payload += families[name].tobytes()
    for array in (b, sqrtg, b2j00):
        payload += array.tobytes()
    path.write_bytes(payload)
    return radius, b


class PlotCoordinateTest(unittest.TestCase):
    def test_field_lines_lift_onto_independent_render_grid(self):
        theta = np.linspace(0.0, PERIOD, 8, endpoint=False)
        zeta = np.linspace(0.0, PERIOD, 4, endpoint=False)
        edge = {
            "bsupu": np.zeros((theta.size, zeta.size)),
            "bsupv": np.ones((theta.size, zeta.size)),
            "r12": np.zeros((theta.size, zeta.size)),
            "z12": np.zeros((theta.size, zeta.size)),
        }
        render_theta = np.linspace(0.0, PERIOD, 16, endpoint=False)
        render_zeta = np.linspace(0.0, PERIOD, 10, endpoint=False)
        geometry = {
            "r": np.repeat(
                (2.0 + 0.2 * np.cos(render_theta))[:, None],
                render_zeta.size, axis=1),
            "z": np.repeat(
                (0.2 * np.sin(render_theta))[:, None],
                render_zeta.size, axis=1),
        }
        line, = field_lines(
            edge, theta, zeta, seeds=(0.0,), nfp=1, geometry=geometry,
            zeta_span=PERIOD, n_steps=64)
        x, y, z = line
        np.testing.assert_allclose(np.hypot(x, y), 2.2, atol=1.0e-12)
        np.testing.assert_allclose(z, 0.0, atol=1.0e-12)

    def test_pyvista_field_lines_use_configured_tubes(self):
        class Centerline:
            def __init__(self):
                self.tube_arguments = None

            def tube(self, **arguments):
                self.tube_arguments = arguments
                return "tube-mesh"

        class Plotter:
            def __init__(self):
                self.mesh = None
                self.arguments = None

            def add_mesh(self, mesh, **arguments):
                self.mesh = mesh
                self.arguments = arguments

        centerline = Centerline()
        plotter = Plotter()
        scene = {
            "figure_parameters": SimpleNamespace(
                pyvista_field_line_color="#101020",
                pyvista_field_line_tube_sides=8),
            "lines": [(np.array([1.0, 2.0]), np.array([0.0, 0.5]),
                       np.array([0.0, 0.1]))],
            "pyvista_field_line_tube_radius": 0.004,
        }
        with patch("cumes_plot.render_3d.pyvista_polyline",
                   return_value=centerline):
            add_pyvista_field_lines(plotter, object(), scene)
        self.assertEqual(centerline.tube_arguments, {
            "radius": 0.004, "n_sides": 8, "capping": False})
        self.assertEqual(plotter.mesh, "tube-mesh")
        self.assertEqual(plotter.arguments["color"], "#101020")
        self.assertTrue(plotter.arguments["smooth_shading"])

    def test_periodic_theta_lines_do_not_duplicate_seam(self):
        indices = _periodic_indices(120, 24)
        np.testing.assert_array_equal(indices, np.arange(0, 120, 5))
        wrapped_gaps = np.diff(np.append(indices, indices[0] + 120))
        np.testing.assert_array_equal(wrapped_gaps, np.full(24, 5))

    def test_directory_only_output_names(self):
        base = resolve_output_base(output_directory="figure_data")
        self.assertEqual(figure_path(base, "perspective"),
                         "figure_data/perspective.png")
        self.assertEqual(figure_path(base, "pest_field_contours"),
                         "figure_data/pest_field_contours.png")
        prefixed = resolve_output_base("plots/equilibrium.png")
        self.assertEqual(figure_path(prefixed, "perspective"),
                         "plots/equilibrium_perspective.png")

    def test_binary_synthesis_and_native_pest_remap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manufactured.bin"
            radius, b = _manufactured_file(path)
            data = load_boozer(path)
            self.assertEqual(data.b.shape, (3, 6, 12))
            np.testing.assert_array_equal(data.b, b)

            boozer = make_boozer_grid(data)
            theta = data.theta_b
            expected_r = 1.5 + radius[:, None] * np.cos(theta)[None, :]
            expected_z = radius[:, None] * np.sin(theta)[None, :]
            np.testing.assert_allclose(boozer.r[:, 0], expected_r,
                                       atol=2.0e-15)
            np.testing.assert_allclose(boozer.z[:, 0], expected_z,
                                       atol=2.0e-15)

            r_native = np.repeat(expected_r[:, None, :], 6, axis=1)
            z_native = np.repeat(expected_z[:, None, :], 6, axis=1)
            lambda_physical = np.zeros_like(r_native)
            lambda_physical[1:] = 0.08 * np.sin(theta)[None, None, :]
            pest = make_pest_grid(data.s, theta, data.zeta, r_native,
                                  z_native, b, lambda_physical)
            # Zero native lambda makes theta and theta_p coincide. The other
            # surfaces exercise the exact native theta-to-PEST remap.
            np.testing.assert_allclose(pest.r[0], r_native[0], atol=2.0e-15)
            np.testing.assert_allclose(pest.b[0], b[0], atol=2.0e-15)
            self.assertTrue(np.all(np.isfinite(pest.r)))
            self.assertTrue(np.all(pest.b > 0.0))
            self.assertGreater(np.max(np.abs(pest.r[2] - boozer.r[2])), 1.0e-3)

    def test_three_figure_families_render(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manufactured.bin"
            _manufactured_file(path)
            outputs = render_coordinate_figures(
                load_boozer(path), str(Path(directory)) + os.sep,
                FIGURE_PARAMETERS)
            self.assertEqual(len(outputs), 3)
            self.assertIn(str(Path(directory) / "field_slices.png"),
                          outputs)
            self.assertNotIn(
                str(Path(directory) / "boozer_field_slices.png"),
                outputs)
            for output in outputs:
                self.assertGreater(Path(output).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
