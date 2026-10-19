"""Core mathematical engine and mesh generation for PyRoid.

Contains GyroidGenerator for computing parametric gyroid surfaces in cylindrical
coordinates and coordinating STL mesh infill conversions.
"""
from collections.abc import Callable
from dataclasses import fields
from typing import Any

import numpy as np
import pyvista as pv
import trimesh

from pyroid.mesh_utils import (convert_stl_to_gyroid, load_stl,
                                prepare_for_printing, render_offscreen)
from pyroid.models import GyroidParams, STLConversionParams
from pyroid.tpms import get_tpms_kernel


class GyroidGenerator:
    """Core class for generating parametric gyroid structures and converting STL meshes."""

    DEFAULT_PARAMS = GyroidParams().to_dict()

    def __init__(self, params: GyroidParams | dict[str, Any] | None = None):
        """
        Initialize the gyroid generator with parameters.

        Args:
            params: GyroidParams instance or dictionary of parameter overrides.
        """
        if isinstance(params, GyroidParams):
            self.params = params
        elif isinstance(params, dict):
            self.params = GyroidParams.from_dict(params)
        else:
            self.params = GyroidParams()

        self.mesh: pv.PolyData | None = None
        self.cavity_a: pv.PolyData | None = None
        self.cavity_b: pv.PolyData | None = None

    def validate_params(self) -> tuple[bool, str]:
        """Validate parameters using the GyroidParams data model."""
        return self.params.validate()

    def generate(
        self, progress_callback: Callable[[int, str], None] | None = None
    ) -> pv.PolyData:
        """
        Generate the radially symmetrical gyroid/TPMS mesh using current parameters.

        Returns:
            PyVista PolyData mesh object.

        Raises:
            ValueError: If parameters are invalid.
        """
        is_valid, error = self.validate_params()
        if not is_valid:
            raise ValueError(error)

        p = self.params
        res = int(p.res)
        kernel = get_tpms_kernel(p.tpms_type)

        if progress_callback:
            progress_callback(10, f"Initializing cylindrical grid (res={res})...")

        # Generate cylindrical/radial grid domain
        ky = 2.0 * np.pi / p.b
        r_aux, phi, z = np.mgrid[
            0 : p.a : res * 1j, 0 : p.b : res * 1j, 0 : p.c : res * 1j
        ]

        r = (p.r2 - p.r1) / p.a * r_aux + p.r1

        if progress_callback:
            progress_callback(35, f"Evaluating radial {kernel.display_name} scalar field...")

        # Calculate radial TPMS function values
        scale_x = 2.0 * np.pi * p.cell_radius / p.a
        scale_y = 2.0 * np.pi * p.cell_radius / p.b
        scale_z = 2.0 * np.pi * p.cell_height / p.c

        phi_scaled = phi * p.phi_scale
        X = r_aux * scale_x
        Y = phi_scaled * scale_y
        Z = z * scale_z
        raw_g = kernel.eval_radial(X, Y, Z) + p.bias

        # Apply radial mask
        mask = (r >= (p.r1 - p.wall_thickness)) & (r <= p.r1)
        fun_values = np.where(mask, raw_g, 1.0)

        if progress_callback:
            progress_callback(60, "Transforming coordinates to Cartesian...")

        # Convert coordinates to Cartesian
        x = r * np.cos(phi * ky)
        y = r * np.sin(phi * ky)

        if progress_callback:
            progress_callback(80, "Extracting isosurface contour...")

        # Build PyVista StructuredGrid and extract isosurface + dual cavities
        grid = pv.StructuredGrid(x, y, z)
        grid["vol"] = fun_values.ravel("F")
        self.mesh = grid.contour([0])
        t_sheet = 0.2 * (kernel.thickness_multiplier / 4.80)
        grid["vol"] = np.where(mask, t_sheet - raw_g, 1.0).ravel("F")
        self.cavity_a = grid.contour([0])
        grid["vol"] = np.where(mask, raw_g + t_sheet, 1.0).ravel("F")
        self.cavity_b = grid.contour([0])

        if progress_callback:
            progress_callback(95, "Finalizing mesh...")

        return self.mesh

    def convert_from_stl(
        self,
        stl_params: STLConversionParams,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> pv.PolyData:
        """
        Import an STL file and convert its 3D volume into a gyroid lattice.

        Args:
            stl_params: STLConversionParams configuration object.
            progress_callback: Optional callback receiving (percent, status_message).

        Returns:
            PyVista PolyData mesh of the gyroidized STL volume.
        """
        is_valid, err = stl_params.validate()
        if not is_valid:
            raise ValueError(err)

        if progress_callback:
            progress_callback(5, "Loading STL geometry...")

        tri_mesh, _ = load_stl(stl_params.stl_path)
        self.mesh = convert_stl_to_gyroid(
            tri_mesh, stl_params, progress_callback=progress_callback
        )
        self.cavity_a = getattr(self.mesh, "cavity_a", None)
        self.cavity_b = getattr(self.mesh, "cavity_b", None)
        return self.mesh

    def save_stl(
        self, filename: str, prepare_manifold: bool = False, flat_cap: bool = False
    ) -> None:
        """Save the generated mesh as an STL file."""
        if self.mesh is None:
            raise ValueError("No mesh has been generated yet")

        if prepare_manifold:
            tri = prepare_for_printing(self.mesh, flat_cap=flat_cap)
            tri.export(filename)
        else:
            self.mesh.save(filename)

    def save_obj(self, filename: str) -> None:
        """Save the generated mesh as an OBJ file."""
        if self.mesh is None:
            raise ValueError("No mesh has been generated yet")

        tri = trimesh.Trimesh(
            vertices=self.mesh.points, faces=self.mesh.faces.reshape((-1, 4))[:, 1:]
        )
        tri.export(filename)

    def save_screenshot(self, filename: str) -> None:
        """Save a rendered PNG thumbnail of the mesh."""
        if self.mesh is None:
            raise ValueError("No mesh has been generated yet")
        render_offscreen(self.mesh, filename)

    def get_mesh_stats(self) -> dict[str, Any]:
        """Return quantitative mesh metrics."""
        if self.mesh is None:
            raise ValueError("No mesh has been generated yet")

        return {
            "n_points": self.mesh.n_points,
            "n_cells": self.mesh.n_cells,
            "volume": self.mesh.volume,
            "surface_area": self.mesh.area,
            "has_normals": self.mesh.face_normals is not None,
        }

    @classmethod
    def get_param_descriptions(cls) -> dict[str, str]:
        """Return parameter descriptions derived from model metadata."""
        return {f.name: str(f.metadata.get("help", "")) for f in fields(GyroidParams)}

    @classmethod
    def get_presets(cls) -> dict[str, dict[str, Any]]:
        """Return predefined parameter presets."""
        return {
            "default": cls.DEFAULT_PARAMS,
            "fine_detail": {
                "res": 120,
                "a": 24,
                "b": 24,
                "c": 10,
                "r1": 12,
                "r2": 0,
                "phi_scale": 8,
                "wall_thickness": 11.5,
                "cell_radius": 1.5,
                "cell_height": 2.5,
                "bias": 0.0,
            },
            "thick_walls": {
                "res": 80,
                "a": 24,
                "b": 24,
                "c": 10,
                "r1": 12,
                "r2": 0,
                "phi_scale": 8,
                "wall_thickness": 15.0,
                "cell_radius": 2,
                "cell_height": 3,
                "bias": 0.0,
            },
            "dense_pattern": {
                "res": 100,
                "a": 24,
                "b": 24,
                "c": 10,
                "r1": 12,
                "r2": 0,
                "phi_scale": 12,
                "wall_thickness": 11.5,
                "cell_radius": 1.2,
                "cell_height": 1.8,
                "bias": 0.0,
            },
        }
