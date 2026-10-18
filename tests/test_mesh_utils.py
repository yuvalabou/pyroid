"""Unit tests for mesh processing utilities and STL infill conversion."""
import os
import tempfile
import pytest
import trimesh
import pyvista as pv

from pyroid.models import STLConversionParams
from pyroid.mesh_utils import load_stl, convert_stl_to_gyroid, prepare_for_printing


def test_stl_gyroid_conversion():
    # Create a temporary primitive cube STL
    cube = trimesh.creation.box(extents=[10, 10, 10])
    
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        cube.export(tmp.name)
        tmp_path = tmp.name

    try:
        # Load STL
        tri_mesh, pv_mesh = load_stl(tmp_path)
        assert len(tri_mesh.vertices) > 0

        # Convert to Gyroid lattice volume
        params = STLConversionParams(
            stl_path=tmp_path,
            resolution=30,
            cell_size=3.0,
            wall_thickness=0.5
        )

        gyroid_mesh = convert_stl_to_gyroid(tri_mesh, params)
        assert isinstance(gyroid_mesh, pv.PolyData)
        assert gyroid_mesh.n_points > 0

        # Test watertight preparation
        tri_printable = prepare_for_printing(gyroid_mesh, flat_cap=False)
        assert isinstance(tri_printable, trimesh.Trimesh)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_stl_gyroid_conversion_with_preserved_shell():
    # Create a temporary primitive cube STL
    cube = trimesh.creation.box(extents=[12, 12, 12])
    
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        cube.export(tmp.name)
        tmp_path = tmp.name

    try:
        tri_mesh, _ = load_stl(tmp_path)
        params = STLConversionParams(
            stl_path=tmp_path,
            resolution=30,
            cell_size=3.0,
            wall_thickness=0.5,
            preserve_shell=True,
            shell_thickness=1.0
        )

        gyroid_mesh = convert_stl_to_gyroid(tri_mesh, params)
        assert isinstance(gyroid_mesh, pv.PolyData)
        assert gyroid_mesh.n_points > 0

        tri_printable = prepare_for_printing(gyroid_mesh, flat_cap=False)
        assert isinstance(tri_printable, trimesh.Trimesh)
        assert tri_printable.is_watertight

        # Verify that the original outer boundary geometry was exactly preserved
        import numpy as np
        np.testing.assert_allclose(tri_printable.bounds, cube.bounds, atol=1e-3)
        diffs = np.min(np.linalg.norm(tri_printable.vertices[:, None, :] - cube.vertices[None, :, :], axis=2), axis=0)
        assert (diffs < 1e-4).all()

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_stl_gyroid_calibrated_wall_thickness():
    # A larger box to measure volume fraction scaling with thickness
    cube = trimesh.creation.box(extents=[15, 15, 15])
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        cube.export(tmp.name)
        tmp_path = tmp.name

    try:
        tri_mesh, _ = load_stl(tmp_path)
        # Thinner wall (0.4 mm)
        params_thin = STLConversionParams(
            stl_path=tmp_path,
            resolution=40,
            cell_size=5.0,
            wall_thickness=0.4
        )
        mesh_thin = convert_stl_to_gyroid(tri_mesh, params_thin)

        # Thicker wall (1.0 mm)
        params_thick = STLConversionParams(
            stl_path=tmp_path,
            resolution=40,
            cell_size=5.0,
            wall_thickness=1.0
        )
        mesh_thick = convert_stl_to_gyroid(tri_mesh, params_thick)

        assert mesh_thin.n_points > 0
        assert mesh_thick.n_points > 0
        # Thicker wall must result in higher mesh volume than thinner wall
        assert mesh_thick.volume > mesh_thin.volume

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_stl_gyroid_progress_callback():
    cube = trimesh.creation.box(extents=[10, 10, 10])
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        cube.export(tmp.name)
        tmp_path = tmp.name

    try:
        tri_mesh, _ = load_stl(tmp_path)
        params = STLConversionParams(
            stl_path=tmp_path,
            resolution=30,
            cell_size=5.0,
            wall_thickness=0.8
        )
        updates = []
        def callback(pct, msg):
            updates.append((pct, msg))

        convert_stl_to_gyroid(tri_mesh, params, progress_callback=callback)

        assert len(updates) >= 4
        percentages = [u[0] for u in updates]
        assert percentages == sorted(percentages)
        assert all(0 <= p <= 100 for p in percentages)
        assert any("containment" in u[1].lower() for u in updates)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_stl_gyroid_skeletal_conversion():
    cube = trimesh.creation.box(extents=[10, 10, 10])
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        cube.export(tmp.name)
        tmp_path = tmp.name

    try:
        tri_mesh, _ = load_stl(tmp_path)
        params_sheet = STLConversionParams(
            stl_path=tmp_path, resolution=30, cell_size=5.0, wall_thickness=0.8, is_skeletal=False
        )
        params_skeletal = STLConversionParams(
            stl_path=tmp_path, resolution=30, cell_size=5.0, wall_thickness=0.8, is_skeletal=True
        )

        mesh_sheet = convert_stl_to_gyroid(tri_mesh, params_sheet)
        mesh_skeletal = convert_stl_to_gyroid(tri_mesh, params_skeletal)

        assert mesh_skeletal.n_points > 0
        assert mesh_skeletal.n_cells > 0
        # Skeletal gyroid uses single-boundary strut network, generating fewer faces than dual-walled sheet
        assert mesh_skeletal.n_cells < mesh_sheet.n_cells

        # Test with shell preservation as well
        params_skeletal_shell = STLConversionParams(
            stl_path=tmp_path,
            resolution=30,
            cell_size=5.0,
            wall_thickness=0.8,
            preserve_shell=True,
            shell_thickness=0.8,
            is_skeletal=True,
        )
        mesh_shell = convert_stl_to_gyroid(tri_mesh, params_skeletal_shell)
        assert mesh_shell.n_points > 0
        assert mesh_shell.n_cells > 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)



