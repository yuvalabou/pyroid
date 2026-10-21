"""Unit tests for mesh processing utilities and STL infill conversion."""

import numpy as np
import pyvista as pv
import trimesh

from pyroid.mesh_utils import convert_stl_to_gyroid, load_stl, prepare_for_printing
from pyroid.models import STLConversionParams


def test_stl_gyroid_conversion(tmp_path):
    cube = trimesh.creation.box(extents=[10, 10, 10])
    tmp_path_stl = str(tmp_path / "cube.stl")
    cube.export(tmp_path_stl)

    tri_mesh, _ = load_stl(tmp_path_stl)
    assert len(tri_mesh.vertices) > 0

    params = STLConversionParams(
        stl_path=tmp_path_stl, resolution=30, cell_size=3.0, wall_thickness=0.5
    )
    gyroid_mesh = convert_stl_to_gyroid(tri_mesh, params)
    assert isinstance(gyroid_mesh, pv.PolyData)
    assert gyroid_mesh.n_points > 0

    tri_printable = prepare_for_printing(gyroid_mesh, flat_cap=False)
    assert isinstance(tri_printable, trimesh.Trimesh)


def test_stl_gyroid_conversion_with_preserved_shell(tmp_path):
    cube = trimesh.creation.box(extents=[12, 12, 12])
    tmp_path_stl = str(tmp_path / "cube.stl")
    cube.export(tmp_path_stl)

    tri_mesh, _ = load_stl(tmp_path_stl)
    params = STLConversionParams(
        stl_path=tmp_path_stl,
        resolution=30,
        cell_size=3.0,
        wall_thickness=0.5,
        preserve_shell=True,
        shell_thickness=1.0,
    )

    gyroid_mesh = convert_stl_to_gyroid(tri_mesh, params)
    assert isinstance(gyroid_mesh, pv.PolyData)
    assert gyroid_mesh.n_points > 0

    tri_printable = prepare_for_printing(gyroid_mesh, flat_cap=False)
    assert isinstance(tri_printable, trimesh.Trimesh)
    assert tri_printable.is_watertight

    np.testing.assert_allclose(tri_printable.bounds, cube.bounds, atol=1e-3)
    diffs = np.min(
        np.linalg.norm(tri_printable.vertices[:, None, :] - cube.vertices[None, :, :], axis=2),
        axis=0,
    )
    assert (diffs < 1e-4).all()


def test_stl_gyroid_calibrated_wall_thickness(tmp_path):
    cube = trimesh.creation.box(extents=[15, 15, 15])
    tmp_path_stl = str(tmp_path / "cube.stl")
    cube.export(tmp_path_stl)

    tri_mesh, _ = load_stl(tmp_path_stl)
    params_thin = STLConversionParams(
        stl_path=tmp_path_stl, resolution=40, cell_size=5.0, wall_thickness=0.4
    )
    mesh_thin = convert_stl_to_gyroid(tri_mesh, params_thin)

    params_thick = STLConversionParams(
        stl_path=tmp_path_stl, resolution=40, cell_size=5.0, wall_thickness=1.0
    )
    mesh_thick = convert_stl_to_gyroid(tri_mesh, params_thick)

    assert mesh_thin.n_points > 0
    assert mesh_thick.n_points > 0
    assert mesh_thick.volume > mesh_thin.volume


def test_stl_gyroid_progress_callback(tmp_path):
    cube = trimesh.creation.box(extents=[10, 10, 10])
    tmp_path_stl = str(tmp_path / "cube.stl")
    cube.export(tmp_path_stl)

    tri_mesh, _ = load_stl(tmp_path_stl)
    params = STLConversionParams(
        stl_path=tmp_path_stl, resolution=30, cell_size=5.0, wall_thickness=0.8
    )
    updates = []

    convert_stl_to_gyroid(
        tri_mesh, params, progress_callback=lambda pct, msg: updates.append((pct, msg))
    )

    assert len(updates) >= 4
    percentages = [u[0] for u in updates]
    assert percentages == sorted(percentages)
    assert all(0 <= p <= 100 for p in percentages)
    assert any("containment" in u[1].lower() for u in updates)


def test_stl_gyroid_skeletal_conversion(tmp_path):
    cube = trimesh.creation.box(extents=[10, 10, 10])
    tmp_path_stl = str(tmp_path / "cube.stl")
    cube.export(tmp_path_stl)

    tri_mesh, _ = load_stl(tmp_path_stl)
    params_sheet = STLConversionParams(
        stl_path=tmp_path_stl, resolution=30, cell_size=5.0, wall_thickness=0.8, is_skeletal=False
    )
    params_skeletal = STLConversionParams(
        stl_path=tmp_path_stl, resolution=30, cell_size=5.0, wall_thickness=0.8, is_skeletal=True
    )

    mesh_sheet = convert_stl_to_gyroid(tri_mesh, params_sheet)
    mesh_skeletal = convert_stl_to_gyroid(tri_mesh, params_skeletal)

    assert mesh_skeletal.n_points > 0
    assert mesh_skeletal.n_cells > 0
    assert mesh_skeletal.n_cells < mesh_sheet.n_cells

    params_skeletal_shell = STLConversionParams(
        stl_path=tmp_path_stl,
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


def test_stl_gyroid_asymmetric_cavities(tmp_path):
    cube = trimesh.creation.box(extents=[10, 10, 10])
    tmp_path_stl = str(tmp_path / "cube.stl")
    cube.export(tmp_path_stl)

    tri_mesh, _ = load_stl(tmp_path_stl)
    params_pos = STLConversionParams(
        stl_path=tmp_path_stl, resolution=30, cell_size=5.0, wall_thickness=0.6, bias=0.4
    )
    mesh_pos = convert_stl_to_gyroid(tri_mesh, params_pos)
    assert hasattr(mesh_pos, "cavity_a") and hasattr(mesh_pos, "cavity_b")
    assert mesh_pos.cavity_a.n_cells > mesh_pos.cavity_b.n_cells


def test_stl_gyroid_target_infill_percent(tmp_path):
    cube = trimesh.creation.box(extents=[12, 12, 12])
    tmp_path_stl = str(tmp_path / "cube.stl")
    cube.export(tmp_path_stl)
    tri_mesh, _ = load_stl(tmp_path_stl)

    cell_25 = STLConversionParams.estimate_cell_size(25.0, 0.8)
    cell_60 = STLConversionParams.estimate_cell_size(60.0, 0.8)
    assert cell_25 > cell_60
    assert abs(STLConversionParams.estimate_infill(cell_25, 0.8) - 25.0) < 0.5

    mesh_low = convert_stl_to_gyroid(
        tri_mesh,
        STLConversionParams(
            stl_path=tmp_path_stl,
            resolution=35,
            cell_size=cell_25,
            wall_thickness=0.8,
            infill_percent=25.0,
        ),
    )
    mesh_high = convert_stl_to_gyroid(
        tri_mesh,
        STLConversionParams(
            stl_path=tmp_path_stl,
            resolution=35,
            cell_size=cell_60,
            wall_thickness=0.8,
            infill_percent=60.0,
        ),
    )
    assert mesh_high.volume > mesh_low.volume
