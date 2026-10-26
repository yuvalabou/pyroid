"""Integration tests for 2D cross-section slicing (BuildStar emulation)."""

import trimesh
from pyroid.mesh_utils import convert_stl_to_gyroid, export_stl_binary, load_stl
from pyroid.models import STLConversionParams


def test_slice_stl_cylinder_preserved_shell(tmp_path):
    """Verify that a hollowed cylinder with gyroid infill produces actual 2D void cavities when sliced."""
    cyl = trimesh.creation.cylinder(radius=10, height=20, sections=40)
    in_stl = str(tmp_path / "in_cyl.stl")
    out_stl = str(tmp_path / "out_cyl.stl")
    cyl.export(in_stl)

    tri_mesh, _ = load_stl(in_stl)
    params = STLConversionParams(
        stl_path=in_stl,
        resolution=35,
        cell_size=5.0,
        wall_thickness=0.8,
        preserve_shell=True,
        shell_thickness=1.5,
    )
    pv_mesh = convert_stl_to_gyroid(tri_mesh, params)
    export_stl_binary(pv_mesh, out_stl)

    loaded = trimesh.load(out_stl, force="mesh")
    assert loaded.is_watertight
    # Volume must be strictly smaller than original solid cylinder
    assert 0 < loaded.volume < cyl.volume

    # Slicing at mid-height must produce a polygon with internal hole cavities
    slice_2d = loaded.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
    assert slice_2d is not None
    planar = slice_2d.to_2D()[0] if hasattr(slice_2d, "to_2D") else slice_2d.to_planar()[0]
    assert len(planar.polygons_full) >= 1
    # Check that holes (channels for laser not to hatch) exist
    total_holes = sum(len(p.interiors) for p in planar.polygons_full)
    assert total_holes > 0, "Slice must have void interiors instead of being a solid hatch"
