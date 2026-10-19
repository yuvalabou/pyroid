"""Unit and integration tests for PyRoid CLI."""
import numpy as np
import trimesh

from pyroid.cli import main


def test_cli_parametric_generation(monkeypatch, tmp_path):
    out_path = str(tmp_path / "out.stl")
    monkeypatch.setattr("sys.argv", ["pyroid-cli", "--output", out_path, "--res", "25", "--print-stats"])

    assert main() == 0
    assert (tmp_path / "out.stl").stat().st_size > 0


def test_cli_stl_conversion(monkeypatch, tmp_path):
    cube = trimesh.creation.box(extents=[8, 8, 8])
    in_path = str(tmp_path / "in.stl")
    out_path = str(tmp_path / "out.stl")
    cube.export(in_path)

    monkeypatch.setattr("sys.argv", ["pyroid-cli", "-i", in_path, "-o", out_path, "--res", "25", "--cell-size", "2.5"])
    assert main() == 0
    assert (tmp_path / "out.stl").stat().st_size > 0


def test_cli_stl_conversion_with_shell(monkeypatch, tmp_path):
    cube = trimesh.creation.box(extents=[10, 10, 10])
    in_path = str(tmp_path / "in.stl")
    out_path = str(tmp_path / "out.stl")
    cube.export(in_path)

    test_args = [
        "pyroid-cli", "-i", in_path, "-o", out_path,
        "--res", "25", "--cell-size", "2.5",
        "--preserve-shell", "--shell-thickness", "1.0",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    assert main() == 0

    out_mesh = trimesh.load(out_path, force="mesh")
    assert out_mesh.is_watertight
    np.testing.assert_allclose(out_mesh.bounds, cube.bounds, atol=1e-3)
    diffs = np.min(np.linalg.norm(out_mesh.vertices[:, None, :] - cube.vertices[None, :, :], axis=2), axis=0)
    assert (diffs < 1e-4).all()


def test_cli_stl_conversion_skeletal(monkeypatch, tmp_path):
    cube = trimesh.creation.box(extents=[10, 10, 10])
    in_path = str(tmp_path / "in.stl")
    out_path = str(tmp_path / "out.stl")
    cube.export(in_path)

    test_args = ["pyroid-cli", "-i", in_path, "-o", out_path, "--res", "20", "--cell-size", "5.0", "--skeletal"]
    monkeypatch.setattr("sys.argv", test_args)
    assert main() == 0
    assert (tmp_path / "out.stl").stat().st_size > 0
