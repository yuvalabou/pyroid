"""Tests for Triply Periodic Minimal Surface (TPMS) mathematical kernels,
models, and integration in PyRoid.
"""

import os
import tempfile

import numpy as np
import pytest
import trimesh

from pyroid.core import GyroidGenerator
from pyroid.models import GyroidParams, STLConversionParams
from pyroid.tpms import (
    TPMSType,
    get_all_tpms_kernels,
)


def test_tpms_type_parsing():
    """Verify tolerant string parsing and aliases for TPMS types."""
    assert TPMSType.from_str("gyroid") == TPMSType.GYROID
    assert TPMSType.from_str("Diamond") == TPMSType.DIAMOND
    assert TPMSType.from_str("d") == TPMSType.DIAMOND
    assert TPMSType.from_str("schwarz-d") == TPMSType.DIAMOND
    assert TPMSType.from_str("primitive") == TPMSType.PRIMITIVE
    assert TPMSType.from_str("P") == TPMSType.PRIMITIVE
    assert TPMSType.from_str("i_wp") == TPMSType.I_WP
    assert TPMSType.from_str("iwp") == TPMSType.I_WP
    assert TPMSType.from_str("fischer_koch_s") == TPMSType.FISCHER_KOCH_S
    assert TPMSType.from_str("fischer-koch") == TPMSType.FISCHER_KOCH_S
    assert TPMSType.from_str("neovius") == TPMSType.NEOVIUS
    assert TPMSType.from_str("c_p") == TPMSType.NEOVIUS

    with pytest.raises(ValueError):
        TPMSType.from_str("non_existent_tpms")


def test_tpms_kernels_registry():
    """Verify all kernels are registered and define required attributes."""
    kernels = get_all_tpms_kernels()
    assert len(kernels) == 6

    x = np.linspace(0, 2 * np.pi, 10)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    for k in kernels:
        assert k.thickness_multiplier > 0
        assert k.display_name
        assert k.description
        val = k.eval(X, Y, Z)
        assert val.shape == (10, 10, 10)
        assert not np.isnan(val).any()
        assert not np.isinf(val).any()

        radial_val = k.eval_radial(X, Y, Z)
        assert radial_val.shape == (10, 10, 10)


@pytest.mark.parametrize(
    "tpms_name",
    ["gyroid", "diamond", "primitive", "i_wp", "fischer_koch_s", "neovius"],
)
def test_parametric_tpms_generation(tpms_name):
    """Verify radial generation produces non-empty PolyData meshes for each TPMS."""
    params = GyroidParams(tpms_type=tpms_name, res=30)
    gen = GyroidGenerator(params)
    mesh = gen.generate()
    assert mesh is not None
    assert mesh.n_points > 0
    assert mesh.n_cells > 0
    assert gen.cavity_a is not None
    assert gen.cavity_b is not None


@pytest.fixture
def sample_box_stl():
    """Create a temporary closed box STL file for testing infill."""
    mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        path = f.name
    mesh.export(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.mark.parametrize(
    "tpms_name",
    ["gyroid", "diamond", "primitive", "i_wp", "fischer_koch_s", "neovius"],
)
def test_stl_conversion_sheet_tpms(sample_box_stl, tpms_name):
    """Verify STL volume conversion into sheet lattices for each TPMS."""
    params = STLConversionParams(
        stl_path=sample_box_stl,
        tpms_type=tpms_name,
        resolution=25,
        cell_size=5.0,
        wall_thickness=0.8,
        is_skeletal=False,
    )
    gen = GyroidGenerator()
    mesh = gen.convert_from_stl(params)
    assert mesh is not None
    assert mesh.n_points > 0
    assert mesh.n_cells > 0


@pytest.mark.parametrize(
    "tpms_name",
    ["diamond", "i_wp", "primitive"],
)
def test_stl_conversion_skeletal_tpms(sample_box_stl, tpms_name):
    """Verify STL volume conversion into skeletal struts for various TPMS."""
    params = STLConversionParams(
        stl_path=sample_box_stl,
        tpms_type=tpms_name,
        resolution=25,
        cell_size=5.0,
        wall_thickness=0.8,
        is_skeletal=True,
    )
    gen = GyroidGenerator()
    mesh = gen.convert_from_stl(params)
    assert mesh is not None
    assert mesh.n_points > 0
    assert mesh.n_cells > 0


def test_infill_estimation_per_tpms():
    """Verify infill estimation scales reasonably with cell size and wall thickness across TPMS types."""
    infill_g = STLConversionParams.estimate_infill(5.0, 0.8, is_skeletal=False, tpms_type="gyroid")
    infill_d = STLConversionParams.estimate_infill(5.0, 0.8, is_skeletal=False, tpms_type="diamond")
    infill_p = STLConversionParams.estimate_infill(
        5.0, 0.8, is_skeletal=False, tpms_type="primitive"
    )
    infill_n = STLConversionParams.estimate_infill(5.0, 0.8, is_skeletal=False, tpms_type="neovius")

    assert 10.0 < infill_p < infill_g < infill_d < infill_n < 95.0

    c_size = STLConversionParams.estimate_cell_size(
        30.0, 0.8, is_skeletal=False, tpms_type="diamond"
    )
    assert 5.0 < c_size < 15.0
