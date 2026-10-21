# AGENTS.md - Development & Agent Architecture Guide

## System Overview
PyRoid is a Python package, CLI tool, and Qt graphical interface for generating, visualizing, and exporting radially symmetrical gyroid structures and converting arbitrary 3D STL geometries into gyroid lattice solids.

## Package Architecture
- `pyroid/tpms.py`: TPMS mathematical kernels (Gyroid, Diamond, Primitive, I-WP, Fischer-Koch S, Neovius), nodal level-set equations, and thickness calibration factors.
- `pyroid/core.py`: Pure mathematical engine and mesh generation (`GyroidGenerator`).
- `pyroid/models.py`: Strongly-typed dataclasses for parameters (`GyroidParams`, `STLConversionParams`).
- `pyroid/mesh_utils.py`: Utilities for mesh processing, watertight repair, STL volumetric containment masking, and offscreen rendering.
- `pyroid/cli.py`: Command Line Interface (`pyroid-cli`).
- `pyroid/gui.py`: Desktop PyQt5 GUI with embedded 3D PyVista renderer (`pyroid`).
- `scripts/release.py`: Automated executable release pipeline (tests, compilation, packaging, SHA-256 checksums, watch mode).
- `tests/`: Automated unit and integration tests powered by `pytest`.

## Technical Standards & Package Management
- Language: Python >= 3.11
- Package Manager: `uv` (fast Python environment & package manager)
- Dependencies: `numpy`, `pyvista`, `trimesh`, `scipy`, `PyQt5`, `pyvistaqt`, `QDarkStyle`
- Build System: PEP 621 via `pyproject.toml` with `uv.lock`
- Code Style: PEP 8, strict type hints, explicit dataclasses.

## Essential Workflows (with `uv`)
- **Initialize & Sync Environment**: `uv venv` && `uv sync --all-extras`
- **Run GUI**: `uv run main.py` or `uv run pyroid`
- **Run CLI**: `uv run main.py --cli --output model.stl` or `uv run pyroid-cli -o model.stl`
- **Run Tests**: `uv run pytest -v`
- **Build Executable Release**: `uv run python scripts/release.py` or `.\scripts\build_release.bat`
- **Auto-Rebuild on Code Update (Watch Mode)**: `uv run python scripts/release.py --watch`
- **Install Git Commit Auto-Release Hook**: `uv run python scripts/release.py --install-git-hook`
- **Generate Parametric Diamond**: `uv run pyroid-cli -o diamond.stl --tpms diamond`
- **Convert STL with Diamond**: `uv run pyroid-cli -i input.stl -o diamond_output.stl --tpms diamond --cell-size 5.0`
- **Convert STL with Shell**: `uv run pyroid-cli -i input.stl -o gyroid_output.stl --cell-size 5.0 --preserve-shell --shell-thickness 1.0`
- **Convert STL Skeletal (Resin Drainage)**: `uv run pyroid-cli -i input.stl -o gyroid_output.stl --cell-size 5.0 --skeletal --tpms i_wp`
