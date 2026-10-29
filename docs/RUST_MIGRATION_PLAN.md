# PyRoid: Rust Backend Migration Plan & Architecture

> **Objective:** Port PyRoid's computational geometry and volumetric infill engine to pure Rust (`pyroid-core`), expose it to Python via PyO3/Maturin (`pyroid_rs`), keep the PyQt5 + PyVista desktop frontend completely intact for daily workflows, and establish an unencumbered foundation for future integration into **SolidCraft**.

---

## 1. Architectural Strategy

### The Decoupling Principle
To ensure PyRoid remains your primary, responsive daily driver while opening the door for SolidCraft:
1. **`crates/pyroid-core` (Pure Rust):** Zero Python symbols, zero PyO3, zero CPython runtime dependencies. Pure mathematical level-sets, Rayon-parallelized containment checks, distance transforms, and marching cubes. This crate will eventually be consumed directly by SolidCraft's B-Rep engine.
2. **`crates/pyroid-pyo3` (Rust/Python FFI):** Thin bridge built with **Maturin**. Handles argument decoding, drops the Python GIL (`py.allow_threads`), coordinates progress callbacks to Qt, and yields zero-copy NumPy buffers.
3. **`pyroid/` (Python Frontend):** `gui.py`, `cli.py`, and PyVista viewport logic stay in Python. The heavy compute in `core.py` and `mesh_utils.py` is replaced with one-line calls into `pyroid_rs`.

```gfm
pyroid/
├── Cargo.toml                  # Cargo workspace manifest
├── crates/
│   ├── pyroid-core/            # Standalone geometry & TPMS engine (Pure Rust)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── tpms.rs         # Multi-TPMS equations (Gyroid, Diamond, Primitive, etc.)
│   │       ├── bvh.rs          # Rayon parallel containment & ray-casting
│   │       ├── edt.rs          # 3D Euclidean Distance Transform
│   │       ├── marching.rs     # Chunked parallel Marching Cubes
│   │       └── manifold.rs     # Watertight winding & boolean operations
│   └── pyroid-pyo3/            # PyO3 module bindings
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs          # Exposes pyroid_rs C-extension
├── pyroid/                     # Untouched PyQt5 / PyVista user interface
│   ├── gui.py                  # PyQt5 GUI
│   ├── cli.py                  # CLI entry point
│   ├── core.py                 # Forwarding shim calling pyroid_rs
│   └── mesh_utils.py           # Forwarding shim calling pyroid_rs
├── pyproject.toml              # Updated to use maturin build-backend
└── uv.lock
```

---

## 2. Zero-Copy Data Handoff (Rust $\to$ PyVista)

PyVista does not need complex Python objects passed back from Rust. It only requires:
1. **Points array:** Contiguous `f32` buffer of shape `(N, 3)`.
2. **Cell connectivity array:** VTK-style `i64` or `i32` buffer of shape `(M * 4,)` where each cell is formatted as `[3, v0, v1, v2]`.

### Rust Handoff Example (`pyroid-pyo3/src/lib.rs`):
```rust
use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyroid_core::{TPMSEngine, STLParams};

#[pyfunction]
fn convert_stl_to_tpms<'py>(
    py: Python<'py>,
    stl_path: &str,
    params: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<i64>>)> {
    // 1. Parse Python parameters
    let rust_params = parse_params(params)?;

    // 2. Compute in pure Rust releasing GIL
    let mesh = py.allow_threads(|| {
        pyroid_core::convert_stl(stl_path, &rust_params)
    }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // 3. Zero-copy transfer to Python NumPy arrays
    let py_points = mesh.points_flat().into_pyarray_bound(py).reshape([mesh.n_points(), 3])?;
    let py_faces = mesh.vtk_cells().into_pyarray_bound(py);

    Ok((py_points, py_faces))
}
```

### Python Consumption (`pyroid/mesh_utils.py`):
```python
import pyvista as pv
import pyroid_rs

def convert_stl_to_gyroid(stl_mesh, params, progress_callback=None):
    # Rust generates vertices and VTK cells in parallel with GIL released
    points, cells = pyroid_rs.convert_stl_to_tpms(params.stl_path, params.to_dict())
    
    # Instantaneous PyVista construction (zero copying)
    poly = pv.PolyData(points, cells)
    return poly
```

---

## 3. Migration Roadmap & Execution Checklist

### Phase 1: Build System & Workspace Setup
- [ ] Create `Cargo.toml` at repository root configured for multi-crate workspace.
- [ ] Initialize `crates/pyroid-core` and `crates/pyroid-pyo3`.
- [ ] Switch `pyproject.toml` to `maturin`:
  ```toml
  [build-system]
  requires = ["maturin>=1.5,<2.0"]
  build-backend = "maturin"
  ```
- [ ] Verify local dev setup: `uv pip install maturin && maturin develop`.

### Phase 2: `pyroid-core` Geometric Kernels
- [ ] Port `pyroid/tpms.py` mathematical equations into `crates/pyroid-core/src/tpms.rs`:
  - Gyroid (Schoen G), Diamond (Schwarz D), Primitive (Schwarz P), I-WP, Fischer-Koch S, Neovius.
  - Implement radial cylindrical transformations ($r, \phi, z$).
  - Add analytical gradient scaling factors and thickness multipliers.
- [ ] Benchmark single-point and chunked evaluations using `criterion`.

### Phase 3: Spatial Containment & Euclidean Distance Field
- [ ] Implement parallel point-in-mesh containment using `parry3d` or AABB ray-traversal with `rayon`.
- [ ] Implement multithreaded 3D Euclidean Distance Transform (Felzenszwalb-Huttenlocher algorithm) for boundary shell preservation (`shell_thickness`).
- [ ] Implement fast streaming quantile estimator for automatic `infill_percent` targeting.

### Phase 4: Isosurface Extraction & Industrial Slicing Output
- [ ] Implement parallel Marching Cubes in Rust yielding indexed triangles.
- [ ] Enforce outward-pointing normals and non-degenerate manifold winding (eliminating Farsoon BuildStar slicing errors at the vertex/face emission level).
- [ ] Integrate native CSG boolean subtraction for shell boundary retention.

### Phase 5: Python Integration & GUI Verification
- [ ] Wire PyO3 extension into `pyroid/core.py` and `pyroid/mesh_utils.py`.
- [ ] Ensure PyQt5 background thread progress bar operates smoothly with Rust callback hooks.
- [ ] Run full test suite: `uv run pytest -v`.
- [ ] Test standalone build: `uv run python scripts/release.py`.

### Phase 6: SolidCraft Hooking (Future Target)
- [ ] Expose clean Rust trait in `pyroid-core`:
  ```rust
  pub trait LatticeInfillProvider {
      fn generate_infill(&self, boundary: &Trimesh, params: &InfillParams) -> Result<Trimesh, String>;
  }
  ```
- [ ] Reference `pyroid-core` from SolidCraft via Cargo path dependency.

---

## 4. Benchmark Targets

| Task | Python (Current Baseline) | Rust Target | Target Speedup |
| :--- | :--- | :--- | :--- |
| **Cylindrical Gyroid ($120^3$)** | ~4.2 s | **~0.15 s** | **~28x** |
| **STL Containment ($150^3$, 3.3M pts)** | ~32.0 s | **~0.90 s** | **~35x** |
| **STL Preserved Shell + TPMS ($150^3$)** | ~58.0 s | **~2.20 s** | **~26x** |
| **Peak Memory Consumption** | 3.5 – 8.0 GB | **< 600 MB** | **~85% Reduction** |
