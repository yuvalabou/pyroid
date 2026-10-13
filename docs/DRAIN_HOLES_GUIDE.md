# PyRoid Drain Holes Engine: Comprehensive Implementation Guide

## 1. Executive Summary & Context

This document lays out the exact architecture, mathematics, code modifications, and test suite required to finalize the **Resin Drainage & Vent Hole System** in PyRoid on branch `drain-holes`.

### What We Observed in Initial Testing
1. **Capped Holes ("Blind Pipes"):** Punching a drain hole created a cylindrical cut that terminated in a flat circular cap inside the part rather than cleanly opening into the internal void cavity.
2. **Crooked Drill Angles:** Holes were drilled at odd, tilted angles instead of cleanly perpendicular to the exterior face or aligned with gravity.
3. **Guesswork in Hole Placement:** Users had to guess where to drill to avoid hitting internal gyroid struts instead of voids.

---

## 2. Root Cause Analysis

### A. Why Holes Were Capped
* **Current mechanism:** `trimesh.creation.cylinder(radius, segment=[p_start, p_end])` creates a **closed solid mesh** with flat disk caps on both ends.
* **The Failure Mode:** When drilling with a fixed depth (e.g., $6\text{ mm}$), the bottom cap stops inside solid internal gyroid material (or past a thin void into the opposite strut). 
* When `manifold3d` subtracts `Mesh - Cylinder`, it imprints the cylinder's bottom disk into the solid geometry, producing a blind hole.
* **The Solution ("Pierce Until Void"):** A drill must not use a static blind depth. It must cast an inward ray, detect the exact point where it breaches into the open internal cavern, and extend the cutter cylinder into empty air ($+2\text{ to }4\text{ mm}$ into the void). Because the cylinder's bottom cap floats in empty air where there is no mesh geometry, Manifold produces a clean through-port with **zero end cap**.

```
[ Exterior Air ]        [ Shell Wall ]         [ Internal Gyroid Void ]
       |                      |                         |
Surface Point P ──────> (Wall Thickness) ────> Breach Point (Empty Cavern) ────> End Cap Floating in Air
       |                                                |                              |
       +─────────────────── Cylinder Cutter Length ─────┴──────────────────────────────+
                                  (Wall + 3.0 mm clearance)
```

---

### B. Why Holes Were Crooked
* **Current mechanism:** `tri.kdtree.query(p)[1]` looks up the nearest **mesh vertex**.
* **The Failure Mode:** On coarse planar faces (e.g., an STL's flat bottom face), the nearest outer vertex can be $10\text{–}15\text{ mm}$ away at a corner. But an internal gyroid vertex behind the shell wall may only be $1.2\text{ mm}$ away!
* The KDTree query was erroneously grabbing **internal, highly curved gyroid vertices** whose normals point diagonally at 45°–70°, causing the cutter to shoot sideways.
* **The Solution:** 
  1. Use `trimesh.proximity.closest_point` to find the exact **polygon triangle face** and its true geometric face normal.
  2. In the GUI, support **Orientation Modes**:
     * **Perpendicular (Face Normal):** Drills directly inward along $-N_{\text{face}}$.
     * **Gravity Aligned (Vertical):** Drills straight along $-Z$ (ideal for flat bottom drain ports in resin printing).

---

### C. Preemptive Candidate Generation (Option 3)
Instead of forcing the user to interpret a complex heatmap and click blindly:
1. PyRoid automatically inspects the lower surface of the shelled model.
2. Identifies **2 to 5 candidate drain ports** where:
   * The face points downward ($N_z < -0.3$).
   * An inward ray penetrates through a thin shell ($< 3\text{ mm}$) and hits a large void cavern (clearance $> 5\text{ mm}$).
3. Visualizes these candidates in the 3D viewport as green interactive target rings.
4. The user can either click "Punch Selected", click individual candidate rings to toggle them, or freely click anywhere using the precision cutter.

---

## 3. Architecture & File Breakdown

```
pyroid/
├── models.py           # Dataclasses (DrainPort, DrainHoleParams)
├── mesh_utils.py       # Core geometry: Face normal picking, Ray-march void breach,
│                       # Candidate port detection, Manifold uncapped boolean cutter
├── core.py             # GyroidGenerator API wrappers
├── gui.py              # Visual candidate markers, direction/depth toggles, interactive picking
└── cli.py              # CLI flags for automated candidate and precision drainage
```

---

## 4. Detailed Implementation Blueprint

### Step 1: Data Structures (`pyroid/models.py`)
Add strongly typed configurations for drain holes:

```python
from dataclasses import dataclass
from typing import Literal
import numpy as np

@dataclass
class DrainHoleConfig:
    radius: float = 1.5                  # 3.0 mm diameter
    mode: Literal["until_void", "fixed", "through"] = "until_void"
    fixed_depth: float = 6.0
    orientation: Literal["normal", "gravity_down", "custom"] = "normal"
    clearance_into_void: float = 3.0     # mm to extend past shell into cavern

@dataclass
class DrainPortCandidate:
    id: int
    point: tuple[float, float, float]
    direction: tuple[float, float, float]
    wall_thickness: float
    cavern_clearance: float
    score: float                         # Higher = thinnest wall + largest cavern
```

---

### Step 2: Precision Geometry Engine (`pyroid/mesh_utils.py`)

#### A. Precise Face Normal Extraction
```python
def get_surface_face_normal(tri: trimesh.Trimesh, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the exact closest surface point on the exterior mesh and its true face normal.
    Prevents picking internal gyroid vertices.
    """
    closest_pts, distances, face_indices = trimesh.proximity.closest_point(tri, [point])
    face_idx = face_indices[0]
    face_normal = tri.face_normals[face_idx]
    norm = np.linalg.norm(face_normal)
    normal = face_normal / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
    return closest_pts[0], normal
```

#### B. Dynamic "Pierce Until Void" Calculation
```python
def compute_pierce_depth_until_void(
    tri: trimesh.Trimesh,
    point: np.ndarray,
    direction: np.ndarray,
    max_search_depth: float = 25.0,
    step_size: float = 0.2,
    cell_size: float = 5.0,
    wall_thickness: float = 1.0,
    is_skeletal: bool = True,
) -> float:
    """
    Ray-marches from `point` along `direction` through the shell wall into the internal void.
    Returns the depth required to cleanly emerge into the void + clearance, preventing capped holes.
    """
    # 1. Cast ray inward against internal geometry faces using trimesh ray-mesh intersection:
    ray_origins = np.array([point + 0.01 * direction])
    ray_directions = np.array([direction])
    
    locations, index_ray, index_tri = tri.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
    )
    
    # If ray hits an internal wall (the inner boundary of the shell):
    if len(locations) > 0:
        dists = np.linalg.norm(locations - point, axis=1)
        # The first intersection is the inner surface of the shell wall
        wall_t = float(np.min(dists))
        # Add 2.5mm clearance so the cylinder cap hangs in open cavern air
        return wall_t + 2.5

    # Fallback default if ray misses internal triangle:
    return max(wall_thickness * 2.0 + 2.0, 6.0)
```

#### C. Candidate Port Detection Algorithm
```python
def detect_candidate_drain_ports(
    mesh: pv.PolyData | trimesh.Trimesh,
    count: int = 4,
    min_spacing: float = 10.0,
) -> list[DrainPortCandidate]:
    """
    Scans the downward-facing exterior faces (Nz < -0.3) of the model.
    Evaluates:
      1. Proximity to bottom (Z elevation).
      2. Distance through shell into void.
      3. Void cavern space behind the wall.
    Returns ranked candidate ports with non-overlapping spacing.
    """
```

#### D. Uncapped Boolean Subtraction
```python
def pierce_drain_hole_precision(
    mesh: pv.PolyData | trimesh.Trimesh,
    point: np.ndarray,
    normal: np.ndarray | None = None,
    radius: float = 1.5,
    orientation_mode: str = "normal",  # 'normal' or 'gravity_down'
    depth_mode: str = "until_void",    # 'until_void' or 'fixed'
    fixed_depth: float = 6.0,
) -> pv.PolyData:
    """
    Carves an uncapped, perfectly oriented hole through the shell into the void cavern.
    """
```

---

### Step 3: GUI Enhancements (`pyroid/gui.py`)

#### Controls in the "Drain Holes" Dock Section:
1. **Direction Radio/Combo:**
   * `Perpendicular (Surface Normal)` — follows local face orientation.
   * `Vertical (-Z Gravity)` — drills straight downward for horizontal build plates.
2. **Depth Mode Combo:**
   * `Pierce Until Void (Auto-Through)` — automatically computes penetration into cavity.
   * `Custom Depth (mm)` — manual depth spinbox.
3. **"Detect Optimal Ports" Button:**
   * Runs `detect_candidate_drain_ports`.
   * Adds green ring/target actors in the PyVista 3D viewport.
   * Adds a list/table of candidates with checkboxes.
4. **"Punch Candidate #X" / "Punch All Detected":**
   * Executes the uncapped cuts with progress feedback.
5. **Interactive Picking Refinement:**
   * Switch PyVista picker to `picker="cell"` instead of `picker="point"`.
   * Extract cell/face normal directly from the picked cell.

---

### Step 4: CLI Flags (`pyroid/cli.py`)

Extend existing CLI args:
* `--drain-orientation {normal,vertical}`: Selects drill orientation vector.
* `--drain-mode {until-void,fixed}`: Auto-breach vs fixed length.
* `--detect-drain-ports`: Analyzes and outputs suggested coordinates.

---

## 5. Verification & Test Plan

1. **Unit Test: Face Normal Accuracy**
   * Create a cube mesh. Pick a point on the $-Z$ bottom face. Verify that normal is `[0, 0, -1]`, not tilted.
2. **Unit Test: Uncapped Borehole Verification**
   * Carve a hole with `depth_mode="until_void"` on a shelled gyroid sphere.
   * Verify that the internal void volume is now geometrically open to the exterior (Euler characteristic and boundary loops change as expected, indicating an open genus puncture).
3. **Unit Test: Preemptive Candidate Detection**
   * Generate a shelled model. Run `detect_candidate_drain_ports(mesh, count=3)`.
   * Verify candidate coordinates lie on the exterior boundary and point downward.
4. **CLI Integration Test**
   * Run `uv run pyroid-cli -i test_input.stl -o out.stl --preserve-shell --auto-drain-holes --drain-orientation vertical`.

---

## 6. Tomorrow's Step-by-Step Execution Checklist

- [ ] **Checklist Item 1:** Update `pyroid/mesh_utils.py` with `get_surface_face_normal` and ray-cast `compute_pierce_depth_until_void`.
- [ ] **Checklist Item 2:** Refactor `pierce_drain_hole` to use the face normal and dynamic void breach depth.
- [ ] **Checklist Item 3:** Implement `detect_candidate_drain_ports` in `pyroid/mesh_utils.py`.
- [ ] **Checklist Item 4:** Expose the new methods on `GyroidGenerator` in `pyroid/core.py`.
- [ ] **Checklist Item 5:** Update `pyroid/gui.py` with the candidate visual markers and new parameter controls.
- [ ] **Checklist Item 6:** Update `pyroid/cli.py` with `--drain-orientation` and `--drain-mode`.
- [ ] **Checklist Item 7:** Run `uv run pytest -v` to ensure all tests pass.
- [ ] **Checklist Item 8:** Test in the GUI (`uv run main.py`) with a sample STL.
