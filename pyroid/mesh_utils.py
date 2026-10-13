"""Mesh processing and volumetric utility functions.

Provides utilities for STL loading, volumetric voxel containment testing,
gyroid lattice infill generation, watertight manifold preparation, and offscreen rendering.
"""
import os
from collections.abc import Callable

import numpy as np
import pyvista as pv
import scipy.ndimage as ndi
import trimesh

from pyroid.models import STLConversionParams


def load_stl(filepath: str) -> tuple[trimesh.Trimesh, pv.PolyData]:
    """Load an STL file into both Trimesh and PyVista objects."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"STL file not found: {filepath}")

    tri_mesh = trimesh.load(filepath, force="mesh")
    pv_mesh = pv.read(filepath)
    return tri_mesh, pv_mesh


def convert_stl_to_gyroid(
    stl_mesh: trimesh.Trimesh,
    params: STLConversionParams,
    progress_callback: Callable[[int, str], None] | None = None,
) -> pv.PolyData:
    """
    Convert an input STL 3D mesh volume into a gyroid lattice structure.

    Args:
        stl_mesh: Input Trimesh object (should be a manifold/closed surface)
        params: STLConversionParams configuration

    Returns:
        PyVista PolyData object representing the generated gyroid mesh within the STL boundary.
    """
    # Get bounding box
    min_b, max_b = stl_mesh.bounds
    extent = max_b - min_b
    max_dim = max(extent)

    if max_dim <= 0:
        raise ValueError("Invalid STL mesh with zero extent")

    # Determine voxel resolution per dimension with slight bounding box padding
    # to ensure containment and distance transforms do not clip at the bbox boundary.
    res = params.resolution
    pad = 2.0 * max_dim / res
    grid_min = min_b - pad
    grid_max = max_b + pad
    grid_extent = grid_max - grid_min

    nx = max(12, int(np.round(res * (grid_extent[0] / max_dim))))
    ny = max(12, int(np.round(res * (grid_extent[1] / max_dim))))
    nz = max(12, int(np.round(res * (grid_extent[2] / max_dim))))

    if progress_callback:
        progress_callback(10, f"Initializing 3D voxel grid ({nx}x{ny}x{nz})...")

    # Create 3D grid across padded bounding box
    x, y, z = np.mgrid[
        grid_min[0] : grid_max[0] : nx * 1j,
        grid_min[1] : grid_max[1] : ny * 1j,
        grid_min[2] : grid_max[2] : nz * 1j,
    ]

    # Reshape grid points for query
    pts = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    if progress_callback:
        progress_callback(25, f"Testing volume containment ({len(pts):,} points)...")

    # Evaluate interior containment using trimesh ray containment with PyVista fallback
    try:
        inside_mask = stl_mesh.contains(pts)
    except Exception:
        pts_cloud = pv.PolyData(pts)
        pv_stl = pv.wrap(stl_mesh)
        enclosed = pts_cloud.select_enclosed_points(pv_stl, tolerance=1e-5)
        inside_mask = enclosed["SelectedPoints"].astype(bool)

    if progress_callback:
        progress_callback(50, "Evaluating 3D gyroid scalar field...")

    # Gyroid 3D scalar equation: sin(kx*x)*cos(ky*y) + sin(ky*y)*cos(kz*z) + sin(kz*z)*cos(kx*x)
    k = 2.0 * np.pi / max(params.cell_size, 0.001)
    gyroid_val = (
        np.sin(k * x) * np.cos(k * y)
        + np.sin(k * y) * np.cos(k * z)
        + np.sin(k * z) * np.cos(k * x)
    )

    if params.is_skeletal:
        # Skeletal strut network: single continuous void channel for resin draining.
        # Strut thickness is modulated by shifting the isovalue threshold around nominal 50% porosity.
        r = params.wall_thickness / max(params.cell_size, 0.001)
        thresh = np.clip(3.0 * (r - 0.2), -1.1, 1.1)
        void_scalar = thresh - gyroid_val
        field = gyroid_val - thresh
    else:
        # Sheet gyroid: two intertwined void channels separated by continuous walls.
        # Thickness threshold calibrated to physical Euclidean thickness:
        # Scaling by 5.0 accounts for the steep spatial gradient of the gyroid trigonometric field (||grad G|| ≈ 1.53 * 2*pi/L ≈ 9.6/L).
        t = np.clip(5.0 * params.wall_thickness / params.cell_size, 0.05, 1.5)
        void_scalar = t - np.abs(gyroid_val)
        field = np.abs(gyroid_val) - t

    # Mask out points outside the STL geometry
    inside_3d = inside_mask.reshape(x.shape)

    # Preserve outer shell if configured:
    # Instead of voxelizing the outer boundary, we compute the internal gyroid voids
    # (receded by shell_thickness from the boundary) and subtract them from the exact original STL mesh.
    if params.preserve_shell and params.shell_thickness > 0:
        if progress_callback:
            progress_callback(65, "Computing shell distance transform...")

        dx = grid_extent[0] / max(nx - 1, 1)
        dy = grid_extent[1] / max(ny - 1, 1)
        dz = grid_extent[2] / max(nz - 1, 1)
        dist_inside = ndi.distance_transform_edt(inside_3d, sampling=(dx, dy, dz))

        # Void exists only where dist_inside > shell_thickness AND pore condition met.
        field_void = np.maximum(
            params.shell_thickness - dist_inside, void_scalar
        )

        if progress_callback:
            progress_callback(75, "Extracting internal void isosurface...")

        grid = pv.StructuredGrid(x, y, z)
        grid["vol"] = field_void.ravel("F")
        void_mesh = grid.contour([0])

        # If no internal void formed (e.g. wall thickness encompasses the entire volume)
        if void_mesh.n_points == 0 or void_mesh.n_cells == 0:
            if progress_callback:
                progress_callback(95, "Finalizing mesh...")
            return pv.wrap(stl_mesh)

        faces = void_mesh.faces.reshape((-1, 4))[:, 1:]

        # Prepare clean STL mesh for boolean subtraction
        clean_stl = stl_mesh.copy()
        if not clean_stl.is_watertight:
            clean_stl.fill_holes()
        if clean_stl.volume < 0:
            clean_stl.faces = clean_stl.faces[:, ::-1]

        # Robust boolean subtraction: outer STL minus internal gyroid void.
        # Keeps original CAD surface triangles intact while skipping redundant Python validation.
        import manifold3d

        if progress_callback:
            progress_callback(85, "Performing manifold boolean subtraction...")

        m_outer = manifold3d.Manifold(
            manifold3d.Mesh(
                vert_properties=np.asarray(clean_stl.vertices, dtype=np.float32),
                tri_verts=np.asarray(clean_stl.faces, dtype=np.uint32),
            )
        )
        m_void = manifold3d.Manifold(
            manifold3d.Mesh(
                vert_properties=np.asarray(void_mesh.points, dtype=np.float32),
                tri_verts=np.asarray(faces, dtype=np.uint32),
            )
        )
        diff = m_outer - m_void
        res_mesh = diff.to_mesh()
        tri_result = trimesh.Trimesh(
            vertices=res_mesh.vert_properties, faces=res_mesh.tri_verts, process=False
        )

        if progress_callback:
            progress_callback(95, "Wrapping final mesh...")

        return pv.wrap(tri_result)

    # No shell preservation: gyroid infill throughout the volume
    if progress_callback:
        progress_callback(75, "Extracting gyroid isosurface contour...")

    field[~inside_3d] = (
        1.0  # Set outside points high so isosurface doesn't form outside boundary
    )

    # Construct PyVista grid and contour isosurface at 0
    grid = pv.StructuredGrid(x, y, z)
    grid["vol"] = field.ravel("F")
    mesh = grid.contour([0])

    if progress_callback:
        progress_callback(95, "Finalizing mesh...")

    return mesh


def prepare_for_printing(
    mesh: pv.PolyData | trimesh.Trimesh, flat_cap: bool = False
) -> trimesh.Trimesh:
    """Ensure generated mesh is manifold/watertight for 3D printing."""
    if isinstance(mesh, pv.PolyData):
        faces = mesh.faces.reshape((-1, 4))[:, 1:]
        tri = trimesh.Trimesh(vertices=mesh.points, faces=faces)
    else:
        tri = mesh.copy()

    if flat_cap and len(tri.vertices) > 0:
        z_min, z_max = tri.bounds[:, 2]
        # Trim slightly off top/bottom extreme limits if needed
        margin = (z_max - z_min) * 0.01
        if margin > 0.001:
            try:
                tri = trimesh.intersections.slice_mesh_plane(
                    tri,
                    plane_normal=[0, 0, -1],
                    plane_origin=[0, 0, z_max - margin],
                    cap=True,
                )
                tri = trimesh.intersections.slice_mesh_plane(
                    tri,
                    plane_normal=[0, 0, 1],
                    plane_origin=[0, 0, z_min + margin],
                    cap=True,
                )
            except Exception:
                pass

    if not tri.is_watertight:
        tri.fill_holes()

    if tri.volume < 0:
        tri.faces = tri.faces[:, ::-1]

    return tri


def render_offscreen(
    mesh: pv.PolyData, output_image_path: str, window_size: tuple[int, int] = (800, 600)
):
    """Render a thumbnail image of the mesh headlessly."""
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color="lightblue", show_edges=True)
    plotter.add_bounding_box()
    plotter.show_axes()
    plotter.reset_camera()
    plotter.screenshot(output_image_path, window_size=window_size)
    plotter.close()


def compute_subsurface_void_map(
    mesh: pv.PolyData,
    cell_size: float = 5.0,
    wall_thickness: float = 0.8,
    shell_thickness: float = 1.0,
    is_skeletal: bool = False,
    depth: float | None = None,
) -> np.ndarray:
    """
    Compute a subsurface void score across mesh vertices.

    Samples points at distance `depth` inwards along each vertex normal to evaluate
    the internal gyroid scalar field. Positive scores represent open internal caverns/voids
    (ideal drainage piercing locations), whereas negative scores represent solid struts/walls.
    """
    if len(mesh.points) == 0:
        return np.array([], dtype=np.float32)

    sample_depth = depth if depth is not None else (shell_thickness + 0.5)

    mesh_copy = mesh.copy()
    if "Normals" not in mesh_copy.point_data:
        mesh_copy.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    normals = mesh_copy.point_data["Normals"]

    pts_sub = mesh_copy.points - sample_depth * normals
    k = 2.0 * np.pi / max(cell_size, 0.001)
    x, y, z = pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2]
    gyroid_val = (
        np.sin(k * x) * np.cos(k * y)
        + np.sin(k * y) * np.cos(k * z)
        + np.sin(k * z) * np.cos(k * x)
    )

    if is_skeletal:
        r = wall_thickness / max(cell_size, 0.001)
        thresh = np.clip(3.0 * (r - 0.2), -1.1, 1.1)
        void_scores = gyroid_val - thresh
    else:
        t = np.clip(5.0 * wall_thickness / cell_size, 0.05, 1.5)
        void_scores = np.abs(gyroid_val) - t

    return void_scores.astype(np.float32)


def pierce_drain_hole(
    mesh: pv.PolyData | trimesh.Trimesh,
    point: tuple[float, float, float] | np.ndarray,
    normal: tuple[float, float, float] | np.ndarray | None = None,
    radius: float = 1.5,
    depth: float = 6.0,
) -> pv.PolyData:
    """
    Pierce a cylindrical drainage or vent hole through the shell into the internal cavity.

    Args:
        mesh: Input PyVista PolyData or Trimesh.
        point: 3D coordinates on the surface where the hole should be drilled.
        normal: Outward surface normal vector. If None, derived from closest vertex normal.
        radius: Hole radius in mm (default 1.5 mm = 3.0 mm diameter).
        depth: Hole penetration depth in mm (default 6.0 mm).

    Returns:
        PyVista PolyData mesh with the hole cleanly carved out.
    """
    import manifold3d

    if isinstance(mesh, pv.PolyData):
        faces = mesh.faces.reshape((-1, 4))[:, 1:]
        tri = trimesh.Trimesh(vertices=mesh.points, faces=faces, process=False)
    else:
        tri = mesh.copy()

    p = np.asarray(point, dtype=np.float64)
    if normal is None:
        tri.fix_normals()
        idx = tri.kdtree.query(p)[1]
        n = tri.vertex_normals[idx]
    else:
        n = np.asarray(normal, dtype=np.float64)

    norm = np.linalg.norm(n)
    n = n / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])

    p_start = p + 0.5 * n
    p_end = p - depth * n
    cyl = trimesh.creation.cylinder(radius=max(radius, 0.1), segment=[p_start, p_end], sections=32)

    m_mesh = manifold3d.Manifold(
        manifold3d.Mesh(
            vert_properties=np.asarray(tri.vertices, dtype=np.float32),
            tri_verts=np.asarray(tri.faces, dtype=np.uint32),
        )
    )
    m_cyl = manifold3d.Manifold(
        manifold3d.Mesh(
            vert_properties=np.asarray(cyl.vertices, dtype=np.float32),
            tri_verts=np.asarray(cyl.faces, dtype=np.uint32),
        )
    )

    diff = m_mesh - m_cyl
    res = diff.to_mesh()
    tri_res = trimesh.Trimesh(
        vertices=res.vert_properties, faces=res.tri_verts, process=False
    )
    return pv.wrap(tri_res)


def auto_pierce_drain_holes(
    mesh: pv.PolyData | trimesh.Trimesh,
    radius: float = 1.5,
    depth: float = 6.0,
    count: int = 2,
) -> pv.PolyData:
    """
    Automatically pierce drainage and vent holes at optimal extreme positions.

    Hole 1 is placed at the lowest point along the Z axis (gravity drain).
    Hole 2 (if count >= 2) is placed at the highest point along the Z axis (anti-suction air vent).
    """
    current_mesh = mesh if isinstance(mesh, pv.PolyData) else pv.wrap(mesh)
    pts = current_mesh.points
    if len(pts) == 0:
        return current_mesh

    # Primary drain hole at lowest Z point
    p_bottom = pts[np.argmin(pts[:, 2])]
    current_mesh = pierce_drain_hole(current_mesh, point=p_bottom, radius=radius, depth=depth)

    if count >= 2:
        pts2 = current_mesh.points
        p_top = pts2[np.argmax(pts2[:, 2])]
        current_mesh = pierce_drain_hole(current_mesh, point=p_top, radius=radius, depth=depth)

    return current_mesh

