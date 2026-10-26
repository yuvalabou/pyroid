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
from pyroid.tpms import get_tpms_kernel


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

    kernel = get_tpms_kernel(params.tpms_type)
    if progress_callback:
        progress_callback(50, f"Evaluating 3D {kernel.display_name} scalar field...")

    # TPMS 3D scalar equation evaluated on periodic wave grid
    k = 2.0 * np.pi / max(params.cell_size, 0.001)
    scalar_val = kernel.eval(k * x, k * y, k * z) + params.bias

    # Mask out points outside the STL geometry
    inside_3d = inside_mask.reshape(x.shape)
    q = (
        np.clip(params.infill_percent / 100.0, 0.01, 0.99)
        if params.infill_percent is not None and inside_3d.any()
        else None
    )

    t_norm = kernel.thickness_multiplier / 4.80
    t = (
        float(np.quantile(np.abs(scalar_val[inside_3d]), q))
        if (q is not None and not params.is_skeletal)
        else float(
            np.clip(
                kernel.thickness_multiplier * params.wall_thickness / max(params.cell_size, 0.001),
                0.05,
                1.5 * t_norm,
            )
        )
    )
    if params.is_skeletal:
        # Skeletal strut network: single continuous void channel for resin draining.
        r = params.wall_thickness / max(params.cell_size, 0.001)
        scale_strut = 3.0 * t_norm
        thresh = (
            float(np.quantile(scalar_val[inside_3d], q))
            if q is not None
            else float(np.clip(scale_strut * (r - 0.2), -1.1 * t_norm, 1.1 * t_norm))
        )
        void_scalar = thresh - scalar_val
        field = scalar_val - thresh
    else:
        # Sheet TPMS: two intertwined void channels separated by continuous walls.
        void_scalar = t - np.abs(scalar_val)
        field = np.abs(scalar_val) - t

    grid = pv.StructuredGrid(x, y, z)

    def _attach_cavities(out_mesh: pv.PolyData, mask_3d: np.ndarray) -> pv.PolyData:
        if params.is_skeletal:
            return out_mesh
        grid["vol"] = np.where(mask_3d, t - scalar_val, 1.0).ravel("F")
        pv.set_new_attribute(out_mesh, "cavity_a", grid.contour([0]))
        grid["vol"] = np.where(mask_3d, scalar_val + t, 1.0).ravel("F")
        pv.set_new_attribute(out_mesh, "cavity_b", grid.contour([0]))
        return out_mesh

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
        inner_mask = dist_inside > params.shell_thickness

        # Void exists only where dist_inside > shell_thickness AND pore condition met.
        field_void = np.maximum(params.shell_thickness - dist_inside, void_scalar)

        if progress_callback:
            progress_callback(75, "Extracting internal void isosurface...")

        grid["vol"] = field_void.ravel("F")
        void_mesh = grid.contour([0])

        # If no internal void formed (e.g. wall thickness encompasses the entire volume)
        if void_mesh.n_points == 0 or void_mesh.n_cells == 0:
            if progress_callback:
                progress_callback(95, "Finalizing mesh...")
            return _attach_cavities(pv.wrap(stl_mesh), inner_mask)

        faces = void_mesh.faces.reshape((-1, 4))[:, 1:]

        # Prepare clean STL mesh for boolean subtraction
        clean_stl = stl_mesh.copy()
        if not clean_stl.is_watertight:
            clean_stl.fill_holes()
        if clean_stl.volume < 0:
            clean_stl.faces = clean_stl.faces[:, ::-1]

        # Ensure void mesh has positive signed volume and consistent outward winding
        # (VTK contour generates inverted winding for maximum distance-field thresholds)
        tri_void = trimesh.Trimesh(vertices=void_mesh.points, faces=faces, process=False)
        if len(tri_void.faces) > 0:
            tri_void.update_faces(tri_void.nondegenerate_faces())
            tri_void.update_faces(tri_void.unique_faces())
            tri_void.remove_unreferenced_vertices()
            if not tri_void.is_watertight:
                tri_void.fill_holes()
            if tri_void.volume < 0:
                tri_void.faces = tri_void.faces[:, ::-1]
            faces = tri_void.faces
            void_points = tri_void.vertices
        else:
            void_points = void_mesh.points

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
                vert_properties=np.asarray(void_points, dtype=np.float32),
                tri_verts=np.asarray(faces, dtype=np.uint32),
            )
        )
        diff = m_outer - m_void
        res_mesh = diff.to_mesh()
        tri_result = trimesh.Trimesh(
            vertices=res_mesh.vert_properties, faces=res_mesh.tri_verts, process=False
        )
        if tri_result.volume < 0:
            tri_result.faces = tri_result.faces[:, ::-1]

        if progress_callback:
            progress_callback(95, "Wrapping final mesh...")

        return _attach_cavities(pv.wrap(tri_result), inner_mask)

    # No shell preservation: gyroid infill throughout the volume
    if progress_callback:
        progress_callback(75, "Extracting gyroid isosurface contour...")

    field[~inside_3d] = 1.0  # Set outside points high so isosurface doesn't form outside boundary

    # Construct PyVista grid and contour isosurface at 0
    grid["vol"] = field.ravel("F")
    mesh = grid.contour([0])
    _attach_cavities(mesh, inside_3d)

    if progress_callback:
        progress_callback(95, "Finalizing mesh...")

    return mesh


def prepare_for_printing(
    mesh: pv.PolyData | trimesh.Trimesh, flat_cap: bool = False
) -> trimesh.Trimesh:
    """
    Ensure generated mesh is manifold/watertight with consistent winding for 3D printing.
    Optimized for industrial SLS/SLM slicers (such as Farsoon BuildStar) and Netfabb.
    """
    if isinstance(mesh, pv.PolyData):
        tri_pv = mesh.triangulate() if mesh.faces.size > 0 else mesh
        faces = tri_pv.faces.reshape((-1, 4))[:, 1:]
        tri = trimesh.Trimesh(vertices=tri_pv.points, faces=faces, process=False)
    else:
        tri = mesh.copy()

    # Clean initial duplicate/degenerate geometry
    if len(tri.faces) > 0:
        tri.update_faces(tri.nondegenerate_faces())
        tri.update_faces(tri.unique_faces())
        tri.remove_unreferenced_vertices()

    if flat_cap and len(tri.vertices) > 0:
        z_min, z_max = tri.bounds[:, 2]
        margin = (z_max - z_min) * 0.01
        if margin > 0.001:
            clipped = False
            try:
                import manifold3d

                m = manifold3d.Manifold(
                    manifold3d.Mesh(
                        vert_properties=np.asarray(tri.vertices, dtype=np.float32),
                        tri_verts=np.asarray(tri.faces, dtype=np.uint32),
                    )
                )
                m = m.trim_by_plane((0, 0, -1), -(z_max - margin)).trim_by_plane(
                    (0, 0, 1), z_min + margin
                )
                res = m.to_mesh()
                tri = trimesh.Trimesh(
                    vertices=res.vert_properties, faces=res.tri_verts, process=False
                )
                clipped = True
            except Exception:
                clipped = False

            if not clipped:
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

    # Ensure watertightness and repair broken faces
    if not tri.is_watertight:
        tri.fill_holes()

    if len(tri.faces) > 0:
        tri.update_faces(tri.nondegenerate_faces())
        tri.update_faces(tri.unique_faces())
        tri.remove_unreferenced_vertices()

        # If manifold3d is available, validate 2-manifold orientation
        try:
            import manifold3d

            m_check = manifold3d.Manifold(
                manifold3d.Mesh(
                    vert_properties=np.asarray(tri.vertices, dtype=np.float32),
                    tri_verts=np.asarray(tri.faces, dtype=np.uint32),
                )
            )
            if m_check.status().name == "NoError" and m_check.volume() > 0:
                m_mesh = m_check.to_mesh()
                tri = trimesh.Trimesh(
                    vertices=m_mesh.vert_properties, faces=m_mesh.tri_verts, process=False
                )
        except Exception:
            pass

        # Unify counter-clockwise winding order and outward normals
        trimesh.repair.fix_winding(tri)
        trimesh.repair.fix_normals(tri)
        trimesh.repair.fix_inversion(tri)

        if tri.volume < 0:
            tri.faces = tri.faces[:, ::-1]
            trimesh.repair.fix_normals(tri)

    return tri


def export_stl_binary(
    mesh: pv.PolyData | trimesh.Trimesh,
    filepath: str,
    header_text: str = "PyRoid 3D Model - Farsoon BuildStar and Netfabb Compatible",
    flat_cap: bool = False,
) -> None:
    """
    Export mesh to a strictly standard-compliant binary STL file.

    Guarantees compatibility with industrial Laser Powder Bed Fusion (SLS/SLM)
    slicers (such as Farsoon BuildStar, EOS RP-Tools, Materialise Magics) and CAD tools (Netfabb):
      - 80-byte ASCII header padded with spaces (never starts with 'solid' or null bytes).
      - Little-endian 32-bit unsigned facet count.
      - 50 bytes per facet with valid, normalized non-zero unit normals.
      - Watertight topology, consistent counter-clockwise face winding, no degenerate facets.
    """
    tri = prepare_for_printing(mesh, flat_cap=flat_cap)

    # Clean any residual degenerate or unreferenced elements
    if len(tri.faces) > 0:
        tri.update_faces(tri.nondegenerate_faces())
        tri.update_faces(tri.unique_faces())
        tri.remove_unreferenced_vertices()
        trimesh.repair.fix_winding(tri)
        trimesh.repair.fix_normals(tri)

    # Compute explicit unit normal vectors per face to prevent any zero/NaN vectors in CAM slicers
    v = tri.vertices[tri.faces]
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    zero_mask = (norms < 1e-12).squeeze()
    normals[zero_mask] = [0.0, 0.0, 1.0]
    norms[zero_mask] = 1.0
    normals = (normals / norms).astype("<f4")

    vertices = tri.vertices[tri.faces].astype("<f4")
    n_faces = len(tri.faces)

    # 80-byte header: valid ASCII, padded with spaces, never starts with 'solid'
    header_clean = (
        header_text.replace("\r", " ").replace("\n", " ").encode("ascii", errors="replace")
    )
    if header_clean.lower().startswith(b"solid"):
        header_clean = b"PyRoid " + header_clean
    header = header_clean[:80].ljust(80, b" ")

    dtype = np.dtype([
        ("normals", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attr", "<u2"),
    ])
    data = np.zeros(n_faces, dtype=dtype)
    data["normals"] = normals
    data["vertices"] = vertices
    data["attr"] = 0

    with open(filepath, "wb") as f:
        f.write(header)
        f.write(np.uint32(n_faces).tobytes())
        f.write(data.tobytes())


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
