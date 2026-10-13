#!/usr/bin/env python3
"""Command Line Interface (CLI) for PyRoid.

Provides commands and options to generate radially symmetrical gyroid meshes
and convert arbitrary STL 3D models into gyroid lattices.
"""
import argparse
import json
import os
import sys

from pyroid.core import GyroidGenerator
from pyroid.models import GyroidParams, STLConversionParams


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PyRoid - Command line tool for generating gyroid structures and converting STL meshes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output file
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output mesh file path (.stl or .obj extension)",
    )

    # STL Import & Conversion Group
    stl_group = parser.add_argument_group("STL Volume Infill Conversion")
    stl_group.add_argument(
        "--input-stl",
        "-i",
        type=str,
        default=None,
        help="Input STL file path to convert volume into a gyroid lattice",
    )
    stl_group.add_argument(
        "--cell-size",
        type=float,
        default=5.0,
        help="Gyroid unit cell size (mm or model units) for STL conversion",
    )
    stl_group.add_argument(
        "--flat-cap",
        action="store_true",
        default=False,
        help="Planar slice and cap surface ends for 3D printing",
    )
    stl_group.add_argument(
        "--preserve-shell",
        action="store_true",
        default=False,
        help="Preserve outer boundary of input STL as a solid shell",
    )
    stl_group.add_argument(
        "--shell-thickness",
        type=float,
        default=1.0,
        help="Outer shell thickness (mm or model units) when --preserve-shell is enabled",
    )
    stl_group.add_argument(
        "--skeletal",
        action="store_true",
        default=False,
        help="Generate skeletal strut-based lattice (single continuous void) instead of sheet walls",
    )
    stl_group.add_argument(
        "--auto-drain-holes",
        action="store_true",
        default=False,
        help="Automatically pierce drain and vent holes at lowest and highest Z coordinates for resin draining",
    )
    stl_group.add_argument(
        "--drain-hole-diameter",
        type=float,
        default=3.0,
        help="Diameter (mm) of pierced drain holes (default: 3.0)",
    )
    stl_group.add_argument(
        "--drain-hole-depth",
        type=float,
        default=6.0,
        help="Penetration depth (mm) of pierced drain holes (default: 6.0)",
    )

    # Parametric Gyroid Group
    param_group = parser.add_argument_group("Parametric Gyroid Parameters")
    param_group.add_argument(
        "--res", type=int, default=80, help="Resolution of the grid in each dimension"
    )
    param_group.add_argument("--a", type=float, default=24, help="X-axis length")
    param_group.add_argument("--b", type=float, default=24, help="Y-axis length")
    param_group.add_argument("--c", type=float, default=10, help="Z-axis length")
    param_group.add_argument("--r1", type=float, default=12, help="Inner radius")
    param_group.add_argument("--r2", type=float, default=0, help="Outer radius")
    param_group.add_argument(
        "--phi-scale", type=float, default=8, help="Angular scaling factor"
    )
    param_group.add_argument(
        "--wall-thickness",
        type=float,
        default=None,
        help="Wall thickness (mm for STL conversion, default 0.8; annular shell width for parametric, default 11.5)",
    )
    param_group.add_argument("--cell-radius", type=float, default=2, help="Cell radius")
    param_group.add_argument("--cell-height", type=float, default=3, help="Cell height")

    # Preset handling
    preset_group = parser.add_argument_group("Preset Management")
    preset_group.add_argument(
        "--preset",
        type=str,
        help="Use a named preset (default, fine_detail, thick_walls, dense_pattern)",
    )
    preset_group.add_argument(
        "--load-preset", type=str, help="Load parameters from a preset JSON file"
    )
    preset_group.add_argument(
        "--save-preset", type=str, help="Save current parameters to a preset JSON file"
    )

    # Extras
    parser.add_argument(
        "--render",
        type=str,
        default=None,
        help="Render offscreen PNG image to specified filepath",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print statistics about the generated mesh",
    )

    return parser.parse_args()


def main():
    """Main function for the CLI application."""
    args = parse_args()

    try:
        generator = GyroidGenerator()

        # Check if converting an STL file
        if args.input_stl:
            print(f"Loading input STL: {args.input_stl}...")
            stl_wall = args.wall_thickness if args.wall_thickness is not None else 0.8
            stl_params = STLConversionParams(
                stl_path=args.input_stl,
                resolution=args.res,
                cell_size=args.cell_size,
                wall_thickness=stl_wall,
                flat_cap=args.flat_cap,
                preserve_shell=args.preserve_shell,
                shell_thickness=args.shell_thickness,
                is_skeletal=args.skeletal,
            )
            print("Converting STL volume to Gyroid lattice...")
            generator.convert_from_stl(stl_params)

        else:
            # Standard parametric gyroid generation
            params_dict = {}

            if args.preset:
                presets = GyroidGenerator.get_presets()
                if args.preset in presets:
                    params_dict.update(presets[args.preset])
                else:
                    print(
                        f"Error: Preset '{args.preset}' not found. Available: {', '.join(presets.keys())}"
                    )
                    return 1

            if args.load_preset:
                try:
                    with open(args.load_preset, "r") as f:
                        params_dict.update(json.load(f))
                except Exception as e:
                    print(f"Error loading preset file: {e}")
                    return 1

            # Override with explicit flags
            param_wall = (
                args.wall_thickness if args.wall_thickness is not None else 11.5
            )
            cli_overrides = {
                "res": args.res,
                "a": args.a,
                "b": args.b,
                "c": args.c,
                "r1": args.r1,
                "r2": args.r2,
                "phi_scale": args.phi_scale,
                "wall_thickness": param_wall,
                "cell_radius": args.cell_radius,
                "cell_height": args.cell_height,
            }
            params_dict.update(cli_overrides)
            generator.params = GyroidParams.from_dict(params_dict)

            # Save preset if requested
            if args.save_preset:
                with open(args.save_preset, "w") as f:
                    json.dump(generator.params.to_dict(), f, indent=4)
                print(f"Preset saved to {args.save_preset}")

            print("Generating parametric gyroid structure...")
            generator.generate()

        # Auto-pierce drain holes if configured
        if getattr(args, "auto_drain_holes", False) and generator.mesh is not None:
            print("Auto-piercing drainage and vent holes...")
            generator.auto_pierce_drain_holes(
                radius=args.drain_hole_diameter / 2.0,
                depth=args.drain_hole_depth,
                count=2,
            )

        # Save output mesh
        ext = os.path.splitext(args.output)[1].lower()
        match ext:
            case ".obj":
                generator.save_obj(args.output)
            case _:
                flat_cap = args.flat_cap if args.input_stl else True
                generator.save_stl(
                    args.output, prepare_manifold=True, flat_cap=flat_cap
                )
        print(f"Mesh saved successfully to {args.output}")

        # Offscreen render thumbnail if requested
        if args.render:
            generator.save_screenshot(args.render)
            print(f"Offscreen render saved to {args.render}")

        # Print statistics
        if args.print_stats:
            stats = generator.get_mesh_stats()
            print("\nMesh Statistics:")
            print(f"Points: {stats['n_points']:,}")
            print(f"Cells: {stats['n_cells']:,}")
            print(f"Volume: {stats['volume']:.2f} cubic units")
            print(f"Surface Area: {stats['surface_area']:.2f} square units")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
