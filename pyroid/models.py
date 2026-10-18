"""Data models and configuration schemas for PyRoid.

Defines strongly-typed dataclasses for parametric gyroid generation (GyroidParams)
and volumetric STL lattice infill conversion (STLConversionParams).
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Self


@dataclass(slots=True)
class GyroidParams:
    """Parameters for generating radially symmetrical gyroid structures."""

    res: int = field(default=80, metadata={"help": "Grid resolution in each dimension"})
    a: float = field(default=24.0, metadata={"help": "Dimension length along X-axis"})
    b: float = field(default=24.0, metadata={"help": "Dimension length along Y-axis"})
    c: float = field(default=10.0, metadata={"help": "Dimension length along Z-axis"})
    r1: float = field(
        default=12.0, metadata={"help": "Inner radius (central void size)"}
    )
    r2: float = field(default=0.0, metadata={"help": "Outer radius"})
    phi_scale: float = field(
        default=8.0, metadata={"help": "Angular coordinate scaling factor (twists)"}
    )
    wall_thickness: float = field(
        default=11.5, metadata={"help": "Wall thickness threshold"}
    )
    cell_radius: float = field(
        default=2.0, metadata={"help": "Gyroid cell pore radius"}
    )
    cell_height: float = field(
        default=3.0, metadata={"help": "Gyroid cell vertical height"}
    )
    bias: float = field(
        default=0.0,
        metadata={"help": "Cavity A/B volume balance offset (-1.0 to 1.0)"},
    )

    def validate(self) -> tuple[bool, str]:
        """Validate parameter values for mathematical correctness."""
        if not isinstance(self.res, int) or self.res <= 0:
            return False, "Resolution must be a positive integer"
        if self.a <= 0 or self.b <= 0 or self.c <= 0:
            return False, "Dimensions (a, b, c) must be positive values"
        if self.r1 < 0:
            return False, "Inner radius (r1) must be non-negative"
        if self.wall_thickness <= 0:
            return False, "Wall thickness must be positive"
        if self.cell_radius <= 0 or self.cell_height <= 0:
            return False, "Cell radius and height must be positive"
        if not (-1.4 <= self.bias <= 1.4):
            return False, "Bias must be between -1.4 and 1.4"
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**{k: (int(v) if k == "res" else float(v)) for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class STLConversionParams:
    """Parameters for converting an external STL file's volume into a gyroid lattice."""

    stl_path: str = field(metadata={"help": "Path to input STL file"})
    resolution: int = field(
        default=80, metadata={"help": "Voxel grid resolution along max dimension"}
    )
    cell_size: float = field(
        default=5.0, metadata={"help": "Gyroid unit cell size X (mm or model units)"}
    )
    cell_size_y: float = field(
        default=0.0,
        metadata={"help": "Gyroid unit cell size Y (0 = same as cell_size)"},
    )
    cell_size_z: float = field(
        default=0.0,
        metadata={"help": "Gyroid unit cell size Z (0 = same as cell_size)"},
    )
    wall_thickness: float = field(
        default=0.8,
        metadata={"help": "Thickness of the gyroid lattice walls (in model units/mm)"},
    )
    bias: float = field(
        default=0.0,
        metadata={"help": "Cavity A/B volume balance offset (-1.0 to 1.0)"},
    )
    flat_cap: bool = field(
        default=False,
        metadata={"help": "Trim and cap top/bottom surfaces for 3D printing"},
    )
    preserve_shell: bool = field(
        default=False,
        metadata={"help": "Preserve original STL boundary as a solid shell"},
    )
    shell_thickness: float = field(
        default=1.0,
        metadata={"help": "Thickness of the preserved shell (in model units/mm)"},
    )
    is_skeletal: bool = field(
        default=False,
        metadata={
            "help": "Generate skeletal strut-based lattice (single continuous void) instead of sheet walls"
        },
    )

    def validate(self) -> tuple[bool, str]:
        if not self.stl_path:
            return False, "Input STL path must be specified"
        if self.resolution <= 10:
            return False, "Resolution must be at least 10"
        if self.cell_size <= 0 or self.cell_size_y < 0 or self.cell_size_z < 0:
            return False, "Cell size must be positive"
        if self.wall_thickness <= 0:
            return False, "Wall thickness must be positive"
        if not (-1.4 <= self.bias <= 1.4):
            return False, "Bias must be between -1.4 and 1.4"
        if self.preserve_shell and self.shell_thickness <= 0:
            return (
                False,
                "Shell thickness must be positive when shell preservation is enabled",
            )
        return True, ""
