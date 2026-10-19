"""Data models and configuration schemas for PyRoid.

Defines strongly-typed dataclasses for parametric gyroid generation (GyroidParams)
and volumetric STL lattice infill conversion (STLConversionParams).
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Self


@dataclass(slots=True)
class GyroidParams:
    """Parameters for generating radially symmetrical gyroid structures."""

    res: int = field(default=80, metadata={"help": "Grid resolution in each dimension (higher = smoother mesh, slower generation)"})
    a: float = field(default=24.0, metadata={"help": "Radial domain length along X-axis"})
    b: float = field(default=24.0, metadata={"help": "Angular domain length along Y-axis"})
    c: float = field(default=10.0, metadata={"help": "Vertical height along Z-axis"})
    r1: float = field(
        default=12.0, metadata={"help": "Outer cylinder radius"}
    )
    r2: float = field(default=0.0, metadata={"help": "Inner core radius (0 = solid center)"})
    phi_scale: float = field(
        default=8.0, metadata={"help": "Angular coordinate scaling factor (number of radial twists)"}
    )
    wall_thickness: float = field(
        default=11.5, metadata={"help": "Radial annular ring thickness (from r1 inward)"}
    )
    cell_radius: float = field(
        default=2.0, metadata={"help": "Gyroid unit cell radial/angular pore frequency"}
    )
    cell_height: float = field(
        default=3.0, metadata={"help": "Gyroid unit cell vertical pore frequency"}
    )
    bias: float = field(
        default=0.0,
        metadata={"help": "Asymmetric cavity balance (-1.0 to 1.0): >0 widens Cavity A (Red), <0 widens Cavity B (Blue)"},
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
        default=5.0, metadata={"help": "Gyroid unit cell size (mm or model units)"}
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
    infill_percent: float | None = field(
        default=None,
        metadata={"help": "Target solid infill volume percentage (1 to 99)"},
    )

    @staticmethod
    def estimate_infill(
        cell_size: float, wall_thickness: float, is_skeletal: bool = False
    ) -> float:
        """Estimate solid infill percentage from cell size and wall thickness."""
        if cell_size <= 0 or wall_thickness <= 0:
            return 0.0
        ratio = wall_thickness / cell_size
        if is_skeletal:
            return float(min(95.0, max(5.0, 50.0 + 95.5 * (ratio - 0.2))))
        return float(min(95.0, max(5.0, 318.3 * ratio)))

    @staticmethod
    def estimate_cell_size(
        infill_percent: float, wall_thickness: float, is_skeletal: bool = False
    ) -> float:
        """Estimate required cell size (mm) to achieve target infill % at given wall thickness."""
        pct = min(95.0, max(5.0, infill_percent))
        if wall_thickness <= 0:
            return 5.0
        if is_skeletal:
            ratio = max(0.05, 0.2 + (pct - 50.0) / 95.5)
            return float(wall_thickness / ratio)
        return float(318.3 * wall_thickness / pct)

    def validate(self) -> tuple[bool, str]:
        if not self.stl_path:
            return False, "Input STL path must be specified"
        if self.resolution <= 10:
            return False, "Resolution must be at least 10"
        if self.cell_size <= 0:
            return False, "Cell size must be positive"
        if self.wall_thickness <= 0:
            return False, "Wall thickness must be positive"
        if not (-1.4 <= self.bias <= 1.4):
            return False, "Bias must be between -1.4 and 1.4"
        if self.infill_percent is not None and not (0.0 < self.infill_percent < 100.0):
            return False, "Infill percent must be between 0 and 100"
        if self.preserve_shell and self.shell_thickness <= 0:
            return (
                False,
                "Shell thickness must be positive when shell preservation is enabled",
            )
        return True, ""
