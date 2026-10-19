"""Triply Periodic Minimal Surface (TPMS) mathematical kernels and registry.

Defines supported TPMS geometries, nodal level-set equations, gradient scale factors,
and manufacturing characteristics for both sheet and skeletal scaffolds.
"""
from enum import Enum
from typing import Any
import numpy as np


class TPMSType(str, Enum):
    """Supported TPMS geometry types."""

    GYROID = "gyroid"
    DIAMOND = "diamond"
    PRIMITIVE = "primitive"
    I_WP = "i_wp"
    FISCHER_KOCH_S = "fischer_koch_s"
    NEOVIUS = "neovius"

    @classmethod
    def from_str(cls, val: str | None) -> "TPMSType":
        """Parse string to TPMSType with tolerant fallback."""
        if not val:
            return cls.GYROID
        cleaned = val.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == cleaned or member.name.lower() == cleaned:
                return member
        # Fallback aliases
        if cleaned in ("d", "schwarz_d"):
            return cls.DIAMOND
        if cleaned in ("p", "schwarz_p"):
            return cls.PRIMITIVE
        if cleaned in ("iwp", "schoen_i_wp"):
            return cls.I_WP
        if cleaned in ("fks", "fischer_koch", "fischerkoch"):
            return cls.FISCHER_KOCH_S
        if cleaned in ("neo", "c_p"):
            return cls.NEOVIUS
        raise ValueError(f"Unknown TPMS type: {val}. Supported: {[m.value for m in cls]}")


class TPMSKernel:
    """Base class for TPMS mathematical kernels."""

    def __init__(
        self,
        tpms_type: TPMSType,
        display_name: str,
        thickness_multiplier: float,
        description: str,
        category: str = "General",
    ):
        self.tpms_type = tpms_type
        self.display_name = display_name
        self.thickness_multiplier = thickness_multiplier
        self.description = description
        self.category = category

    def eval(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Evaluate 3D scalar field at given coordinates (in radians)."""
        raise NotImplementedError

    def eval_radial(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Evaluate radial scalar field in cylindrical transformed space."""
        return self.eval(X, Y, Z)


class GyroidKernel(TPMSKernel):
    """Schoen Gyroid (Ia3d space group).

    Isotropic, continuous self-supporting curvature, excellent for fluid exchange and general AM.
    """

    def __init__(self):
        super().__init__(
            tpms_type=TPMSType.GYROID,
            display_name="Gyroid (Schoen G)",
            thickness_multiplier=4.80,
            description="Optimal balance of self-supporting printability, isotropic modulus, and drainability.",
            category="Balanced / Heat Exchange / Bio",
        )

    def eval(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)

    def eval_radial(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return np.cos(X) * np.sin(Y) + np.cos(Y) * np.sin(Z) + np.cos(Z) * np.sin(X)


class DiamondKernel(TPMSKernel):
    """Schwarz Diamond (Fd3m space group).

    Tetrahedral node coordination, maximum specific stiffness and energy absorption.
    """

    def __init__(self):
        super().__init__(
            tpms_type=TPMSType.DIAMOND,
            display_name="Diamond (Schwarz D)",
            thickness_multiplier=3.32,
            description="Highest specific stiffness and energy absorption; stretching-dominated sheet deformation.",
            category="High Strength / Structural",
        )

    def eval(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return np.cos(X) * np.cos(Y) * np.cos(Z) - np.sin(X) * np.sin(Y) * np.sin(Z)


class PrimitiveKernel(TPMSKernel):
    """Schwarz Primitive (Pm3m space group).

    Simple cubic symmetry with large axial open pore channels.
    """

    def __init__(self):
        super().__init__(
            tpms_type=TPMSType.PRIMITIVE,
            display_name="Primitive (Schwarz P)",
            thickness_multiplier=4.12,
            description="Simple cubic topology with open axial channels; high axial stiffness, easy flow through axes.",
            category="Porous Transport / Filtration",
        )

    def eval(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return np.cos(X) + np.cos(Y) + np.cos(Z)


class IWPKernel(TPMSKernel):
    """Schoen I-WP (Im3m space group).

    Body-centered cubic and wrapped package, exceptional compressive crushing resistance.
    """

    def __init__(self):
        super().__init__(
            tpms_type=TPMSType.I_WP,
            display_name="I-WP (Schoen I-WP)",
            thickness_multiplier=13.50,
            description="BCC interlocking package with very high crushing plateau and multi-directional strength.",
            category="Energy Absorption / Crash",
        )

    def eval(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return 2.0 * (
            np.cos(X) * np.cos(Y) + np.cos(Y) * np.cos(Z) + np.cos(Z) * np.cos(X)
        ) - (np.cos(2.0 * X) + np.cos(2.0 * Y) + np.cos(2.0 * Z))


class FischerKochSKernel(TPMSKernel):
    """Fischer-Koch S (I41 32 space group).

    Chiral non-centrosymmetric labyrinth with high tortuosity and high specific surface area.
    """

    def __init__(self):
        super().__init__(
            tpms_type=TPMSType.FISCHER_KOCH_S,
            display_name="Fischer-Koch S",
            thickness_multiplier=5.52,
            description="Chiral labyrinth with twisted channels; high tortuosity and compact surface area.",
            category="Catalysis / Tortuous Transport",
        )

    def eval(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return (
            np.cos(2.0 * X) * np.sin(Y) * np.cos(Z)
            + np.cos(X) * np.cos(2.0 * Y) * np.sin(Z)
            + np.sin(X) * np.cos(Y) * np.cos(2.0 * Z)
        )


class NeoviusKernel(TPMSKernel):
    """Neovius (Im3m space group).

    Truncated octahedron geometry with ultra-high specific surface area.
    """

    def __init__(self):
        super().__init__(
            tpms_type=TPMSType.NEOVIUS,
            display_name="Neovius (C(P))",
            thickness_multiplier=8.03,
            description="14-face polyhedra with ultra-high surface area; best for catalytic carriers and micro-exchangers.",
            category="Ultra-High Surface Area",
        )

    def eval(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return 3.0 * (np.cos(X) + np.cos(Y) + np.cos(Z)) + 4.0 * (
            np.cos(X) * np.cos(Y) * np.cos(Z)
        )


# Global Registry
_KERNELS: dict[TPMSType, TPMSKernel] = {
    TPMSType.GYROID: GyroidKernel(),
    TPMSType.DIAMOND: DiamondKernel(),
    TPMSType.PRIMITIVE: PrimitiveKernel(),
    TPMSType.I_WP: IWPKernel(),
    TPMSType.FISCHER_KOCH_S: FischerKochSKernel(),
    TPMSType.NEOVIUS: NeoviusKernel(),
}


def get_tpms_kernel(tpms: str | TPMSType | None) -> TPMSKernel:
    """Retrieve TPMSKernel instance by type or string name."""
    if isinstance(tpms, TPMSType):
        return _KERNELS[tpms]
    tpms_enum = TPMSType.from_str(tpms)
    return _KERNELS[tpms_enum]


def get_all_tpms_kernels() -> list[TPMSKernel]:
    """Return all registered TPMS kernels."""
    return list(_KERNELS.values())
