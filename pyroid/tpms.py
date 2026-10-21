"""Triply Periodic Minimal Surface (TPMS) mathematical kernels and registry.

Defines supported TPMS geometries, nodal level-set equations, gradient scale factors,
and manufacturing characteristics for both sheet and skeletal scaffolds.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

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
        aliases = {
            "d": cls.DIAMOND,
            "schwarz_d": cls.DIAMOND,
            "p": cls.PRIMITIVE,
            "schwarz_p": cls.PRIMITIVE,
            "iwp": cls.I_WP,
            "schoen_i_wp": cls.I_WP,
            "fks": cls.FISCHER_KOCH_S,
            "fischer_koch": cls.FISCHER_KOCH_S,
            "fischerkoch": cls.FISCHER_KOCH_S,
            "neo": cls.NEOVIUS,
            "c_p": cls.NEOVIUS,
        }
        if cleaned in aliases:
            return aliases[cleaned]
        raise ValueError(f"Unknown TPMS type: {val}. Supported: {[m.value for m in cls]}")


@dataclass(slots=True, frozen=True)
class TPMSKernel:
    """Mathematical definition and physical properties of a TPMS geometry."""

    tpms_type: TPMSType
    display_name: str
    thickness_multiplier: float
    surface_factor: float
    description: str
    _eval: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    _eval_radial: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None

    def eval(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Evaluate 3D scalar field at given coordinates (in radians)."""
        return self._eval(X, Y, Z)

    def eval_radial(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Evaluate radial scalar field in cylindrical transformed space."""
        return self._eval_radial(X, Y, Z) if self._eval_radial else self._eval(X, Y, Z)


# Registry of TPMS kernels with nodal level-set equations
_KERNELS: dict[TPMSType, TPMSKernel] = {
    TPMSType.GYROID: TPMSKernel(
        tpms_type=TPMSType.GYROID,
        display_name="Gyroid (Schoen G)",
        thickness_multiplier=4.80,
        surface_factor=318.3,
        description="Optimal balance of self-supporting printability, isotropic modulus, and drainability.",
        _eval=lambda X, Y, Z: np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X),
        _eval_radial=lambda X, Y, Z: (
            np.cos(X) * np.sin(Y) + np.cos(Y) * np.sin(Z) + np.cos(Z) * np.sin(X)
        ),
    ),
    TPMSType.DIAMOND: TPMSKernel(
        tpms_type=TPMSType.DIAMOND,
        display_name="Diamond (Schwarz D)",
        thickness_multiplier=3.32,
        surface_factor=384.0,
        description="Highest specific stiffness and energy absorption; stretching-dominated sheet deformation.",
        _eval=lambda X, Y, Z: np.cos(X) * np.cos(Y) * np.cos(Z) - np.sin(X) * np.sin(Y) * np.sin(Z),
    ),
    TPMSType.PRIMITIVE: TPMSKernel(
        tpms_type=TPMSType.PRIMITIVE,
        display_name="Primitive (Schwarz P)",
        thickness_multiplier=4.12,
        surface_factor=235.0,
        description="Simple cubic topology with open axial channels; high axial stiffness, easy flow through axes.",
        _eval=lambda X, Y, Z: np.cos(X) + np.cos(Y) + np.cos(Z),
    ),
    TPMSType.I_WP: TPMSKernel(
        tpms_type=TPMSType.I_WP,
        display_name="I-WP (Schoen I-WP)",
        thickness_multiplier=13.50,
        surface_factor=358.0,
        description="BCC interlocking package with very high crushing plateau and multi-directional strength.",
        _eval=lambda X, Y, Z: (
            2.0 * (np.cos(X) * np.cos(Y) + np.cos(Y) * np.cos(Z) + np.cos(Z) * np.cos(X))
            - (np.cos(2.0 * X) + np.cos(2.0 * Y) + np.cos(2.0 * Z))
        ),
    ),
    TPMSType.FISCHER_KOCH_S: TPMSKernel(
        tpms_type=TPMSType.FISCHER_KOCH_S,
        display_name="Fischer-Koch S",
        thickness_multiplier=5.52,
        surface_factor=396.0,
        description="Chiral labyrinth with twisted channels; high tortuosity and compact surface area.",
        _eval=lambda X, Y, Z: (
            np.cos(2.0 * X) * np.sin(Y) * np.cos(Z)
            + np.cos(X) * np.cos(2.0 * Y) * np.sin(Z)
            + np.sin(X) * np.cos(Y) * np.cos(2.0 * Z)
        ),
    ),
    TPMSType.NEOVIUS: TPMSKernel(
        tpms_type=TPMSType.NEOVIUS,
        display_name="Neovius (C(P))",
        thickness_multiplier=8.03,
        surface_factor=512.0,
        description="14-face polyhedra with ultra-high surface area; best for catalytic carriers and micro-exchangers.",
        _eval=lambda X, Y, Z: (
            3.0 * (np.cos(X) + np.cos(Y) + np.cos(Z)) + 4.0 * (np.cos(X) * np.cos(Y) * np.cos(Z))
        ),
    ),
}


def get_tpms_kernel(tpms: str | TPMSType | None) -> TPMSKernel:
    """Retrieve TPMSKernel instance by type or string name."""
    if isinstance(tpms, TPMSType):
        return _KERNELS[tpms]
    return _KERNELS[TPMSType.from_str(tpms)]


def get_all_tpms_kernels() -> list[TPMSKernel]:
    """Return all registered TPMS kernels."""
    return list(_KERNELS.values())
