from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CATEGORIES = (
    "mutation_site",
    "3d_contact",
    "sequence_local",
    "hpd_motif",
    "other_residues",
    "unresolved_diffuse",
)


@dataclass(frozen=True)
class LatentRole:
    latent_id: str
    category: str
    mass_fraction: dict[str, float]
    supporting_residues: list[int]
    diffuseness: float
    citation: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
