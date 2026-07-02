"""Parser for SWC neuron morphology files (NeuroMorpho.org format).

The SWC format describes a neuron as a rooted tree of sample points. Each
non-comment line holds seven whitespace-separated fields::

    n  T  x  y  z  radius  parent

where ``n`` is a 1-based sample id, ``T`` is a structure type code
(1=soma, 2=axon, 3=basal dendrite, 4=apical dendrite, ...), ``x y z`` are
coordinates in micrometres, ``radius`` is the sample radius in micrometres,
and ``parent`` is the id of the parent sample (or ``-1`` for a root).

This module only parses geometry; turning a morphology into a simulate-able
cable model is handled in :mod:`src.simulation.cable`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Human-readable names for the standard SWC structure type codes.
SWC_TYPE_NAMES: dict[int, str] = {
    0: "undefined",
    1: "soma",
    2: "axon",
    3: "basal dendrite",
    4: "apical dendrite",
}


@dataclass(frozen=True)
class SWCNode:
    """A single SWC sample point.

    Attributes:
        id: 1-based sample identifier.
        type: SWC structure type code (see :data:`SWC_TYPE_NAMES`).
        x, y, z: Sample coordinates in micrometres.
        radius: Sample radius in micrometres.
        parent: Parent sample id, or -1 for a root sample.
    """

    id: int
    type: int
    x: float
    y: float
    z: float
    radius: float
    parent: int


@dataclass(frozen=True)
class Morphology:
    """A parsed SWC morphology as an ordered collection of samples."""

    nodes: tuple[SWCNode, ...]

    def __post_init__(self) -> None:
        """Validate identifiers, parent references, and radii."""

        if not self.nodes:
            raise ValueError("Morphology must contain at least one sample.")
        ids = [node.id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("SWC sample ids must be unique.")
        id_set = set(ids)
        for node in self.nodes:
            if node.parent != -1 and node.parent not in id_set:
                raise ValueError(
                    f"SWC sample {node.id} references missing parent {node.parent}."
                )
            if node.parent == node.id:
                raise ValueError(f"SWC sample {node.id} lists itself as its parent.")
            if node.radius <= 0:
                raise ValueError(f"SWC sample {node.id} has non-positive radius.")

    @property
    def node_by_id(self) -> dict[int, SWCNode]:
        """Return a mapping from sample id to node."""

        return {node.id: node for node in self.nodes}

    def roots(self) -> tuple[SWCNode, ...]:
        """Return all root samples (parent == -1)."""

        return tuple(node for node in self.nodes if node.parent == -1)

    def type_counts(self) -> dict[str, int]:
        """Return a count of samples per structure-type name."""

        counts: dict[str, int] = {}
        for node in self.nodes:
            name = SWC_TYPE_NAMES.get(node.type, f"type {node.type}")
            counts[name] = counts.get(name, 0) + 1
        return counts


def parse_swc(text: str) -> Morphology:
    """Parse SWC-formatted text into a :class:`Morphology`.

    Blank lines and comment lines (starting with ``#``) are ignored. Each
    remaining line must contain the seven standard SWC fields.
    """

    nodes: list[SWCNode] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 7:
            raise ValueError(
                f"SWC line {line_number} has {len(fields)} fields, expected 7: {raw_line!r}"
            )
        try:
            node = SWCNode(
                id=int(fields[0]),
                type=int(float(fields[1])),
                x=float(fields[2]),
                y=float(fields[3]),
                z=float(fields[4]),
                radius=float(fields[5]),
                parent=int(float(fields[6])),
            )
        except ValueError as error:
            raise ValueError(f"SWC line {line_number} is malformed: {raw_line!r}") from error
        nodes.append(node)

    if not nodes:
        raise ValueError("No SWC samples found (file was empty or all comments).")
    return Morphology(nodes=tuple(nodes))


def load_swc(path: str | Path) -> Morphology:
    """Load and parse an SWC morphology file from disk."""

    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    return parse_swc(text)
