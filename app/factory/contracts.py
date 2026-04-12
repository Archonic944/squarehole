"""Contract definitions — shape packs the player can accept.

Contracts are the player's progression system. The starter pack is
accepted automatically on a new factory; additional packs are accepted
explicitly from the Contracts UI, which adds that pack's shape
categories to the active pool.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Contract:
    id: str
    name: str
    description: str
    categories: tuple[str, ...]
    # Contracts can be gated behind a coin cost later; 0 for now.
    cost: float = 0.0


STARTER = Contract(
    id="starter",
    name="Starter Pack",
    description="Basic shapes to learn the ropes.",
    categories=("circle", "triangle", "square", "star_5", "heart"),
)

PACK1_SILHOUETTES = Contract(
    id="silhouettes",
    name="Tricky Silhouettes",
    description="Distinctive outlines: arrow, crescent, cloud, lightning, teardrop.",
    categories=("arrow", "crescent", "cloud", "lightning", "teardrop"),
)

PACK2_HOLES = Contract(
    id="holes",
    name="Holes & Cutouts",
    description="Shapes with interior holes: donut, picture frame, key, gear.",
    categories=("donut", "picture_frame", "key", "gear"),
)

PACK3_MULTICOLOR = Contract(
    id="multicolor",
    name="Multicolor",
    description="Multicolor shapes: mushroom, tree, flower, candy cane, rainbow.",
    categories=("mushroom", "tree", "flower", "candy_cane", "rainbow"),
)


ALL_CONTRACTS: list[Contract] = [
    STARTER,
    PACK1_SILHOUETTES,
    PACK2_HOLES,
    PACK3_MULTICOLOR,
]


def get_contract(contract_id: str) -> Contract | None:
    for c in ALL_CONTRACTS:
        if c.id == contract_id:
            return c
    return None
