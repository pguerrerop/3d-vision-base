from __future__ import annotations

from typing import Final

SUPERCLASS_BALL_GOOD: Final[str] = "BALL_GOOD"
SUPERCLASS_BALL_SCRAP: Final[str] = "BALL_SCRAP"
SUPERCLASS_SCRAP: Final[str] = "SCRAP"
SUPERCLASS_UNKNOWN: Final[str] = "UNKNOWN"

_LABEL_TO_SUPERCLASS: dict[str, str] = {
    "bola buena": SUPERCLASS_BALL_GOOD,
    "bola con chip": SUPERCLASS_BALL_SCRAP,
    "bola partida": SUPERCLASS_BALL_SCRAP,
    "bola deformada": SUPERCLASS_BALL_SCRAP,
    "scrap de bola": SUPERCLASS_BALL_SCRAP,
    "perno": SUPERCLASS_SCRAP,
    "pernos": SUPERCLASS_SCRAP,
    "planchuela": SUPERCLASS_SCRAP,
    "planchuelas": SUPERCLASS_SCRAP,
    "malla": SUPERCLASS_SCRAP,
    "mallas": SUPERCLASS_SCRAP,
    "chatarra": SUPERCLASS_SCRAP,
}


def map_label_to_superclass(label: str | None) -> str:
    normalized = str(label or "").strip().lower()
    if not normalized:
        return SUPERCLASS_UNKNOWN
    for key, superclass in _LABEL_TO_SUPERCLASS.items():
        if key in normalized:
            return superclass
    return SUPERCLASS_UNKNOWN
