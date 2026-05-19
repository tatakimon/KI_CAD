from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_xywh(self) -> list[int]:
        return [self.x1, self.y1, self.width, self.height]

    def translated(self, dx: int, dy: int) -> "Box":
        return Box(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: Box
    source: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["box"] = {
            "x1": self.box.x1,
            "y1": self.box.y1,
            "x2": self.box.x2,
            "y2": self.box.y2,
            "xywh": self.box.to_xywh(),
        }
        return data


@dataclass(frozen=True)
class Tile:
    id: str
    file: str
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
