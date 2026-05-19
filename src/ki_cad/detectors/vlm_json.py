from __future__ import annotations

import json
from pathlib import Path

from ki_cad.core.models import Box, Detection


def load_vlm_or_manual_detections(path: Path | None, label: str) -> list[Detection]:
    """Load future VLM/manual detections for comparison with geometry results.

    Expected format:
    [{"label": "valve", "score": 0.8, "box": [x1, y1, x2, y2]}]
    """
    if path is None:
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    detections: list[Detection] = []
    for item in data:
        box = item["box"]
        detections.append(
            Detection(
                label=item.get("label", label),
                score=float(item.get("score", 1.0)),
                box=Box(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                source="vlm_or_manual_json",
            )
        )
    return detections
