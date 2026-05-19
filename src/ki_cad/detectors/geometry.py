from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ki_cad.core.models import Box, Detection
from ki_cad.core.nms import non_max_suppression


def _edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(gray, 50, 150)


def _scale_template(template: np.ndarray, scale: float) -> np.ndarray | None:
    height, width = template.shape[:2]
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    if scaled_width < 8 or scaled_height < 8:
        return None
    return cv2.resize(template, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)


def detect_by_geometry(
    image: np.ndarray,
    symbol_path: Path,
    label: str,
    threshold: float,
    scales: list[float],
    nms_iou: float,
    max_detections: int,
) -> list[Detection]:
    symbol = cv2.imread(str(symbol_path), cv2.IMREAD_COLOR)
    if symbol is None:
        raise ValueError(f"Could not read symbol crop: {symbol_path}")

    image_edges = _edges(image)
    symbol_edges = _edges(symbol)
    raw: list[Detection] = []

    for scale in scales:
        template = _scale_template(symbol_edges, scale)
        if template is None:
            continue
        th, tw = template.shape[:2]
        if th > image_edges.shape[0] or tw > image_edges.shape[1]:
            continue

        result = cv2.matchTemplate(image_edges, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= threshold)
        for x, y in zip(xs.tolist(), ys.tolist()):
            raw.append(
                Detection(
                    label=label,
                    score=float(result[y, x]),
                    box=Box(x, y, x + tw, y + th),
                    source=f"geometry_template_scale_{scale:.2f}",
                )
            )

    kept = non_max_suppression(raw, iou_threshold=nms_iou)
    return kept[:max_detections]
