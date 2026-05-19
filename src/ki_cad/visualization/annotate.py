from __future__ import annotations

import cv2
import numpy as np

from ki_cad.core.models import Detection


GEOMETRY_COLOR = (0, 180, 255)
VLM_COLOR = (255, 80, 80)


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = image.copy()
    for det in detections:
        color = VLM_COLOR if det.source.startswith("vlm") else GEOMETRY_COLOR
        cv2.rectangle(annotated, (det.box.x1, det.box.y1), (det.box.x2, det.box.y2), color, 2)
        text = f"{det.label} {det.score:.2f}"
        cv2.putText(
            annotated,
            text,
            (det.box.x1, max(18, det.box.y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated
