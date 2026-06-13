from __future__ import annotations

from .models import Detection


def iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.box.x1, a.box.y1, a.box.x2, a.box.y2
    bx1, by1, bx2, by2 = b.box.x1, b.box.y1, b.box.x2, b.box.y2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def non_max_suppression(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    ordered = sorted(detections, key=lambda item: item.score, reverse=True)
    kept: list[Detection] = []

    for candidate in ordered:
        if all(iou(candidate, existing) < iou_threshold for existing in kept):
            kept.append(candidate)

    return kept


def non_max_suppression_by_label(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    grouped: dict[str, list[Detection]] = {}
    for detection in detections:
        grouped.setdefault(detection.label, []).append(detection)

    kept: list[Detection] = []
    for label_detections in grouped.values():
        kept.extend(non_max_suppression(label_detections, iou_threshold=iou_threshold))

    return sorted(kept, key=lambda item: item.score, reverse=True)
