from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ki_cad.core.models import Box, Detection
from ki_cad.core.nms import iou


@dataclass(frozen=True)
class EvaluationResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_iou: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_iou": self.mean_iou,
        }


def _box_from_value(value: object) -> Box:
    if isinstance(value, list):
        if len(value) != 4:
            raise ValueError("List box must have four values: [x1, y1, x2, y2]")
        return Box(int(value[0]), int(value[1]), int(value[2]), int(value[3]))
    if isinstance(value, dict):
        if "x1" in value:
            return Box(int(value["x1"]), int(value["y1"]), int(value["x2"]), int(value["y2"]))
        if "xywh" in value:
            x, y, width, height = value["xywh"]
            return Box(int(x), int(y), int(x + width), int(y + height))
    raise ValueError(f"Unsupported box format: {value}")


def load_detections(path: Path, default_label: str = "target") -> list[Detection]:
    data = json.loads(path.read_text(encoding="utf-8"))
    detections: list[Detection] = []
    for index, item in enumerate(data):
        detections.append(
            Detection(
                label=item.get("label", default_label),
                score=float(item.get("score", 1.0)),
                box=_box_from_value(item["box"]),
                source=item.get("source", f"loaded_{index}"),
            )
        )
    return detections


def evaluate_detections(
    predictions: list[Detection],
    truth: list[Detection],
    iou_threshold: float,
) -> EvaluationResult:
    ordered_predictions = sorted(predictions, key=lambda item: item.score, reverse=True)
    unmatched_truth = set(range(len(truth)))
    matched_ious: list[float] = []
    true_positives = 0
    false_positives = 0

    for prediction in ordered_predictions:
        best_index: int | None = None
        best_iou = 0.0
        for truth_index in unmatched_truth:
            candidate_iou = iou(prediction, truth[truth_index])
            if candidate_iou > best_iou:
                best_iou = candidate_iou
                best_index = truth_index

        if best_index is not None and best_iou >= iou_threshold:
            unmatched_truth.remove(best_index)
            matched_ious.append(best_iou)
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(unmatched_truth)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    return EvaluationResult(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=mean_iou,
    )
