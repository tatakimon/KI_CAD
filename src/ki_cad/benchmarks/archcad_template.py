from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ki_cad.datasets.archcad import export_instance_boxes, extract_sample, find_sample_ids_with_semantic
from ki_cad.evaluation import evaluate_detections, load_detections
from ki_cad.pipeline.detect import DetectConfig, run_detection


@dataclass(frozen=True)
class ArchCadTemplateBenchmarkConfig:
    raw_dir: Path
    symbol_path: Path
    out_dir: Path
    semantic: int
    label: str
    count: int = 20
    dpi: int = 144
    threshold: float = 0.58
    scales: tuple[float, ...] = (0.9, 1.0, 1.1)
    nms_iou: float = 0.25
    eval_iou: float = 0.5
    max_detections: int = 300
    padding: int = 4


def run_archcad_template_benchmark(config: ArchCadTemplateBenchmarkConfig) -> dict[str, object]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = find_sample_ids_with_semantic(
        raw_dir=config.raw_dir,
        semantic=config.semantic,
        limit=config.count,
    )

    per_sample: list[dict[str, object]] = []
    totals = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "predictions": 0,
        "truth": 0,
    }

    for sample_id in sample_ids:
        sample_dir = config.out_dir / "samples" / sample_id
        run_dir = config.out_dir / "runs" / sample_id
        truth_path = sample_dir / f"semantic_{config.semantic}_truth.json"
        truth_preview = sample_dir / f"semantic_{config.semantic}_truth.png"

        sample = extract_sample(
            raw_dir=config.raw_dir,
            out_dir=sample_dir,
            sample_id=sample_id,
            dpi=config.dpi,
        )
        if sample.json_path is None:
            raise FileNotFoundError(f"No annotation JSON extracted for {sample_id}")

        truth = export_instance_boxes(
            json_path=sample.json_path,
            svg_path=sample.svg_path,
            semantic=config.semantic,
            out_json=truth_path,
            dpi=config.dpi,
            preview_path=truth_preview,
            label=config.label,
            padding=config.padding,
        )

        run_detection(
            DetectConfig(
                input_path=sample.svg_path,
                symbol_path=config.symbol_path,
                out_dir=run_dir,
                label=config.label,
                dpi=config.dpi,
                threshold=config.threshold,
                scales=config.scales,
                nms_iou=config.nms_iou,
                max_detections=config.max_detections,
            )
        )

        predictions_path = run_dir / "final_detections.json"
        predictions = load_detections(predictions_path)
        result = evaluate_detections(predictions, truth, iou_threshold=config.eval_iou)
        sample_result = {
            "sample_id": sample_id,
            "predictions": len(predictions),
            "truth": len(truth),
            **result.to_dict(),
            "annotated": str(run_dir / "annotated.png"),
            "truth_preview": str(truth_preview),
        }
        per_sample.append(sample_result)

        totals["true_positives"] += result.true_positives
        totals["false_positives"] += result.false_positives
        totals["false_negatives"] += result.false_negatives
        totals["predictions"] += len(predictions)
        totals["truth"] += len(truth)

    tp = totals["true_positives"]
    fp = totals["false_positives"]
    fn = totals["false_negatives"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    summary = {
        "semantic": config.semantic,
        "label": config.label,
        "count_requested": config.count,
        "count_evaluated": len(per_sample),
        "symbol_path": str(config.symbol_path),
        "threshold": config.threshold,
        "scales": list(config.scales),
        "nms_iou": config.nms_iou,
        "eval_iou": config.eval_iou,
        "totals": {
            **totals,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "samples": per_sample,
    }

    (config.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
