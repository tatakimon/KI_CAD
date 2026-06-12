from __future__ import annotations

import argparse
import json
from pathlib import Path

from ki_cad.benchmarks.archcad_template import ArchCadTemplateBenchmarkConfig, run_archcad_template_benchmark
from ki_cad.datasets.archcad import extract_sample, export_instance_boxes, export_instance_crops, summarize_annotation
from ki_cad.evaluation import evaluate_detections, load_detections
from ki_cad.pipeline.detect import DetectConfig, run_detection


def _parse_scales(value: str) -> tuple[float, ...]:
    scales = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not scales:
        raise argparse.ArgumentTypeError("Provide at least one scale")
    return scales


def detect_command(args: argparse.Namespace) -> None:
    result = run_detection(
        DetectConfig(
            input_path=args.input,
            out_dir=args.out,
            label=args.label,
            symbol_path=args.symbol,
            symbol_dir=args.symbol_dir,
            page=args.page,
            dpi=args.dpi,
            threshold=args.threshold,
            scales=args.scales,
            nms_iou=args.nms_iou,
            max_detections=args.max_detections,
            tile_size=args.tile_size,
            overlap=args.overlap,
            vlm_json=args.vlm_json,
        )
    )

    print(f"Rendered page: {args.out / 'rendered_page.png'}")
    print(f"Tiles written: {result['tiles']}")
    print(f"Templates used: {result['symbols']}")
    print(f"Geometry detections: {result['geometry']}")
    print(f"VLM/manual detections: {result['vlm']}")
    print(f"Final detections after NMS: {result['final']}")
    print(f"Annotated output: {args.out / 'annotated.png'}")


def archcad_extract_command(args: argparse.Namespace) -> None:
    sample = extract_sample(
        raw_dir=args.raw_dir,
        out_dir=args.out,
        sample_id=args.sample_id,
        dpi=args.dpi,
    )
    print(f"Sample ID: {sample.sample_id}")
    print(f"SVG: {sample.svg_path}")
    if sample.json_path:
        summary = summarize_annotation(sample.json_path)
        print(f"JSON: {sample.json_path}")
        print(f"Entities: {summary['entities']}")
        print(f"Top semantic counts: {list(summary['semantic_counts'].items())[:8]}")
    if sample.caption_path:
        print(f"Caption: {sample.caption_path}")
    print(f"Preview: {sample.preview_path}")


def archcad_export_truth_command(args: argparse.Namespace) -> None:
    detections = export_instance_boxes(
        json_path=args.json,
        svg_path=args.svg,
        semantic=args.semantic,
        out_json=args.out,
        dpi=args.dpi,
        preview_path=args.preview,
        label=args.label,
        padding=args.padding,
    )
    print(f"Semantic: {args.semantic}")
    print(f"Boxes exported: {len(detections)}")
    print(f"Truth JSON: {args.out}")
    if args.preview:
        print(f"Preview: {args.preview}")


def archcad_export_templates_command(args: argparse.Namespace) -> None:
    crops = export_instance_crops(
        json_path=args.json,
        svg_path=args.svg,
        semantic=args.semantic,
        out_dir=args.out,
        dpi=args.dpi,
        padding=args.padding,
        max_crops=args.max_crops,
    )
    print(f"Semantic: {args.semantic}")
    print(f"Templates exported: {len(crops)}")
    print(f"Template directory: {args.out}")


def archcad_benchmark_template_command(args: argparse.Namespace) -> None:
    summary = run_archcad_template_benchmark(
        ArchCadTemplateBenchmarkConfig(
            raw_dir=args.raw_dir,
            out_dir=args.out,
            semantic=args.semantic,
            label=args.label,
            symbol_path=args.symbol,
            symbol_dir=args.symbol_dir,
            count=args.count,
            dpi=args.dpi,
            threshold=args.threshold,
            scales=args.scales,
            nms_iou=args.nms_iou,
            eval_iou=args.eval_iou,
            max_detections=args.max_detections,
            padding=args.padding,
        )
    )
    totals = summary["totals"]
    print(f"Samples evaluated: {summary['count_evaluated']}")
    print(f"Predictions/truth: {totals['predictions']}/{totals['truth']}")
    print(f"TP/FP/FN: {totals['true_positives']}/{totals['false_positives']}/{totals['false_negatives']}")
    print(f"Precision: {totals['precision']:.3f}")
    print(f"Recall: {totals['recall']:.3f}")
    print(f"F1: {totals['f1']:.3f}")
    print(f"Summary: {args.out / 'summary.json'}")


def evaluate_command(args: argparse.Namespace) -> None:
    predictions = load_detections(args.predictions)
    truth = load_detections(args.truth)
    result = evaluate_detections(predictions, truth, iou_threshold=args.iou)
    result_data = result.to_dict()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result_data, indent=2), encoding="utf-8")

    print(f"Predictions: {len(predictions)}")
    print(f"Ground truth: {len(truth)}")
    print(f"IoU threshold: {args.iou:.2f}")
    print(f"TP/FP/FN: {result.true_positives}/{result.false_positives}/{result.false_negatives}")
    print(f"Precision: {result.precision:.3f}")
    print(f"Recall: {result.recall:.3f}")
    print(f"F1: {result.f1:.3f}")
    print(f"Mean matched IoU: {result.mean_iou:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ki-cad")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Find a symbol crop in a CAD PDF/image")
    detect.add_argument("--input", required=True, type=Path, help="Input PDF, SVG, or image")
    detect.add_argument("--page", type=int, default=1, help="1-based PDF page number")
    detect.add_argument("--dpi", type=int, default=200, help="PDF render DPI")
    detect.add_argument("--symbol", type=Path, default=None, help="Target symbol crop image")
    detect.add_argument("--symbol-dir", type=Path, default=None, help="Directory of target symbol crop images")
    detect.add_argument("--label", default="target_symbol", help="Detection label")
    detect.add_argument("--out", required=True, type=Path, help="Output run directory")
    detect.add_argument("--threshold", type=float, default=0.45, help="Template match threshold")
    detect.add_argument("--scales", type=_parse_scales, default=_parse_scales("0.75,0.9,1.0,1.1,1.25"))
    detect.add_argument("--nms-iou", type=float, default=0.25, help="NMS IoU threshold")
    detect.add_argument("--max-detections", type=int, default=200, help="Limit geometry detections")
    detect.add_argument("--tile-size", type=int, default=1024, help="Tile size for VLM-ready chunks")
    detect.add_argument("--overlap", type=int, default=256, help="Tile overlap in pixels")
    detect.add_argument("--vlm-json", type=Path, default=None, help="Optional VLM/manual global-box JSON")
    detect.set_defaults(func=detect_command)

    evaluate = subparsers.add_parser("evaluate", help="Score predictions against ground-truth boxes")
    evaluate.add_argument("--predictions", required=True, type=Path, help="Predicted detections JSON")
    evaluate.add_argument("--truth", required=True, type=Path, help="Ground-truth detections JSON")
    evaluate.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a successful match")
    evaluate.add_argument("--out", type=Path, default=None, help="Optional JSON report output path")
    evaluate.set_defaults(func=evaluate_command)

    archcad = subparsers.add_parser("archcad", help="ArchCAD dataset utilities")
    archcad_subparsers = archcad.add_subparsers(dest="archcad_command", required=True)

    extract = archcad_subparsers.add_parser("extract-sample", help="Extract one ArchCAD SVG/JSON/caption sample")
    extract.add_argument("--raw-dir", type=Path, default=Path("data/raw/archcad"), help="Directory with ArchCAD ZIPs")
    extract.add_argument("--sample-id", default=None, help="Sample UUID; defaults to the first SVG in svg.zip")
    extract.add_argument("--out", type=Path, default=Path("data/interim/archcad_sample"), help="Output sample directory")
    extract.add_argument("--dpi", type=int, default=144, help="Preview render DPI")
    extract.set_defaults(func=archcad_extract_command)

    export_truth = archcad_subparsers.add_parser(
        "export-truth",
        help="Export ArchCAD instance boxes for one semantic class",
    )
    export_truth.add_argument("--json", required=True, type=Path, help="ArchCAD annotation JSON")
    export_truth.add_argument("--svg", required=True, type=Path, help="Matching ArchCAD SVG")
    export_truth.add_argument("--semantic", required=True, type=int, help="Semantic class ID to export")
    export_truth.add_argument("--label", default=None, help="Output label; defaults to semantic_<id>")
    export_truth.add_argument("--dpi", type=int, default=144, help="Render DPI used by predictions")
    export_truth.add_argument("--padding", type=int, default=0, help="Pixel padding around exported boxes")
    export_truth.add_argument("--out", required=True, type=Path, help="Output truth JSON")
    export_truth.add_argument("--preview", type=Path, default=None, help="Optional annotated preview image")
    export_truth.set_defaults(func=archcad_export_truth_command)

    export_templates = archcad_subparsers.add_parser(
        "export-templates",
        help="Export template crops from ArchCAD instances for one semantic class",
    )
    export_templates.add_argument("--json", required=True, type=Path, help="ArchCAD annotation JSON")
    export_templates.add_argument("--svg", required=True, type=Path, help="Matching ArchCAD SVG")
    export_templates.add_argument("--semantic", required=True, type=int, help="Semantic class ID to crop")
    export_templates.add_argument("--dpi", type=int, default=144, help="Render DPI")
    export_templates.add_argument("--padding", type=int, default=8, help="Pixel padding around crops")
    export_templates.add_argument("--max-crops", type=int, default=None, help="Optional crop limit")
    export_templates.add_argument("--out", required=True, type=Path, help="Output template directory")
    export_templates.set_defaults(func=archcad_export_templates_command)

    benchmark = archcad_subparsers.add_parser(
        "benchmark-template",
        help="Run template matching over ArchCAD samples and score against annotations",
    )
    benchmark.add_argument("--raw-dir", type=Path, default=Path("data/raw/archcad"), help="Directory with ArchCAD ZIPs")
    benchmark.add_argument("--symbol", type=Path, default=None, help="Target symbol crop image")
    benchmark.add_argument("--symbol-dir", type=Path, default=None, help="Directory of target symbol crop images")
    benchmark.add_argument("--semantic", required=True, type=int, help="Semantic class ID to evaluate")
    benchmark.add_argument("--label", default="target_symbol", help="Output label")
    benchmark.add_argument("--count", type=int, default=20, help="Number of samples to evaluate")
    benchmark.add_argument("--out", required=True, type=Path, help="Benchmark output directory")
    benchmark.add_argument("--dpi", type=int, default=144, help="Render DPI")
    benchmark.add_argument("--threshold", type=float, default=0.58, help="Template match threshold")
    benchmark.add_argument("--scales", type=_parse_scales, default=_parse_scales("0.9,1.0,1.1"))
    benchmark.add_argument("--nms-iou", type=float, default=0.25, help="Detection NMS IoU threshold")
    benchmark.add_argument("--eval-iou", type=float, default=0.5, help="Evaluation IoU threshold")
    benchmark.add_argument("--max-detections", type=int, default=300, help="Per-sample detection cap")
    benchmark.add_argument("--padding", type=int, default=4, help="Truth box padding in pixels")
    benchmark.set_defaults(func=archcad_benchmark_template_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
