from __future__ import annotations

import argparse
from pathlib import Path

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
            symbol_path=args.symbol,
            out_dir=args.out,
            label=args.label,
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
    print(f"Geometry detections: {result['geometry']}")
    print(f"VLM/manual detections: {result['vlm']}")
    print(f"Final detections after NMS: {result['final']}")
    print(f"Annotated output: {args.out / 'annotated.png'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ki-cad")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Find a symbol crop in a CAD PDF/image")
    detect.add_argument("--input", required=True, type=Path, help="Input PDF or image")
    detect.add_argument("--page", type=int, default=1, help="1-based PDF page number")
    detect.add_argument("--dpi", type=int, default=200, help="PDF render DPI")
    detect.add_argument("--symbol", required=True, type=Path, help="Target symbol crop image")
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
