from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ki_cad.core.nms import non_max_suppression_by_label
from ki_cad.core.render import load_input, save_image
from ki_cad.core.slicing import draw_tile_debug, write_tiles
from ki_cad.detectors.geometry import detect_by_labeled_geometry_templates
from ki_cad.detectors.vlm_json import load_vlm_or_manual_detections
from ki_cad.visualization.annotate import draw_detections


@dataclass(frozen=True)
class DetectConfig:
    input_path: Path
    out_dir: Path
    label: str
    symbol_path: Path | None = None
    symbol_dir: Path | None = None
    template_root: Path | None = None
    page: int = 1
    dpi: int = 200
    threshold: float = 0.45
    scales: tuple[float, ...] = (0.75, 0.9, 1.0, 1.1, 1.25)
    nms_iou: float = 0.25
    max_detections: int = 200
    tile_size: int = 1024
    overlap: int = 256
    vlm_json: Path | None = None


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _symbol_templates(config: DetectConfig) -> list[tuple[Path, str]]:
    templates: list[tuple[Path, str]] = []
    if config.symbol_path is not None:
        templates.append((config.symbol_path, config.label))
    if config.symbol_dir is not None:
        templates.extend((path, config.label) for path in _image_files(config.symbol_dir))
    if config.template_root is not None:
        for class_dir in sorted(path for path in config.template_root.iterdir() if path.is_dir()):
            label = class_dir.name.split("_", 1)[1] if "_" in class_dir.name else class_dir.name
            templates.extend((path, label) for path in _image_files(class_dir))
    if not templates:
        raise ValueError("Provide symbol_path, symbol_dir, or template_root")
    return templates


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )


def run_detection(config: DetectConfig) -> dict[str, int]:
    config.out_dir.mkdir(parents=True, exist_ok=True)

    image = load_input(config.input_path, page=config.page, dpi=config.dpi)
    save_image(config.out_dir / "rendered_page.png", image)

    tiles = write_tiles(
        image,
        config.out_dir,
        page=config.page,
        tile_size=config.tile_size,
        overlap=config.overlap,
    )
    save_image(config.out_dir / "tile_debug.png", draw_tile_debug(image, tiles))

    templates = _symbol_templates(config)
    geometry = detect_by_labeled_geometry_templates(
        image=image,
        templates=templates,
        threshold=config.threshold,
        scales=list(config.scales),
        nms_iou=config.nms_iou,
        max_detections=config.max_detections,
    )
    vlm = load_vlm_or_manual_detections(config.vlm_json, label=config.label)
    final = non_max_suppression_by_label(geometry + vlm, iou_threshold=config.nms_iou)

    _write_json(config.out_dir / "geometry_detections.json", [item.to_dict() for item in geometry])
    _write_json(config.out_dir / "vlm_detections.json", [item.to_dict() for item in vlm])
    _write_json(config.out_dir / "final_detections.json", [item.to_dict() for item in final])
    _write_json(
        config.out_dir / "manifest.json",
        {
            "source": str(config.input_path),
            "page": config.page,
            "dpi": config.dpi,
            "image_width": image.shape[1],
            "image_height": image.shape[0],
            "tile_size": config.tile_size,
            "overlap": config.overlap,
            "templates": [{"file": str(path), "label": label} for path, label in templates],
            "tiles": [tile.to_dict() for tile in tiles],
        },
    )

    save_image(config.out_dir / "annotated.png", draw_detections(image, final))

    return {
        "tiles": len(tiles),
        "symbols": len(templates),
        "geometry": len(geometry),
        "vlm": len(vlm),
        "final": len(final),
    }
