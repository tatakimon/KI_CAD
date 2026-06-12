from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from ki_cad.core.models import Box, Detection
from ki_cad.core.render import load_input, save_image
from ki_cad.visualization.annotate import draw_detections


@dataclass(frozen=True)
class ArchCadSample:
    sample_id: str
    svg_path: Path
    json_path: Path | None
    caption_path: Path | None
    preview_path: Path


def first_svg_id(svg_zip: Path) -> str:
    with zipfile.ZipFile(svg_zip) as archive:
        for name in archive.namelist():
            if name.startswith("svg/") and name.endswith(".svg"):
                return Path(name).stem
    raise ValueError(f"No SVG samples found in {svg_zip}")


def extract_sample(raw_dir: Path, out_dir: Path, sample_id: str | None, dpi: int) -> ArchCadSample:
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_zip = raw_dir / "svg.zip"
    json_zip = raw_dir / "json.zip"
    caption_zip = raw_dir / "caption.zip"

    if not svg_zip.exists():
        raise FileNotFoundError(f"Missing {svg_zip}")

    selected_id = sample_id or first_svg_id(svg_zip)
    svg_path = out_dir / f"{selected_id}.svg"
    json_path = out_dir / f"{selected_id}.json"
    caption_path = out_dir / f"{selected_id}.caption.json"
    preview_path = out_dir / f"{selected_id}.preview.png"

    with zipfile.ZipFile(svg_zip) as archive:
        svg_path.write_bytes(archive.read(f"svg/{selected_id}.svg"))

    extracted_json: Path | None = None
    if json_zip.exists():
        with zipfile.ZipFile(json_zip) as archive:
            json_path.write_bytes(archive.read(f"json/{selected_id}.json"))
        extracted_json = json_path

    extracted_caption: Path | None = None
    if caption_zip.exists():
        with zipfile.ZipFile(caption_zip) as archive:
            caption_path.write_bytes(archive.read(f"caption/{selected_id}.json"))
        extracted_caption = caption_path

    image = load_input(svg_path, page=1, dpi=dpi)
    save_image(preview_path, image)

    return ArchCadSample(
        sample_id=selected_id,
        svg_path=svg_path,
        json_path=extracted_json,
        caption_path=extracted_caption,
        preview_path=preview_path,
    )


def summarize_annotation(json_path: Path) -> dict[str, object]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    entities = data.get("entities", [])
    counts: dict[str, int] = {}
    for entity in entities:
        semantic = str(entity.get("semantic", "missing"))
        counts[semantic] = counts.get(semantic, 0) + 1
    return {
        "entities": len(entities),
        "semantic_counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def _svg_viewbox_size(svg_path: Path) -> tuple[float, float]:
    root = ElementTree.parse(svg_path).getroot()
    viewbox = root.attrib.get("viewBox")
    if viewbox:
        _, _, width, height = [float(part) for part in viewbox.split()]
        return width, height
    return float(root.attrib["width"]), float(root.attrib["height"])


def _entity_bounds(entity: dict[str, object]) -> tuple[float, float, float, float] | None:
    points: list[tuple[float, float]] = []
    for key in ("start", "end", "center"):
        value = entity.get(key)
        if isinstance(value, list) and len(value) >= 2:
            points.append((float(value[0]), float(value[1])))

    radius = entity.get("radius")
    center = entity.get("center")
    if isinstance(radius, (int, float)) and isinstance(center, list) and len(center) >= 2:
        cx, cy = float(center[0]), float(center[1])
        r = float(radius)
        points.extend([(cx - r, cy - r), (cx + r, cy + r)])

    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def export_instance_boxes(
    json_path: Path,
    svg_path: Path,
    semantic: int,
    out_json: Path,
    dpi: int,
    preview_path: Path | None = None,
    label: str | None = None,
    padding: int = 0,
) -> list[Detection]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image = load_input(svg_path, page=1, dpi=dpi)
    viewbox_width, viewbox_height = _svg_viewbox_size(svg_path)
    scale_x = image.shape[1] / viewbox_width
    scale_y = image.shape[0] / viewbox_height

    groups: dict[str, list[tuple[float, float, float, float]]] = {}
    for index, entity in enumerate(data.get("entities", [])):
        if entity.get("semantic") != semantic:
            continue
        bounds = _entity_bounds(entity)
        if bounds is None:
            continue
        group = str(entity.get("instance", f"semantic_{semantic}_entity_{index}"))
        groups.setdefault(group, []).append(bounds)

    detections: list[Detection] = []
    output_label = label or f"semantic_{semantic}"
    for group, bounds_list in groups.items():
        x1 = min(bounds[0] for bounds in bounds_list)
        y1 = min(bounds[1] for bounds in bounds_list)
        x2 = max(bounds[2] for bounds in bounds_list)
        y2 = max(bounds[3] for bounds in bounds_list)
        box = Box(
            max(0, int(round(x1 * scale_x)) - padding),
            max(0, int(round(y1 * scale_y)) - padding),
            min(image.shape[1], int(round(x2 * scale_x)) + padding),
            min(image.shape[0], int(round(y2 * scale_y)) + padding),
        )
        detections.append(Detection(label=output_label, score=1.0, box=box, source=group))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps([item.to_dict() for item in detections], indent=2), encoding="utf-8")

    if preview_path is not None:
        save_image(preview_path, draw_detections(image, detections))

    return detections
