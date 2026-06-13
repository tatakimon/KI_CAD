from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from ki_cad.core.models import Box, Detection
from ki_cad.core.render import load_input, save_image
from ki_cad.visualization.annotate import draw_detections


ARCHCAD_CLASS_NAMES = {
    0: "axis_grid",
    1: "single_door",
    2: "double_door",
    3: "parent_child_door",
    4: "other_door",
    5: "elevator",
    6: "staircase",
    7: "sink",
    8: "urinal",
    9: "toilet",
    10: "bathtub",
    11: "squat_toilet",
    12: "other_fixtures",
    13: "drain",
    14: "table",
    15: "chair",
    16: "bed",
    17: "sofa",
    18: "hole",
    19: "glass",
    20: "wall",
    21: "concrete_column",
    22: "steel_column",
    23: "concrete_beam",
    24: "steel_beam",
    25: "parking_space",
    26: "foundation",
    27: "pile",
    28: "rebar",
    29: "fire_hydrant",
    100: "others",
}


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


def find_sample_ids_with_semantic(raw_dir: Path, semantic: int, limit: int) -> list[str]:
    json_zip = raw_dir / "json.zip"
    if not json_zip.exists():
        raise FileNotFoundError(f"Missing {json_zip}")

    sample_ids: list[str] = []
    with zipfile.ZipFile(json_zip) as archive:
        for name in archive.namelist():
            if not name.startswith("json/") or not name.endswith(".json"):
                continue
            data = json.loads(archive.read(name))
            if any(entity.get("semantic") == semantic for entity in data.get("entities", [])):
                sample_ids.append(Path(name).stem)
                if len(sample_ids) >= limit:
                    break

    return sample_ids


def find_sample_ids_with_semantics(raw_dir: Path, semantics: list[int], limit_per_semantic: int) -> dict[int, list[str]]:
    json_zip = raw_dir / "json.zip"
    if not json_zip.exists():
        raise FileNotFoundError(f"Missing {json_zip}")

    targets = set(semantics)
    found: dict[int, list[str]] = {semantic: [] for semantic in semantics}
    with zipfile.ZipFile(json_zip) as archive:
        for name in archive.namelist():
            if not name.startswith("json/") or not name.endswith(".json"):
                continue
            data = json.loads(archive.read(name))
            sample_semantics = {entity.get("semantic") for entity in data.get("entities", [])}
            sample_id = Path(name).stem
            for semantic in targets.intersection(sample_semantics):
                if len(found[semantic]) < limit_per_semantic:
                    found[semantic].append(sample_id)
            if all(len(ids) >= limit_per_semantic for ids in found.values()):
                break

    return found


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


def export_instance_crops(
    json_path: Path,
    svg_path: Path,
    semantic: int,
    out_dir: Path,
    dpi: int,
    padding: int = 8,
    max_crops: int | None = None,
) -> list[Path]:
    truth_path = out_dir / "_truth_boxes.json"
    detections = export_instance_boxes(
        json_path=json_path,
        svg_path=svg_path,
        semantic=semantic,
        out_json=truth_path,
        dpi=dpi,
        padding=padding,
    )
    image = load_input(svg_path, page=1, dpi=dpi)
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_paths: list[Path] = []
    for index, detection in enumerate(detections):
        if max_crops is not None and len(crop_paths) >= max_crops:
            break
        box = detection.box
        if box.width < 8 or box.height < 8:
            continue
        crop = image[box.y1 : box.y2, box.x1 : box.x2]
        crop_path = out_dir / f"semantic_{semantic}_{index:03d}_{detection.source}.png"
        save_image(crop_path, crop)
        crop_paths.append(crop_path)

    return crop_paths


def build_template_library(
    raw_dir: Path,
    out_dir: Path,
    semantics: list[int],
    samples_per_class: int,
    max_crops_per_class: int,
    dpi: int,
    padding: int,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_ids_by_semantic = find_sample_ids_with_semantics(
        raw_dir=raw_dir,
        semantics=semantics,
        limit_per_semantic=samples_per_class,
    )

    manifest: dict[str, object] = {
        "raw_dir": str(raw_dir),
        "dpi": dpi,
        "padding": padding,
        "samples_per_class": samples_per_class,
        "max_crops_per_class": max_crops_per_class,
        "classes": {},
    }

    for semantic in semantics:
        class_name = ARCHCAD_CLASS_NAMES.get(semantic, f"semantic_{semantic}")
        class_dir = out_dir / f"{semantic:03d}_{class_name}"
        class_dir.mkdir(parents=True, exist_ok=True)
        sample_ids = sample_ids_by_semantic.get(semantic, [])
        exported: list[str] = []

        for sample_id in sample_ids:
            if len(exported) >= max_crops_per_class:
                break
            sample_dir = out_dir / "_samples" / sample_id
            sample = extract_sample(raw_dir=raw_dir, out_dir=sample_dir, sample_id=sample_id, dpi=dpi)
            if sample.json_path is None:
                continue
            temp_dir = out_dir / "_tmp" / f"{semantic}_{sample_id}"
            crop_paths = export_instance_crops(
                json_path=sample.json_path,
                svg_path=sample.svg_path,
                semantic=semantic,
                out_dir=temp_dir,
                dpi=dpi,
                padding=padding,
            )
            for crop_path in crop_paths:
                if len(exported) >= max_crops_per_class:
                    break
                target = class_dir / _safe_template_name(semantic, class_name, sample_id, len(exported), crop_path.name)
                target.write_bytes(crop_path.read_bytes())
                exported.append(str(target))

        manifest["classes"][str(semantic)] = {
            "name": class_name,
            "folder": str(class_dir),
            "sample_ids": sample_ids,
            "templates": exported,
            "template_count": len(exported),
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _safe_template_name(semantic: int, class_name: str, sample_id: str, index: int, original_name: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", original_name)
    return f"{semantic:03d}_{class_name}_{index:03d}_{sample_id[:8]}_{suffix}"
