from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ki_cad.core.render import load_input, save_image


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
