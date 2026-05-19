# KI_CAD

MVP pipeline for finding repeated engineering/CAD symbols in large blueprint PDFs or images.

The project is intentionally built around two comparable detection tracks:

- **Geometry/OpenCV track**: fast edge/dot-style template matching from a target symbol crop.
- **VLM/manual track**: optional JSON detections in the same box format, so VLM results can be compared or fused later.

The first working command renders one PDF page, slices it into VLM-ready tiles, runs geometry matching, merges duplicate boxes with NMS, and writes annotated outputs.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m ki_cad detect --input path\to\drawing.pdf --page 1 --symbol path\to\symbol.png --label valve --out runs\demo
```

Outputs:

- `rendered_page.png`
- `tiles/*.png`
- `manifest.json`
- `geometry_detections.json`
- `vlm_detections.json`
- `final_detections.json`
- `tile_debug.png`
- `annotated.png`

## Why This Shape

The MVP needs visible results quickly, but it should still match the intended architecture:

1. Render massive CAD PDFs/images.
2. Slice into overlapping tiles and keep global coordinates.
3. Run a deterministic geometry matcher now.
4. Accept VLM/manual detections in the same schema.
5. Fuse results with NMS and output an annotated page.

Later we can add:

- real VLM API detector over tiles,
- human approval UI,
- YOLO/RT-DETR label export,
- ArchCAD-style dataset experiments.
