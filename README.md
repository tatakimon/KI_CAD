# KI_CAD

MVP pipeline for finding repeated engineering/CAD symbols in large blueprint PDFs or images.

The project is intentionally built around two comparable detection tracks:

- **Geometry/OpenCV track**: fast edge/dot-style template matching from a target symbol crop.
- **VLM/manual track**: optional JSON detections in the same box format, so VLM results can be compared or fused later.

The first working command renders one PDF/SVG page, slices it into VLM-ready tiles, runs geometry matching, merges duplicate boxes with NMS, and writes annotated outputs.

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

## ArchCAD Samples

After downloading the smaller ArchCAD archives into `data/raw/archcad`, extract one aligned SVG/JSON/caption sample:

```powershell
$env:PYTHONPATH="src"
python -m ki_cad archcad extract-sample --out data\interim\archcad_sample
```

This writes a renderable SVG plus a PNG preview for quick detector experiments.

## Scoring Results

Detector confidence is not the same as accuracy. To score accuracy, compare predictions with approved boxes:

```powershell
python -m ki_cad evaluate --predictions runs\demo\final_detections.json --truth data\interim\truth.json --iou 0.5
```

The evaluator reports precision, recall, F1, and mean matched IoU.

For ArchCAD samples, export ground-truth boxes from a semantic class first:

```powershell
python -m ki_cad archcad export-truth `
  --svg data\interim\archcad_sample\sample.svg `
  --json data\interim\archcad_sample\sample.json `
  --semantic 11 `
  --label fixture `
  --out data\interim\archcad_sample\fixture_truth.json `
  --preview data\interim\archcad_sample\fixture_truth.png
```

Run a quick multi-sample ArchCAD benchmark with the same target crop:

```powershell
python -m ki_cad archcad benchmark-template `
  --symbol data\interim\archcad_sample\symbol_crop.png `
  --semantic 11 `
  --label fixture `
  --count 20 `
  --out runs\archcad_semantic11_benchmark
```

To cover visual variants, export multiple templates from ArchCAD annotations and benchmark with the directory:

```powershell
python -m ki_cad archcad export-templates `
  --svg data\interim\archcad_sample\sample.svg `
  --json data\interim\archcad_sample\sample.json `
  --semantic 11 `
  --out data\interim\semantic11_templates

python -m ki_cad archcad benchmark-template `
  --symbol-dir data\interim\semantic11_templates `
  --semantic 11 `
  --label fixture `
  --count 20 `
  --out runs\archcad_semantic11_multitemplate_benchmark
```
