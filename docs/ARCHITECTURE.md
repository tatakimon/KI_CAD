# Architecture

KI_CAD is organized around a stable detection contract rather than one model choice.

## Flow

```text
PDF/SVG/image input
  -> render/load page
  -> slice page into overlapping tiles
  -> geometry detector from symbol crop
  -> optional VLM/manual detections
  -> local/global box normalization
  -> NMS
  -> JSON + annotated image
```

## Detection Contract

All detectors must emit global page coordinates:

```json
{
  "label": "valve",
  "score": 0.82,
  "box": {
    "x1": 120,
    "y1": 200,
    "x2": 180,
    "y2": 260,
    "xywh": [120, 200, 60, 60]
  },
  "source": "dot_geometry_template_scale_1.00"
}
```

This lets OpenCV, VLM, manual review, and future YOLO predictions be compared directly.

## MVP Defaults

- PDF page numbers are 1-based in the CLI.
- Default render DPI is `200`.
- Default tile size is `1024`.
- Default overlap is `256`.
- Geometry matching uses Canny edges plus multi-scale template matching.
