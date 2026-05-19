from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pymupdf


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_input(path: Path, page: int, dpi: int) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return render_pdf_page(path, page=page, dpi=dpi)
    if suffix in IMAGE_SUFFIXES:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        return image
    raise ValueError(f"Unsupported input type: {path.suffix}")


def render_pdf_page(path: Path, page: int, dpi: int) -> np.ndarray:
    if page < 1:
        raise ValueError("--page is 1-based and must be >= 1")

    doc = pymupdf.open(path)
    try:
        if page > doc.page_count:
            raise ValueError(f"{path} has {doc.page_count} pages, cannot render page {page}")

        pdf_page = doc[page - 1]
        scale = dpi / 72.0
        pix = pdf_page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    finally:
        doc.close()


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise ValueError(f"Could not save image: {path}")
