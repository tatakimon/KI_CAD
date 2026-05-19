from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import Tile


def tile_starts(total: int, tile_size: int, overlap: int) -> list[int]:
    if total <= tile_size:
        return [0]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    starts = list(range(0, total - tile_size + 1, stride))
    last = total - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def write_tiles(image: np.ndarray, out_dir: Path, page: int, tile_size: int, overlap: int) -> list[Tile]:
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    height, width = image.shape[:2]
    tiles: list[Tile] = []
    for y in tile_starts(height, tile_size, overlap):
        for x in tile_starts(width, tile_size, overlap):
            crop = image[y : y + tile_size, x : x + tile_size]
            tile_id = f"page{page:03d}_x{x:05d}_y{y:05d}"
            rel_file = f"tiles/{tile_id}.png"
            cv2.imwrite(str(out_dir / rel_file), crop)
            tiles.append(Tile(tile_id, rel_file, x, y, crop.shape[1], crop.shape[0]))

    return tiles


def draw_tile_debug(image: np.ndarray, tiles: list[Tile]) -> np.ndarray:
    debug = image.copy()
    for tile in tiles:
        cv2.rectangle(debug, (tile.x, tile.y), (tile.x + tile.width, tile.y + tile.height), (0, 180, 255), 2)
    return debug
