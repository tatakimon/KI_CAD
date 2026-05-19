from ki_cad.core.slicing import tile_starts


def test_tile_starts_cover_end_boundary() -> None:
    assert tile_starts(total=2500, tile_size=1024, overlap=256) == [0, 768, 1476]


def test_tile_starts_small_image_returns_origin() -> None:
    assert tile_starts(total=800, tile_size=1024, overlap=256) == [0]
