from ki_cad.core.models import Box, Detection
from ki_cad.core.nms import non_max_suppression


def test_nms_keeps_highest_scoring_overlapping_box() -> None:
    detections = [
        Detection("valve", 0.7, Box(0, 0, 100, 100), "a"),
        Detection("valve", 0.9, Box(10, 10, 110, 110), "b"),
        Detection("valve", 0.8, Box(300, 300, 350, 350), "c"),
    ]

    kept = non_max_suppression(detections, iou_threshold=0.25)

    assert [item.source for item in kept] == ["b", "c"]
