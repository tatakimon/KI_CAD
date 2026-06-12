from ki_cad.core.models import Box, Detection
from ki_cad.evaluation import evaluate_detections


def test_evaluate_detections_reports_precision_recall_f1() -> None:
    predictions = [
        Detection("fixture", 0.9, Box(0, 0, 100, 100), "pred_good"),
        Detection("fixture", 0.8, Box(300, 300, 350, 350), "pred_bad"),
    ]
    truth = [
        Detection("fixture", 1.0, Box(10, 10, 110, 110), "truth"),
    ]

    result = evaluate_detections(predictions, truth, iou_threshold=0.5)

    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 0
    assert result.precision == 0.5
    assert result.recall == 1.0
