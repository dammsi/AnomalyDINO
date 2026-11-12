import copy

import numpy as np
import pytest
import torch

from src.detection import run_anomaly_detection
from src.post_eval import eval_classification, parse_dataset_files


@pytest.fixture(autouse=True)
def no_cuda_sync(monkeypatch):
    # run_anomaly_detection synchronizes CUDA for timing, not needed for the dummy model (and fails without GPU)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


@pytest.mark.parametrize("knn_metric", ["L2", "L2_normalized"])
@pytest.mark.parametrize("rotation", [False, True])
def test_run_anomaly_detection(toy_dataset, dummy_model, tmp_path, knn_metric, rotation):
    object_anomalies = {"toy": ["bad"]}
    object_anomalies_before = copy.deepcopy(object_anomalies)

    scores, time_memorybank, inference_times = run_anomaly_detection(
        dummy_model, "toy", str(toy_dataset), n_ref_samples=1, object_anomalies=object_anomalies,
        plots_dir=str(tmp_path), rotation=rotation, knn_metric=knn_metric, faiss_on_cpu=True)

    assert object_anomalies == object_anomalies_before  # input is not modified
    assert set(scores) == set(inference_times) == {f"{t}/{i:03d}.png" for t in ["bad", "good"] for i in range(3)}
    assert time_memorybank >= 0
    good = [scores[f"good/{i:03d}.png"] for i in range(3)]
    bad = [scores[f"bad/{i:03d}.png"] for i in range(3)]
    assert max(bad) > 0 and min(bad) > max(good)
    if knn_metric == "L2_normalized":  # brighter gray has the same color direction -> distance 0
        assert max(good) == pytest.approx(0.0, abs=1e-5)

    # saved patch distances can be evaluated (AUROC = 1 for this easy toy problem)
    anomaly_maps = tmp_path / "anomaly_maps" / "seed=0"
    assert np.load(anomaly_maps / "toy" / "test" / "bad" / "000.npy").shape == (8, 10)
    gt, pred = parse_dataset_files("toy", str(toy_dataset), str(anomaly_maps), dataset="MVTec")
    auroc, _, _ = eval_classification(gt, pred)
    assert auroc == pytest.approx(1.0)


def test_no_reference_samples_left(toy_dataset, dummy_model, tmp_path):
    with pytest.raises(ValueError, match="No reference samples"):
        run_anomaly_detection(dummy_model, "toy", str(toy_dataset), n_ref_samples=1, object_anomalies={"toy": ["bad"]},
                              plots_dir=str(tmp_path), faiss_on_cpu=True, seed=5)


def make_custom(toy_dataset):
    """Turn the toy dataset into a 'custom' one: anomaly folder 'type-1', a .jpg test image, one ground truth mask missing."""
    import cv2
    obj = toy_dataset / "toy"
    (obj / "test" / "bad").rename(obj / "test" / "type-1")
    (obj / "ground_truth" / "bad").rename(obj / "ground_truth" / "type-1")
    img = cv2.imread(str(obj / "test" / "good" / "001.png"))
    cv2.imwrite(str(obj / "test" / "good" / "001.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    (obj / "test" / "good" / "001.png").unlink()
    (obj / "ground_truth" / "type-1" / "002_mask.png").unlink()
    return obj


def test_custom_dataset_end_to_end(toy_dataset, dummy_model, tmp_path):
    from src.post_eval import eval_finished_run, eval_segmentation
    from src.utils import get_dataset_info
    make_custom(toy_dataset)
    objects, object_anomalies, _, _ = get_dataset_info("custom", "agnostic", data_path=str(toy_dataset))
    assert objects == ["toy"] and object_anomalies == {"toy": ["type-1"]}

    run_anomaly_detection(dummy_model, "toy", str(toy_dataset), n_ref_samples=1, object_anomalies=object_anomalies,
                          plots_dir=str(tmp_path), faiss_on_cpu=True, save_tiffs=True)
    anomaly_maps = str(tmp_path / "anomaly_maps" / "seed=0")
    gt, pred = parse_dataset_files("toy", str(toy_dataset), anomaly_maps, dataset="custom")
    assert len(gt) == 6 and sum(g is None for g in gt) == 3  # incl. the .jpg image and the 'type-1' folder

    eval_finished_run("custom", str(toy_dataset), anomaly_maps, str(tmp_path), seed=0, objects=objects, delete_tiff_files=False)
    import json
    metrics = json.load(open(tmp_path / "metrics_seed=0.json"))
    assert metrics["mean_classification_au_roc"] == pytest.approx(1.0)  # image without mask still counted as anomalous

    with pytest.raises(FileNotFoundError, match="ground truth masks"):
        eval_segmentation(gt, pred)
