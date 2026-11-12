import numpy as np
import pytest

from src.post_eval import compute_pro, eval_classification, eval_segmentation, mean_top1p, parse_dataset_files, trapezoid
from src.utils import dists2map
import tifffile as tiff


def test_mean_top1p():
    assert mean_top1p(np.arange(1000, dtype=float)) == pytest.approx(994.5)  # mean of top 10 values
    assert mean_top1p(np.array([0.1, 0.7, 0.3])) == pytest.approx(0.7)      # < 100 values: maximum


def test_trapezoid():
    assert trapezoid([0, 1], [0, 1]) == pytest.approx(0.5)
    assert trapezoid([0, 1], [0, 1], x_max=0.5) == pytest.approx(0.125)  # interpolated at x_max


def test_pro_curve_perfect_and_random_prediction():
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[4:10, 4:10] = 1
    gt[20:28, 15:25] = 1
    fprs, pros = compute_pro([gt.astype(float)], [gt])
    assert trapezoid(fprs, pros, x_max=0.3) / 0.3 == pytest.approx(1.0)

    noise = np.random.default_rng(0).random(gt.shape)
    fprs, pros = compute_pro([noise], [gt])
    assert trapezoid(fprs, pros, x_max=0.3) / 0.3 < 0.5


def write_patch_distances(toy_dataset, anomaly_maps_dir, grid=(8, 10)):
    """Patch distances as saved by run_anomaly_detection: low for good samples, high inside the defect."""
    gt, pred = parse_dataset_files("toy", str(toy_dataset), str(anomaly_maps_dir), dataset="MVTec")
    for gt_name, pred_name in zip(gt, pred):
        dists = np.full(grid, 0.1, dtype=np.float32)
        if gt_name is not None:
            i = int(pred_name[-3:])
            dists[1 + 2 * i:3 + 2 * i, 1 + 2 * i:3 + 2 * i] = 0.9  # 8x8 patches covering the red square
        (anomaly_maps_dir / "toy" / "test" / pred_name.split("/")[-2]).mkdir(parents=True, exist_ok=True)
        np.save(pred_name + ".npy", dists)
        tiff.imwrite(pred_name + ".tiff", dists2map(dists, (64, 80)))
    return gt, pred


def test_parse_dataset_files(toy_dataset, tmp_path):
    gt, pred = parse_dataset_files("toy", str(toy_dataset), str(tmp_path / "maps"), dataset="MVTec")
    assert len(gt) == len(pred) == 6
    assert sum(g is None for g in gt) == 3  # 'good' samples have no ground truth mask
    assert all(g.endswith("_mask.png") for g in gt if g is not None)


@pytest.mark.parametrize("aggregation", ["meantop1p", "max_patch_distance", "max_anomaly_map"])
def test_eval_classification(toy_dataset, tmp_path, aggregation):
    gt, pred = write_patch_distances(toy_dataset, tmp_path / "maps")
    auroc, ap, f1 = eval_classification(gt, pred, aggregation_statistics=aggregation,
                                        dataset_base_dir=str(toy_dataset), dataset="MVTec")
    assert (auroc, ap, f1) == pytest.approx((1.0, 1.0, 1.0))


def test_eval_segmentation(toy_dataset, tmp_path):
    gt, pred = write_patch_distances(toy_dataset, tmp_path / "maps")
    au_pro, auroc_px, f1_px = eval_segmentation(gt, pred, delete_tiff_files=True)
    assert au_pro > 0.9 and auroc_px > 0.95 and f1_px > 0.5
    assert not any((tmp_path / "maps").rglob("*.tiff"))


def test_create_sample_plots_without_existing_output_dir(toy_dataset, tmp_path, monkeypatch):
    # run_anomalydino.py does not create the per-object directories with --no-save_examples
    import src.visualize as visualize
    monkeypatch.setattr(visualize, "get_dataset_info", lambda dataset, preprocess, data_path=None: (["toy"], {"toy": ["bad"]}, None, None))
    write_patch_distances(toy_dataset, tmp_path / "maps")
    visualize.create_sample_plots(str(tmp_path / "results"), str(tmp_path / "maps"), seed=0, dataset="MVTec", data_root=str(toy_dataset))
    assert (tmp_path / "results" / "toy" / "anomaly_maps_examples_seed=0.png").exists()


def test_create_sample_plots_when_anomalous_map_exceeds_good_maps(toy_dataset, tmp_path, monkeypatch):
    # all distances of an anomalous image above the maximum of the good images (vmax) -> vmin > vmax must not crash
    import src.visualize as visualize
    monkeypatch.setattr(visualize, "get_dataset_info", lambda dataset, preprocess, data_path=None: (["toy"], {"toy": ["bad"]}, None, None))
    gt, pred = write_patch_distances(toy_dataset, tmp_path / "maps")
    for gt_name, pred_name in zip(gt, pred):
        if gt_name is not None:
            np.save(pred_name + ".npy", np.full((8, 10), 0.5, dtype=np.float32))  # good maps: 0.1 everywhere
    visualize.create_sample_plots(str(tmp_path / "results"), str(tmp_path / "maps"), seed=0, dataset="MVTec", data_root=str(toy_dataset))
    assert (tmp_path / "results" / "toy" / "anomaly_maps_examples_seed=0.png").exists()
