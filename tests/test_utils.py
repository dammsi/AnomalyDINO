import numpy as np
import pytest

from src.utils import PREPROCESS_OPTIONS, augment_image, dists2map, get_dataset_info, limit_cpu_threads, list_images, resize_mask_img


@pytest.mark.parametrize("dataset, n_objects", [("MVTec", 15), ("VisA", 12)])
@pytest.mark.parametrize("preprocess", PREPROCESS_OPTIONS)
def test_dataset_info_is_complete(dataset, n_objects, preprocess):
    objects, object_anomalies, masking, rotation = get_dataset_info(dataset, preprocess)
    assert len(objects) == n_objects
    for d in (object_anomalies, masking, rotation):
        assert set(d) == set(objects)
    assert all(isinstance(v, bool) for v in [*masking.values(), *rotation.values()])
    assert all("good" not in a for a in object_anomalies.values())


def test_mvtec_preprocessing_settings():
    objects, _, masking, rotation = get_dataset_info("MVTec", "informed")
    assert {o for o in objects if rotation[o]} == {"hazelnut", "screw"}
    assert {o for o in objects if masking[o]} == {"capsule", "hazelnut", "pill", "screw", "toothbrush"}

    _, _, masking_agn, rotation_agn = get_dataset_info("MVTec", "agnostic")
    assert masking_agn == masking and all(rotation_agn.values())

    _, _, masking_nm, _ = get_dataset_info("MVTec", "agnostic_no_mask")
    assert not any(masking_nm.values())


@pytest.mark.parametrize("preprocess, mask, rot", [
    ("force_no_mask_no_rotation", False, False), ("force_mask_no_rotation", True, False),
    ("force_no_mask_rotation", False, True), ("force_mask_rotation", True, True)])
@pytest.mark.parametrize("dataset", ["MVTec", "VisA", "custom"])
def test_force_options(toy_dataset, dataset, preprocess, mask, rot):
    objects, _, masking, rotation = get_dataset_info(dataset, preprocess, data_path=str(toy_dataset))
    assert all(masking[o] is mask and rotation[o] is rot for o in objects)


def test_custom_dataset_is_inferred(toy_dataset):
    objects, object_anomalies, masking, rotation = get_dataset_info("custom", "agnostic", data_path=str(toy_dataset))
    assert objects == ["toy"]
    assert object_anomalies == {"toy": ["bad"]}
    assert masking == {"toy": False} and rotation == {"toy": True}


def test_invalid_inputs():
    with pytest.raises(ValueError):
        get_dataset_info("MVTec", "unknown_preprocessing")
    with pytest.raises(ValueError):
        get_dataset_info("custom", "agnostic", data_path=None)


def test_augment_image():
    img = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    augmented = augment_image(img)
    assert len(augmented) == 8
    assert all(a.shape == img.shape for a in augmented)
    np.testing.assert_array_equal(augmented[0], img)  # first angle is 0 degrees


def test_dists2map_and_mask_resizing_match_image_shape():
    dists = np.random.default_rng(0).random((4, 6)).astype(np.float32)
    assert dists2map(dists, (64, 96, 3)).shape == (64, 96)
    assert resize_mask_img(np.ones(24, dtype=bool), (64, 96, 3), (4, 6)).shape == (64, 96)


def test_list_images_skips_other_files(tmp_path):
    for name in ["b.png", "a.JPG", "c.jpeg", ".DS_Store", "readme.txt", "Thumbs.db"]:
        (tmp_path / name).write_bytes(b"")
    assert list_images(tmp_path) == ["a.JPG", "b.png", "c.jpeg"]  # sorted, extension check ignores case


def test_limit_cpu_threads():
    import torch
    from threadpoolctl import threadpool_info
    before = torch.get_num_threads()
    try:
        limit_cpu_threads(0)  # no limit -> unchanged
        assert torch.get_num_threads() == before
        limit_cpu_threads(2)
        assert torch.get_num_threads() == 2
        assert all(p["num_threads"] <= 2 for p in threadpool_info())
    finally:
        torch.set_num_threads(before)
