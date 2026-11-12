import os
import sys

import cv2
import numpy as np
import pytest

# make 'src' and the run scripts importable when running 'pytest' from the repository root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def toy_dataset(tmp_path):
    """
    Minimal dataset in MVTec format with one object ('toy'): uniform gray images are normal,
    images with a red square are anomalous ('bad', with ground truth masks).
    """
    root = tmp_path / "data"
    obj = root / "toy"
    for d in ["train/good", "test/good", "test/bad", "ground_truth/bad"]:
        (obj / d).mkdir(parents=True)

    h, w = 64, 80
    gray = np.full((h, w, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(obj / "train/good/000.png"), gray)
    for i in range(3):
        cv2.imwrite(str(obj / f"test/good/{i:03d}.png"), gray + 10 * i)
        bad, mask = gray.copy(), np.zeros((h, w), dtype=np.uint8)
        y, x = 8 + 16 * i, 8 + 16 * i
        bad[y:y + 16, x:x + 16] = (0, 0, 255)  # BGR: red square
        mask[y:y + 16, x:x + 16] = 255
        cv2.imwrite(str(obj / f"test/bad/{i:03d}.png"), bad)
        cv2.imwrite(str(obj / f"ground_truth/bad/{i:03d}_mask.png"), mask)
    return root


class DummyModel:
    """
    Stand-in for the backbone wrappers in src/backbones.py (same interface, no network):
    the 'feature' of each 8x8 patch is its mean color.
    """
    patch_size = 8

    def prepare_image(self, img):
        if isinstance(img, str):
            img = cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        grid_size = (img.shape[0] // self.patch_size, img.shape[1] // self.patch_size)
        img = img[:grid_size[0] * self.patch_size, :grid_size[1] * self.patch_size]
        return img.astype(np.float32), grid_size

    def extract_features(self, img):
        g0, g1 = img.shape[0] // self.patch_size, img.shape[1] // self.patch_size
        patches = img.reshape(g0, self.patch_size, g1, self.patch_size, 3)
        return patches.mean(axis=(1, 3)).reshape(-1, 3).astype(np.float32)

    def compute_background_mask(self, img_features, grid_size, threshold=10, masking_type=False):
        return np.ones(img_features.shape[0], dtype=bool)

    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None, normalize=True):
        return np.zeros((*grid_size, 3))


@pytest.fixture
def dummy_model():
    return DummyModel()
