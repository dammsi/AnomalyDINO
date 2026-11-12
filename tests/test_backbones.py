import numpy as np
import pytest
import torch

from src.backbones import DINOv2Wrapper, DINOv3Wrapper, ViTWrapper, get_model

GRID = (16, 16)


def wrapper(cls=DINOv2Wrapper):
    """Backbone wrapper without loading model weights (the tested methods do not need them)."""
    w = object.__new__(cls)
    w.layers, w.last_layer_features = None, None  # as set by VisionTransformerWrapper.__init__
    return w


def block_features(block, dim=8, seed=0):
    """Synthetic patch features: patches inside 'block' (row/col slices) differ strongly from the background."""
    rng = np.random.default_rng(seed)
    foreground = np.zeros(GRID, dtype=bool)
    foreground[block] = True
    features = rng.normal(scale=0.5, size=(GRID[0] * GRID[1], dim))
    features[:, 0] += np.where(foreground.ravel(), 50.0, -50.0)
    return features.astype(np.float32), foreground


@pytest.mark.parametrize("masking_type", [False, None])
def test_no_masking_keeps_all_patches(masking_type):
    features, _ = block_features((slice(4, 12), slice(4, 12)))
    mask = wrapper().compute_background_mask(features, GRID, masking_type=masking_type)
    assert mask.shape == (GRID[0] * GRID[1],) and mask.all()


@pytest.mark.parametrize("seed", range(3))
def test_masking_keeps_centered_object(seed):
    # the sign of the first PCA component is arbitrary, the center check must select the object in any case
    features, foreground = block_features((slice(4, 12), slice(4, 12)), seed=seed)
    mask = wrapper().compute_background_mask(features, GRID, masking_type=True).reshape(GRID)
    assert mask[foreground].all()
    assert not mask[0, 0] and not mask[-1, -1]


def test_mask_postprocessing_current_behavior():
    """
    Documents the current post-processing (results in the paper are based on it): dilation/closing are applied
    to the flattened mask, i.e., the mask is enlarged horizontally (along rows), but not vertically.
    Change this test deliberately if the post-processing is changed.
    """
    features, foreground = block_features((slice(4, 12), slice(4, 12)))
    mask = wrapper().compute_background_mask(features, GRID, masking_type=True).reshape(GRID)
    assert mask[4:12, 3].all() and mask[4:12, 12].all()      # enlarged to the left and right
    assert not mask[3, 4:12].any() and not mask[12, 4:12].any()  # not enlarged above and below


@pytest.mark.parametrize("cls", [ViTWrapper, DINOv3Wrapper])
def test_no_masking_support(cls):
    features, _ = block_features((slice(4, 12), slice(4, 12)))
    assert wrapper(cls).compute_background_mask(features, GRID, masking_type=True).all()


def test_unknown_model_names():
    with pytest.raises(ValueError):
        get_model("resnet50", "cpu")
    with pytest.raises(ValueError, match="Unknown model name"):
        get_model("dinov3_vits14", "cpu")  # checked before anything is downloaded


class FakeTimmViT(torch.nn.Module):
    """Mimics the timm DINOv3 output: class token + 4 register tokens + patch tokens."""
    num_prefix_tokens = 5

    def forward_features(self, x):
        n_patches = (x.shape[2] // 16) * (x.shape[3] // 16)
        prefix = torch.full((1, self.num_prefix_tokens, 4), -1.0)
        patches = torch.arange(n_patches, dtype=torch.float32)[None, :, None].expand(1, n_patches, 4)
        return torch.cat([prefix, patches], dim=1)


def test_dinov3_feature_extraction():
    w = wrapper(DINOv3Wrapper)
    w.model, w.device, w.patch_size = FakeTimmViT(), "cpu", 16
    w.transform = lambda img: torch.zeros(3, 100, 70)  # stands in for resizing/normalization
    image_tensor, grid_size = w.prepare_image(np.zeros((100, 70, 3), dtype=np.uint8))
    assert image_tensor.shape == (3, 96, 64) and grid_size == (6, 4)  # cropped to a multiple of the patch size
    features = w.extract_features(image_tensor)
    assert features.shape == (24, 4) and features.dtype == np.float32
    np.testing.assert_array_equal(features[:, 0], np.arange(24))  # class/register tokens removed


@pytest.mark.parametrize("cls", [DINOv2Wrapper, ViTWrapper])
@pytest.mark.parametrize("use_mask", [False, True])
def test_embedding_visualization(cls, use_mask):
    features, foreground = block_features((slice(4, 12), slice(4, 12)))
    vis = wrapper(cls).get_embedding_visualization(features, GRID, foreground.ravel() if use_mask else None)
    assert vis.shape == (*GRID, 3)
    assert vis.min() >= 0.0 and vis.max() <= 1.0


class FakeBlocksModel(torch.nn.Module):
    """Stand-in for DINOv2 (get_intermediate_layers) and timm DINOv3 (forward_intermediates) with 3 blocks:
    the patch features of block b are b + 1 + arange(N)/N in feature 0 and b + 1 elsewhere."""
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])

    def _block(self, b, n_patches=24):
        x = torch.full((1, n_patches, 4), float(b + 1))
        x[0, :, 0] += torch.arange(n_patches) / n_patches
        return x

    def get_intermediate_layers(self, x, n=1, norm=True):
        return tuple(self._block(b) for b in n)

    def forward_intermediates(self, x, indices, norm, output_fmt, intermediates_only):
        assert norm and output_fmt == "NLC" and intermediates_only
        return [self._block(b) for b in indices]


@pytest.mark.parametrize("cls", [DINOv2Wrapper, DINOv3Wrapper])
def test_layer_selection(cls):
    w = wrapper(cls)
    w.model, w.device, w.model_name = FakeBlocksModel(), "cpu", "fake"
    x = torch.zeros(3, 64, 96)
    w.layers = [1]
    np.testing.assert_allclose(w.extract_features(x), w.model._block(1)[0].numpy())
    np.testing.assert_allclose(w.last_layer_features, w.model._block(2)[0].numpy())  # kept for masking
    w.layers = [0, -2]  # several blocks: mean of L2-normalized features (here: block 0 and block 1)
    n = lambda t: torch.nn.functional.normalize(t[0], dim=-1)
    np.testing.assert_allclose(w.extract_features(x), ((n(w.model._block(0)) + n(w.model._block(1))) / 2).numpy(), rtol=1e-6)
    w.layers = [3]
    with pytest.raises(ValueError, match="out of range"):
        w.extract_features(x)


def test_masking_with_layers_uses_last_block():
    w = wrapper(DINOv2Wrapper)
    w.layers = [5]
    last, foreground = block_features((slice(4, 12), slice(4, 12)))
    w.last_layer_features = last
    noise = np.random.default_rng(0).normal(size=last.shape).astype(np.float32)  # intermediate features: no structure
    mask = w.compute_background_mask(noise, GRID, masking_type=True).reshape(GRID)
    assert mask[foreground].all() and not mask[0, 0]
    w.last_layer_features = last[:10]  # not from the same image -> error instead of a wrong mask
    with pytest.raises(RuntimeError):
        w.compute_background_mask(noise, GRID, masking_type=True)


def test_vit_rejects_layers():
    with pytest.raises(NotImplementedError):
        ViTWrapper("vit_b_16", "cpu", layers=[3])  # raised before any weights are downloaded
