import cv2
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import numpy as np


# Base Wrapper Class
class VisionTransformerWrapper:
    MODELS = ()  # supported model names, set in the subclasses

    def __init__(self, model_name, device, smaller_edge_size=224, layers=None):
        self.device = device
        self.smaller_edge_size = smaller_edge_size
        self.model_name = model_name
        self.layers = layers                # blocks to take patch features from (None: last block)
        self.last_layer_features = None     # last-block features of the last extract_features call (used for masking)
        self.model = self.load_model()

    def load_model(self):
        raise NotImplementedError("This method should be overridden in a subclass")

    def prepare_image(self, img):
        raise NotImplementedError("This method should be overridden in a subclass")

    def extract_features(self, img_tensor):
        raise NotImplementedError("This method should be overridden in a subclass")

    @staticmethod
    def _load_image(img):
        """img: file path, numpy array or PIL image -> PIL image."""
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
        return img

    def compute_background_mask(self, img_features, grid_size, threshold=10, masking_type=False, **kwargs):
        # No masking by default (only supported for DINOv2), all patches are kept
        return np.ones(img_features.shape[0], dtype=bool)

    def compute_background_mask_from_image(self, image, threshold=10, masking_type=None):
        image_tensor, grid_size = self.prepare_image(image)
        tokens = self.extract_features(image_tensor)
        return self.compute_background_mask(tokens, grid_size, threshold, masking_type)

    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens


# torchvision ViT Wrapper (supervised ImageNet weights, fixed input size 224 x 224)
class ViTWrapper(VisionTransformerWrapper):
    MODELS = ("vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32")

    def load_model(self):
        if self.layers is not None:
            raise NotImplementedError("Selecting layers is only supported for DINOv2 and DINOv3 backbones.")
        weights = models.get_model_weights(self.model_name).DEFAULT
        model = models.get_model(self.model_name, weights=weights)
        model.eval()
        self.transform = weights.transforms()
        self.grid_size = (model.image_size // model.patch_size,) * 2
        return model.to(self.device)

    def prepare_image(self, img):
        img_tensor = self.transform(self._load_image(img)).unsqueeze(0)
        return img_tensor, self.grid_size

    def extract_features(self, img_tensor):
        with torch.inference_mode():
            img_tensor = img_tensor.to(self.device)
            patches = self.model._process_input(img_tensor)
            class_token = self.model.class_token.expand(patches.size(0), -1, -1)
            patches = torch.cat((class_token, patches), dim=1)
            patch_features = self.model.encoder(patches)
            tokens = patch_features[0, 1:]  # exclude the class token
        return tokens.float().cpu().numpy()


# DINOv2 Wrapper (models are loaded via torch.hub)
class DINOv2Wrapper(VisionTransformerWrapper):
    MODELS = ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14")

    def load_model(self):
        model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        model.eval()
        self.patch_size = model.patch_size
        self.transform = self._dino_transform()
        return model.to(self.device)

    def _dino_transform(self):
        """Resize the smaller edge, normalize with the ImageNet defaults (same for DINOv2 and DINOv3)."""
        return transforms.Compose([
            transforms.Resize(size=self.smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def prepare_image(self, img):
        image_tensor = self.transform(self._load_image(img))
        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.patch_size, height - height % self.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.patch_size, cropped_width // self.patch_size)
        return image_tensor, grid_size

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            if self.layers is None:
                tokens = self.model.get_intermediate_layers(image_batch)[0][0] # last block (normalized), patch tokens only
            else:
                blocks = self._block_indices()
                outputs = self.model.get_intermediate_layers(image_batch, n=blocks) # with final norm
                tokens = self._combine_layers(dict(zip(blocks, (out[0] for out in outputs))))
        return tokens.float().cpu().numpy()

    def _block_indices(self):
        """Blocks to extract: the selected ones (0-based, negative counts from the end) and the last one (for masking)."""
        n_blocks = len(self.model.blocks)
        for layer in self.layers:
            if not -n_blocks <= layer < n_blocks:
                raise ValueError(f"Layer {layer} out of range for {self.model_name} with {n_blocks} blocks.")
        return sorted(set([layer % n_blocks for layer in self.layers] + [n_blocks - 1]))

    def _combine_layers(self, outputs):
        """
        outputs: dict block index -> patch tokens (N, D). Returns the features of the selected block, or the mean of the
        L2-normalized features if several blocks are selected. Stores the last-block features for masking.
        """
        n_blocks = len(self.model.blocks)
        self.last_layer_features = outputs[n_blocks - 1].float().cpu().numpy()
        selected = [layer % n_blocks for layer in self.layers]
        if len(selected) == 1:
            return outputs[selected[0]]
        return torch.stack([torch.nn.functional.normalize(outputs[b].float(), dim=-1) for b in selected]).mean(dim=0)

    def compute_background_mask(self, img_features, grid_size, threshold=10, masking_type=False, kernel_size=3, border=0.2):
        # Kernel size for morphological operations should be odd
        if not masking_type:
            return np.ones(img_features.shape[0], dtype=bool)  # no masking, all patches are kept (skip the PCA)
        if self.layers is not None:
            # the threshold is calibrated on last-block features -> use those of the same image (from the preceding extract_features call)
            if self.last_layer_features is None or self.last_layer_features.shape[0] != img_features.shape[0]:
                raise RuntimeError("With selected layers, compute_background_mask must directly follow extract_features of the same image.")
            img_features = self.last_layer_features
        pca = PCA(n_components=1, svd_solver='randomized')
        first_pc = pca.fit_transform(img_features.astype(np.float32))
        mask = first_pc > threshold
        # test whether the center crop of the images is kept (adaptive masking), adapt if your objects of interest are not centered!
        m = mask.reshape(grid_size)[int(grid_size[0] * border):int(grid_size[0] * (1-border)), int(grid_size[1] * border):int(grid_size[1] * (1-border))]
        if m.sum() <=  m.size * 0.35:
            mask = - first_pc > threshold
        # postprocess mask, fill small holes in the mask, enlarge slightly
        # (note: applied to the flattened (N, 1) mask, i.e. only along the rows of the patch grid -- kept as in the paper for reproducibility)
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        return mask.squeeze()


# DINOv3 Wrapper (models are loaded via timm, weights are downloaded from the Hugging Face hub)
class DINOv3Wrapper(DINOv2Wrapper):
    """
    DINOv3 ViTs (pretrained on LVD-1689M), image preparation as for DINOv2 (resize smaller edge, crop to multiple of patch size).
    Zero-shot masking is not used/available for DINOv3.
    """
    MODELS = {  # model name -> timm name
        "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
        "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
        "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m",
    }

    def load_model(self):
        import timm # only required for DINOv3
        model = timm.create_model(self.MODELS[self.model_name], pretrained=True, dynamic_img_size=True)
        model.eval()
        self.patch_size = model.patch_embed.patch_size[0]
        self.transform = self._dino_transform()
        return model.to(self.device)

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            if self.layers is None:
                tokens = self.model.forward_features(image_batch) # final layer (normalized), incl. class and register tokens
                tokens = tokens[0, self.model.num_prefix_tokens:] # keep patch tokens only
            else:
                blocks = self._block_indices()
                outputs = self.model.forward_intermediates(image_batch, indices=blocks, norm=True, output_fmt='NLC',
                                                           intermediates_only=True) # patch tokens only, with final norm
                tokens = self._combine_layers(dict(zip(blocks, (out[0] for out in outputs))))
        return tokens.float().cpu().numpy()

    def compute_background_mask(self, img_features, grid_size, threshold=10, masking_type=False, **kwargs):
        # No masking for DINOv3, all patches are kept
        return VisionTransformerWrapper.compute_background_mask(self, img_features, grid_size)


WRAPPERS = (ViTWrapper, DINOv2Wrapper, DINOv3Wrapper)


def get_model(model_name, device, smaller_edge_size=448, layers=None):
    """
    layers: blocks to take the patch features from (0-based, negative counts from the end), default None = last block.
    Several blocks are averaged (after L2 normalization). Masking always uses the last block.
    """
    wrapper = next((w for w in WRAPPERS if model_name in w.MODELS), None)
    if wrapper is None:
        raise ValueError(f"Unknown model name: {model_name}. Choose from {[m for w in WRAPPERS for m in w.MODELS]}.")

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"Smaller edge size: {smaller_edge_size}")
    if layers is not None:
        print(f"Layers: {layers}")
    return wrapper(model_name, device, smaller_edge_size, layers=layers)
