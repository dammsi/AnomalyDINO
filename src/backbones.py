import cv2
import torch
import torchvision.models as models
# import clip
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import numpy as np


# Base Wrapper Class
class VisionTransformerWrapper:
    def __init__(self, model_name, device, smaller_edge_size=224, half_precision=False):
        self.device = device
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        raise NotImplementedError("This method should be overridden in a subclass")
    
    def extract_features(self, img_tensor):
        raise NotImplementedError("This method should be overridden in a subclass")


# ViT-B/16 Wrapper
class ViTWrapper(VisionTransformerWrapper):
    def load_model(self):
        if self.model_name == "vit_b_16":
            model = models.vit_b_16(weights = models.ViT_B_16_Weights.DEFAULT)
            self.transform = models.ViT_B_16_Weights.DEFAULT.transforms()
            self.grid_size = (14,14)
        elif self.model_name == "vit_b_32":
            model = models.vit_b_32(weights = models.ViT_B_32_Weights.DEFAULT)
            self.transform = models.ViT_B_32_Weights.DEFAULT.transforms()
            self.grid_size = (7,7)
        elif self.model_name == "vit_l_16":
            model = models.vit_l_16(weights = models.ViT_L_16_Weights.DEFAULT)
            self.transform = models.ViT_L_16_Weights.DEFAULT.transforms()
            self.grid_size = (14,14)
        elif self.model_name == "vit_l_32":
            model = models.vit_l_32(weights = models.ViT_L_32_Weights.DEFAULT)
            self.transform = models.ViT_L_32_Weights.DEFAULT.transforms()
            self.grid_size = (7,7)
        else:
            raise ValueError(f"Unknown ViT model name: {self.model_name}")
        
        model.eval()
        # print(self.transform)

        return model.to(self.device)
    
    def prepare_image(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor, self.grid_size

    def extract_features(self, img_tensor):
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            patches = self.model._process_input(img_tensor)
            class_token = self.model.class_token.expand(patches.size(0), -1, -1)
            patches = torch.cat((class_token, patches), dim=1)
            patch_features = self.model.encoder(patches)
            return patch_features[:, 1:, :].squeeze().cpu().numpy()  # Exclude the class token

    def get_embedding_visualization(self, tokens, grid_size = (14,14), resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*self.grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens

    def compute_background_mask(self, img_features, grid_size, threshold = 10, masking_type = False):
        # No masking for ViT supported at the moment... (Only DINOv2)
        return np.ones(img_features.shape[0], dtype=bool)
    

# DINOv2 Wrapper
class DINOv2Wrapper(VisionTransformerWrapper):
    def load_model(self):
        model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        model.eval()

        # print(f"Loaded model: {self.model_name}")
        # print("Resizing images to", self.smaller_edge_size)

        # Set transform for DINOv2
        self.transform = transforms.Compose([
            transforms.Resize(size=self.smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
            ])
        
        return model.to(self.device)
    
    def prepare_image(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        image_tensor = self.transform(img)
        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size
    

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)

            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        return tokens.cpu().numpy()


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


    def compute_background_mask_from_image(self, image, threshold = 10, masking_type = None):
        image_tensor, grid_size = self.prepare_image(image)
        tokens = self.extract_features(image_tensor)
        return self.compute_background_mask(tokens, grid_size, threshold, masking_type)


    def compute_background_mask(self, img_features, grid_size, threshold = 10, masking_type = False, kernel_size = 3, border = 0.2):
        # Kernel size for morphological operations should be odd
        pca = PCA(n_components=1, svd_solver='randomized')
        first_pc = pca.fit_transform(img_features.astype(np.float32))
        if masking_type == True:
            mask = first_pc > threshold
            # test whether the center crop of the images is kept (adaptive masking), adapt if your objects of interest are not centered!
            m = mask.reshape(grid_size)[int(grid_size[0] * border):int(grid_size[0] * (1-border)), int(grid_size[1] * border):int(grid_size[1] * (1-border))]
            if m.sum() <=  m.size * 0.35:
                mask = - first_pc > threshold
            # postprocess mask, fill small holes in the mask, enlarge slightly
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        elif masking_type == False:
            mask = np.ones_like(first_pc, dtype=bool)
        return mask.squeeze()


def get_model(model_name, device, smaller_edge_size=448):
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"Smaller edge size: {smaller_edge_size}")

    if model_name.startswith("vit"):
        return ViTWrapper(model_name, device, smaller_edge_size)
    elif model_name.startswith("dinov2"):
        return DINOv2Wrapper(model_name, device, smaller_edge_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")