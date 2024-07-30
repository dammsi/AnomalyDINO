import matplotlib.pyplot as plt
import os 
import cv2
import numpy as np
from tqdm import tqdm
import faiss
from scipy.ndimage import gaussian_filter
import tifffile as tiff

from src.utils import augment_image, dists2map, plot_ref_images


def run_anomaly_detection(dm,
                        object_name, 
                        data_root,
                        n_ref_samples,
                        object_anomalies,
                        plots_dir,
                        save_examples = True,
                        masking = None,
                        rotation = False,
                        knn_metric = 'L2_normalized',
                        knn_neighbors = 1, 
                        seed = 0,
                        save_patch_dists = True,
                        save_tiffs = False):
    """
    Main function to evaluate the anomaly detection performance of a given object/product.

    Parameters:
    - dm: The DINO model for feature extraction and masking.
    - object_name: The name of the object/product to evaluate.
    - data_root: The root directory of the dataset.
    - n_ref_samples: The number of reference samples to use for evaluation (k-shot). Set to -1 for full-shot setting.
    - object_anomalies: The anomaly types for each object/product.
    - plots_dir: The directory to save the example plots.
    - save_examples: Whether to save example images and plots. Default is True.
    - masking: Whether to apply DINOv2 to estimate the foreground mask (and discard background patches).
    - rotation: Whether to augment reference samples with rotation.
    - knn_metric: The metric to use for kNN search. Default is 'L2_normalized' (1 - cosine similarity)
    - knn_neighbors: The number of nearest neighbors to consider. Default is 1.
    - seed: The seed value for deterministic sampling in few-shot setting. Default is 0.
    - save_patch_dists: Whether to save the patch distances. Default is True. Required to eval detection.
    - save_tiffs: Whether to save the anomaly maps as TIFF files. Default is False. Required to eval segmentation.
    """

    assert knn_metric in ["L2", "L2_normalized"]
    
    # add 'good' to the anomaly types
    type_anomalies = object_anomalies[object_name]
    type_anomalies.append('good')

    # ensure that each type is only evaluated once
    type_anomalies = list(set(type_anomalies))

    # Extract reference features
    features_ref = []
    images_ref = []
    masks_ref = []
    vis_backgroud = []

    img_ref_folder = f"{data_root}/{object_name}/train/good/"
    if n_ref_samples == -1:
        # full-shot setting
        img_ref_samples = sorted(os.listdir(img_ref_folder))
    else:
        # few-shot setting, pick samples in deterministic fashion according to seed
        img_ref_samples = sorted(os.listdir(img_ref_folder))[seed*n_ref_samples:(seed + 1)*n_ref_samples]

    if len(img_ref_samples) < n_ref_samples:
        print(f"Warning: Not enough reference samples for {object_name}! Only {len(img_ref_samples)} samples available.")
        
    for img_ref_n in img_ref_samples:
        # load reference image...
        img_ref = f"{img_ref_folder}{img_ref_n}"
        image_ref = cv2.cvtColor(cv2.imread(img_ref, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # augment reference image (if applicable)...
        if rotation:
            img_augmented = augment_image(image_ref)
        else:
            img_augmented = [image_ref]
        for i in range(len(img_augmented)):
            image_ref = img_augmented[i]
            image_ref_tensor, grid_size1, _ = dm.prepare_image(image_ref)
            features_ref_i = dm.extract_features(image_ref_tensor)
            features_ref.append(features_ref_i)
            if save_examples:
                images_ref.append(image_ref)
                mask_ref = dm.compute_background_mask(features_ref_i, grid_size1, threshold=10, masking_type=masking)
                vis_image_background = dm.get_embedding_visualization(features_ref_i, grid_size1, mask_ref)
                masks_ref.append(mask_ref)
                vis_backgroud.append(vis_image_background)

    # plot some reference samples for inspection
    if save_examples:
        plots_dir_ = f"{plots_dir}/{object_name}/"
        plot_ref_images(images_ref, masks_ref, vis_backgroud, grid_size1, plots_dir_, title = "Reference Images", img_names = img_ref_samples)   
    
    # set up the KNN index
    if knn_metric == "L2":
        features_ref = np.concatenate(features_ref, axis=0)
        knn_index1 = faiss.IndexFlatL2(features_ref.shape[1])
        knn_index1.add(features_ref)
    elif knn_metric == "L2_normalized":
        features_ref = np.concatenate(features_ref, axis=0)
        knn_index1 = faiss.IndexFlatL2(features_ref.shape[1])
        faiss.normalize_L2(features_ref)
        knn_index1.add(features_ref)

    idx = 0
    # Evaluate anomalies for each anomaly type (and "good")
    for type_anomaly in tqdm(type_anomalies, desc = f"processing test samples ({object_name})"):
        data_dir = f"{data_root}/{object_name}/test/{type_anomaly}"
        
        if save_patch_dists or save_tiffs:
            os.makedirs(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}", exist_ok=True)
        
        for idx, img_test_nr in enumerate(sorted(os.listdir(data_dir))):
            image_test_path = f"{data_dir}/{img_test_nr}"

            # Extract test features
            image_test = cv2.cvtColor(cv2.imread(image_test_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            image_tensor2, grid_size2, resize_scale2 = dm.prepare_image(image_test)
            features2 = dm.extract_features(image_tensor2)

            # Compute background mask
            if masking:
                mask2 = dm.compute_background_mask(features2, grid_size2, threshold=10, masking_type=masking)
            else:
                mask2 = np.ones(features2.shape[0], dtype=bool)
            if idx < 3:
                vis_image_test_background = dm.get_embedding_visualization(features2, grid_size2, mask2)

            # Compute distances to nearest neighbors in M
            if knn_metric == "L2":
                distances, match2to1 = knn_index1.search(features2, k = knn_neighbors)
                if knn_neighbors >= 1:
                    distances = distances.mean(axis=1)
                distances = np.sqrt(distances)

            elif knn_metric == "L2_normalized":
                faiss.normalize_L2(features2) 
                distances, match2to1 = knn_index1.search(features2, k = knn_neighbors)
                if knn_neighbors >= 1:
                    distances = distances.mean(axis=1)
                distances = distances / 2   # equivalent to cosine distance (1 - cosine similarity)

            d_masked = distances.reshape(grid_size2)
            d_masked[~mask2.reshape(grid_size2)] = 0.0 # set distances of masked out patches to 0
            
            # Save the anomaly maps (raw as .npy or full resolution .tiff files)
            img_test_nr = img_test_nr.split(".")[0]
            if save_tiffs:
                anomaly_map = dists2map(d_masked, image_test.shape)
                tiff.imwrite(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}/{img_test_nr}.tiff", anomaly_map)
            if save_patch_dists:
                np.save(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}/{img_test_nr}.npy", d_masked)

            # Save some example plots (3 per anomaly type)
            if save_examples and idx < 3:

                fig, (ax1, ax2, ax3, ax4,) = plt.subplots(1, 4, figsize=(18, 4.5))

                # plot test image, PCA + mask
                ax1.imshow(image_test)
                ax2.imshow(vis_image_test_background)  

                # plot patch distances 
                d_masked[~mask2.reshape(grid_size2)] = 0.0
                plt.colorbar(ax3.imshow(d_masked), ax=ax3, fraction=0.12, pad=0.05, orientation="horizontal")
                
                # compute image level anomaly score (mean(top 1%) of patches = empirical tail value at risk for quantile 0.99)
                score_top1p = np.mean(sorted(distances, reverse = True)[:int(len(distances) * 0.01)])
                ax4.axvline(score_top1p, color='r', linestyle='dashed', linewidth=1, label=round(score_top1p, 2))
                ax4.legend()
                ax4.hist(distances.flatten()[mask2])

                ax1.axis('off')
                ax2.axis('off')
                ax3.axis('off')

                ax1.title.set_text("Test")
                ax2.title.set_text("Test (PCA + Mask)")
                ax3.title.set_text("Patch Distances (1NN)")
                ax4.title.set_text("Hist of Distances")

                plt.suptitle(f"Object: {object_name}, Type: {type_anomaly}, img = ...{image_test_path[-20:]}, object patches = {mask2.sum()}/{features2.shape[0]}")

                plt.tight_layout()
                plt.savefig(f"{plots_dir}/{object_name}/examples/example_{type_anomaly}_{idx}.png")
                plt.close()