from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from .utils import get_dataset_info, dists2map

from  matplotlib.colors import LinearSegmentedColormap
neon_violet = (0.5, 0.1, 0.5, 0.4)
neon_yellow = (0.8, 1.0, 0.02, 0.7)
red_gt = (1.0, 0, 0.0, 0.5)
colors = [(1.0, 1, 1.0, 0.0),  neon_violet, neon_yellow]
cmap = LinearSegmentedColormap.from_list("AnomalyMap", colors, N=256)


def get_test_gt_map(object_name, anomaly_type, img_nr, experiment, data_root, dataset = "MVTec", good=False):
    """
    Return test sample, ground truth (if not a good sample) and anomaly maps for given experiment and img_nr.
    """ 
    # test sample
    img_test_path = f"{data_root}/{object_name}/test/{anomaly_type}/{img_nr}"
    image_test = cv2.cvtColor(cv2.imread(img_test_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    
    img_nr = img_nr.split(".")[0]

    # ground truth
    if not good:
        gt_path = f"{data_root}/{object_name}/ground_truth/{anomaly_type}/{img_nr}" + ("_mask.png" if dataset == "MVTec" else ".png")
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    # load patch distances for test sample
    dists = np.load(f"{experiment}/{object_name}/test/{anomaly_type}/{img_nr}.npy")
    
    # anomaly maps
    anomaly_map = dists2map(dists, image_test.shape)
    if good:
        return image_test, anomaly_map
    else:
      return image_test, gt_mask, anomaly_map


def plot_sample(image_test, anomaly_map, axs, cmap, vmax):
    axs.imshow(image_test)
    axs.imshow(anomaly_map, cmap=cmap, vmax=vmax)


def infer_vmax(exp_path, objects):
    vmax = {}
    for object_name in objects:
        current_max = 0
        for test_file_good in os.listdir(f"{exp_path}/{object_name}/test/good/"):
            if test_file_good.endswith(".npy"):
                max_score = np.load(f"{exp_path}/{object_name}/test/good/{test_file_good}").max()
                current_max = max(current_max, max_score)

        vmax[object_name] = current_max * 1.0
    return vmax


def create_sample_plots(experiment_path, anomaly_maps_dir, seed, dataset, data_root):
    # infer objects and anomalies, preprocessing does not matter
    objects, object_anomalies, _, _ = get_dataset_info(dataset, preprocess = "informed")
    # infer vmax for each object
    vmax = infer_vmax(anomaly_maps_dir, objects)

    for object_name in tqdm(objects, desc="Plot anomaly maps"):
        n = len(object_anomalies[object_name])
        fig, axs = plt.subplots(n + 1, 5, figsize=(2 * 5, 2* (n + 1)))

        for i, anomaly_type in enumerate(object_anomalies[object_name]):
            # plot five test samples with anomaly maps
            first_five_samples = sorted(os.listdir(f"{data_root}/{object_name}/test/{anomaly_type}/"))[:5]
            for j, img_nr in enumerate(first_five_samples):
                image_test, gt_mask, anomaly_map = get_test_gt_map(object_name, anomaly_type,
                                                                    img_nr, anomaly_maps_dir, dataset = dataset, data_root = data_root)
                plot_sample(image_test, anomaly_map, axs[i, j], cmap=cmap, vmax=vmax[object_name])
                axs[i, j].axis('off')
                if j == 2:
                    axs[i, j].set_title(f"anomaly type: {anomaly_type}")

        first_five_good_samples = sorted(os.listdir(f"{data_root}/{object_name}/test/good/"))[:5]
        for j, img_nr in enumerate(first_five_good_samples):
            # plot five good test samples with anomaly maps for comparison
            image_test, anomaly_map = get_test_gt_map(object_name, "good", img_nr, 
                                                      anomaly_maps_dir, dataset = dataset, data_root = data_root, good=True)
            axs[n, j].imshow(image_test)
            axs[n, j].imshow(anomaly_map, cmap=cmap, vmax=vmax[object_name])
            axs[n, j].axis('off')
            if j == 2:
                axs[n, j].set_title(f"good test samples (for comparison)")

        plt.tight_layout()
        plt.savefig(f"{experiment_path}/{object_name}/anomaly_maps_examples_seed={seed}.png")
        plt.close()
