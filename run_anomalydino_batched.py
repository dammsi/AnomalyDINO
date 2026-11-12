import argparse
import os
import random
import time

from src.backbones import get_model
from src.utils import get_dataset_info, list_images, limit_cpu_threads, PREPROCESS_OPTIONS
from src.post_eval import mean_top1p

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import cv2
import yaml
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap


def parse_args():
    parser = argparse.ArgumentParser(description="Batched zero-shot anomaly detection with AnomalyDINO.")
    parser.add_argument("--dataset", type=str, default="MVTec",
                        help="Dataset name: 'MVTec', 'VisA', or any other name for a custom dataset (objects are then inferred from --data_root).")
    parser.add_argument("--data_root", type=str, default="data/mvtec_anomaly_detection",
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--layer", nargs='+', type=int, default=None,
                        help="Transformer block(s) to take the patch features from (0-based, negative counts from the end). "
                             "Several blocks are averaged after L2 normalization. Default: last block. Masking always uses the last block.")
    parser.add_argument("--model_size", type=str.lower, default="s", choices=["s", "b", "l", "g"],
                        help="Size of the DINOv2 backbone (small, base, large, giant). Ignored if --model_name is given.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Backbone name, e.g., 'dinov2_vits14' or 'dinov3_vits16' (overrides --model_size). "
                             "Masking is only applied for DINOv2.")
    parser.add_argument("--resolution", type=int, default=448,
                        help="Size of the smaller image edge (in pixels) after resizing.")
    parser.add_argument("--preprocess", type=str, default="masking_only", choices=PREPROCESS_OPTIONS,
                        help="Preprocessing setting, see 'get_dataset_info' in src/utils.py (only masking is used in the batched setting).")
    parser.add_argument("--save_examples", default=True, action=argparse.BooleanOptionalAction, help="Save example plots.")
    parser.add_argument("--device", default='cuda:0', help="CUDA device to run on, e.g., 'cuda:0'.")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Max. CPU threads per library (PyTorch, NumPy/scikit-learn, OpenCV, faiss); 0 = no limit (one per core). "
                             "Limiting is much faster when several runs share a machine.")

    args = parser.parse_args()
    return args

def dists_to_score(dists):
    # mean top 1% = empirical tail value at risk (for 99% quantile)
    return mean_top1p(dists)

def calculate_cosine_distances(features_all_tensors, sample_index, quantile = 0.001):
    """
    Calculate the cosine distances on patch level between the sample with index sample_index and all other samples in the list of
    features_all_tensors (patch features already on the GPU).
    The distance of a test patch to all reference patches is calculated as 1 - cosine_similarity,
    then the mean of the closest 0.1% of patches is returned as the score for this test patch. 
    """
    with torch.no_grad():
        sample_features = features_all_tensors[sample_index]

        all_features = torch.cat([features for i, features in enumerate(features_all_tensors) if i != sample_index])

        normalized_sample = torch.nn.functional.normalize(sample_features, dim=1)
        normalized_all = torch.nn.functional.normalize(all_features, dim=1)
        cosine_similarity = torch.mm(normalized_sample, normalized_all.t())
        cosine_distance = 1 - cosine_similarity

        # delete some tensors to free up GPU memory (for VisA memory consumption is quite high)
        del cosine_similarity, normalized_sample, normalized_all, all_features, sample_features

        # Calculate the mean (per patch) of all distances below the given quantile (default: 0.1%)
        quantile_distances = torch.kthvalue(cosine_distance, int(quantile * cosine_distance.shape[1]), dim=1).values.unsqueeze(1)
        mask_lowest = cosine_distance < quantile_distances
        means_below_quantile = (cosine_distance * mask_lowest).sum(dim=1) / mask_lowest.sum(dim=1)
    return means_below_quantile.cpu().numpy()


def evaluate_ad_batched(model,
                        objects,
                        object_anomalies,
                        data_root, 
                        plots_dir,
                        masking_default,
                        device,
                        save_examples = True,
                        cmap = None):
    AUROCs = {}
    
    inference_times = []

    for object_name in tqdm(objects, position=0, leave=True, desc="Evaluating objects"):

        type_anomalies = object_anomalies[object_name] + ['good'] # new list, to not modify the dataset info

        # ensure that each type is only evaluated once
        type_anomalies = list(set(type_anomalies))

        # order type anomalies alphabetically, but with 'good' at the front
        type_anomalies = sorted(type_anomalies, key=lambda x: (x != 'good', x))

        # Extract reference features
        imgs_all = []
        features_all = []
        masks_ref = []
        gt_label = []
        grid_sizes = []

        img_test_folder = f"{data_root}/{object_name}/test/"

        with torch.inference_mode():
            # read in all test samples (with label information for later evaluation)
            for type_anomaly in type_anomalies:
                data_dir = img_test_folder + f"{type_anomaly}"
                for img_test_nr in tqdm(list_images(data_dir), desc=f"Load test set: {object_name} ({type_anomaly})", leave=False):
                    img_test = f"{data_dir}/{img_test_nr}"
                    img_test = cv2.cvtColor(cv2.imread(img_test, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    image_tensor, grid_size = model.prepare_image(img_test)
                    features = model.extract_features(image_tensor)
                    mask_test = model.compute_background_mask(features, grid_size, threshold=10, masking_type=masking_default[object_name])
                    imgs_all.append(img_test)
                    features_all.append(features)
                    masks_ref.append(mask_test)
                    gt_label.append(type_anomaly)
                    grid_sizes.append(grid_size)

            # score test samples (mutual scoring), features are copied to the GPU once per object
            features_all_tensors = [torch.tensor(features, device=device, dtype=torch.float32) for features in features_all]
            test_dists = []
            for test_idx in tqdm(range(len(features_all)), desc = f"Score test set: {object_name}", leave=False):
                start_time = time.time()
                dists = calculate_cosine_distances(features_all_tensors, test_idx)
                dists[~masks_ref[test_idx]] = 0.0
                test_dists.append(dists)
                inference_times.append(time.time() - start_time)
            
            # compute AUROC ("good" vs. not "good")
            y_true = [0 if l == "good" else 1 for l in gt_label]
            y_scores = [dists_to_score(d) for d in test_dists]
            AUROCs[object_name] = roc_auc_score(y_true, y_scores)

            if save_examples:
                # plot anomaly maps of (up to) 5 random samples for each category (one row per category)
                rng = random.Random(0)
                vmax = np.max(np.array(y_scores)[np.array(y_true) == 0]) * 1.2
                fig, axes = plt.subplots(len(type_anomalies), 5, figsize=(10, 2*len(type_anomalies)), squeeze=False)
                for row, type_anomaly in enumerate(type_anomalies):
                    indices = [i for i, l in enumerate(gt_label) if l == type_anomaly]
                    for ax in axes[row]:
                        ax.axis("off")
                    for col, sample_idx in enumerate(rng.sample(indices, min(5, len(indices)))):
                        ax = axes[row, col]
                        ax.imshow(imgs_all[sample_idx])
                        d = cv2.resize(test_dists[sample_idx].reshape(grid_sizes[sample_idx]), (imgs_all[sample_idx].shape[1], imgs_all[sample_idx].shape[0]), interpolation = cv2.INTER_LINEAR)
                        d = gaussian_filter(d, sigma=4)
                        ax.imshow(d, cmap = cmap, vmax=vmax)
                        ax.set_title(f"{gt_label[sample_idx]}: {dists_to_score(test_dists[sample_idx]):.2f}")
                plt.tight_layout()
                plt.savefig(f"{plots_dir}/{object_name}_examples.png")
                plt.close()

            # free GPU memory and empty CUDA cache
            del features_all_tensors
            torch.cuda.empty_cache()
            
    print(f"Average inference time per sample: {np.mean(inference_times):.5f} s/sample = {1.0/np.mean(inference_times):.2f} samples/s")
    return AUROCs

if __name__=="__main__":
    args = parse_args()
    limit_cpu_threads(args.num_threads)

    # set torch device
    torch.cuda.set_device(args.device)

    if args.model_name is None:
        args.model_name = "dinov2_vit" + args.model_size.lower() + "14"
    model = get_model(args.model_name, args.device, smaller_edge_size=args.resolution, layers=args.layer)
    objects, object_anomalies, masking_default, _ = get_dataset_info(args.dataset, args.preprocess, data_path=args.data_root)
    # zero-shot masking is calibrated for DINOv2 only (same as in run_anomalydino.py)
    if not args.model_name.startswith("dinov2"):
        masking_default = {o: False for o in objects}

    # colors for anomaly map
    neon_violet = (0.5, 0.1, 0.5, 0.4)
    neon_yellow = (0.8, 1.0, 0.02, 0.7)
    colors = [(1.0, 1, 1.0, 0.0),  neon_violet, neon_yellow]
    cmap = LinearSegmentedColormap.from_list("AnomalyMap", colors, N=256)

    model_dir = f"{args.model_name}_{args.resolution}" + (f"_layer{'+'.join(map(str, args.layer))}" if args.layer is not None else "")
    plot_dir = f"results_{args.dataset}/{model_dir}/batched-0-shot_{args.preprocess}"
    
    os.makedirs(plot_dir, exist_ok = True)

    print("Results will be saved to", plot_dir)

    # save args to yaml
    with open(f"{plot_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)
    
    AUROCs = evaluate_ad_batched(model,
                                 objects,
                                 object_anomalies,
                                 args.data_root, 
                                 plot_dir, save_examples = args.save_examples,
                                 device = args.device,
                                 masking_default = masking_default,
                                 cmap = cmap)
    
    df = pd.DataFrame.from_dict(AUROCs, orient='index', columns=['AUROC'])

    # compute mean over categories and save to file
    df.loc['MEAN'] = df.mean()
    df = df * 100
    df.to_csv(f"{plot_dir}/AUROCs.csv")
