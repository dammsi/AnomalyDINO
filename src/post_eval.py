""""
This evaluation scripts partly builds on the official evaluation script for MVTec-AD (original code available here: https://www.mvtec.com/company/research/datasets/mvtec-ad).
"""

import json
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
import numpy as np
import os
import numpy as np
import cv2
from os import makedirs, path, listdir
import argparse
from PIL import Image
from tqdm import tqdm
import tifffile as tiff
from scipy.ndimage import label
from bisect import bisect


def parse_dataset_files(object_name, dataset_base_dir, anomaly_maps_dir, dataset="MVTec"):
    """Parse the filenames for one object of the MVTec AD dataset.

    Args:
        object_name: Name of the dataset object.
        dataset_base_dir: Base directory of the MVTec AD dataset.
        anomaly_maps_dir: Base directory where anomaly maps are located.
    """
    # Store a list of all ground truth filenames.
    gt_filenames = []

    # Store a list of all corresponding anomaly map filenames.
    prediction_filenames = []

    # Test images are located here.
    test_dir = path.join(dataset_base_dir, object_name, 'test')
    gt_base_dir = path.join(dataset_base_dir, object_name, 'ground_truth')
    anomaly_maps_base_dir = path.join(anomaly_maps_dir, object_name, 'test')

    # List all ground truth and corresponding anomaly images.
    for subdir in listdir(str(test_dir)):

        if not subdir.replace('_', '').isalpha():
            continue

        # Get paths to all test images in the dataset for this subdir.
        test_images = [path.splitext(file)[0]
                       for file
                       in listdir(path.join(test_dir, subdir))
                       if path.splitext(file)[1] == ('.png' if dataset == "MVTec" else '.JPG')]

        # If subdir is not 'good', derive corresponding GT names.
        if subdir != 'good':
            gt_filenames.extend(
                [path.join(gt_base_dir, subdir, file + '_mask.png' if dataset == "MVTec" else file + '.png')
                 for file in test_images])
        else:
            # No ground truth maps exist for anomaly-free images.
            gt_filenames.extend([None] * len(test_images))

        # Fetch corresponding anomaly maps.
        prediction_filenames.extend(
            [path.join(anomaly_maps_base_dir, subdir, file)
             for file in test_images])

    print(f"Parsed {len(gt_filenames)} ground truth image files.")

    return gt_filenames, prediction_filenames


def trapezoid(x, y, x_max=None):
    """
    This function calculates the definit integral of a curve given by
    x- and corresponding y-values. In contrast to, e.g., 'numpy.trapz()',
    this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a
    warning.

    Args:
        x: Samples from the domain of the function to integrate
          Need to be sorted in ascending order. May contain the same value
          multiple times. In that case, the order of the corresponding
          y values will affect the integration with the trapezoidal rule.
        y: Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be
          determined by interpolating between its neighbors. Must not lie
          outside of the range of x.

    Returns:
        Area under the curve.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("WARNING: Not all x and y values passed to trapezoid(...)"
              " are finite. Will continue with only the finite values.")
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between
            # the last x[ins-1] and x_max. Since we do not know the exact value
            # of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) *
                                     (x_max - x[ins - 1]) /
                                     (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def read_tiff(file_path_no_ext, exts=('.tif', '.tiff', '.TIF', '.TIFF')):
    """Read a TIFF file from a given path without the TIFF extension.

    Args:
        file_path_no_ext: Path to the TIFF file without a file extension.
        exts: TIFF extensions to consider when searching for the file.

    Raises:
        FileNotFoundError: The given file path does not exist with any of the
          given extensions.
        IOError: The given file path exists with multiple of the given
          extensions.
    """
    # Get all file paths that exist
    file_paths = []

    for ext in exts:
        # Make sure the file path does not already end with a tiff extension.
        assert not file_path_no_ext.endswith(ext)
        file_path = file_path_no_ext + ext
        if os.path.exists(file_path):
            file_paths.append(file_path)

    if len(file_paths) == 0:
        raise FileNotFoundError('Could not find a file with a TIFF extension'
                                f' at {file_path_no_ext}')
    elif len(file_paths) > 1:
        raise IOError('Found multiple files with a TIFF extension at'
                      f' {file_path_no_ext}'
                      '\nPlease specify which TIFF extension to use via the'
                      ' `exts` parameter of this function.')

    return tiff.imread(file_paths[0])


def compute_pro(anomaly_maps, ground_truth_maps):
    """Compute the PRO curve for a set of anomaly maps with corresponding ground
    truth maps.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain a
          real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that
          contain binary-valued ground truth labels for each pixel.
          0 indicates that a pixel is anomaly-free.
          1 indicates that a pixel contains an anomaly.

    Returns:
        fprs: numpy array of false positive rates.
        pros: numpy array of corresponding PRO values.
    """

    print("Compute PRO curve...", end=" ")

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    num_ok_pixels = 0
    num_gt_regions = 0

    shape = (len(anomaly_maps),
             anomaly_maps[0].shape[0],
             anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    assert shape[0] * shape[1] * shape[2] < np.iinfo(fp_changes.dtype).max, \
        'Potential overflow when using np.cumsum(), consider using np.uint64.'

    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):

        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)
        num_gt_regions += n_components

        # Compute the mask that gives us all ok pixels.
        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        # Compute by how much the FPR changes when each anomaly score is
        # added to the set of positives.
        # fp_change needs to be normalized later when we know the final value
        # of num_ok_pixels -> right now it is only the change in the number of
        # false positives
        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        # Compute by how much the PRO changes when each anomaly score is
        # added to the set of positives.
        # pro_change needs to be normalized later when we know the final value
        # of num_gt_regions.
        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1. / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    # Flatten the numpy arrays before sorting.
    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    # Sort all anomaly scores.
    print(f"Sort {len(anomaly_scores_flat)} anomaly scores...")
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]

    # Info: np.take(a, ind, out=a) followed by b=a instead of
    # b=a[ind] showed to be more memory efficient.
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    del sort_idxs

    # Get the (FPR, PRO) curve values.
    np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
    fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
    np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
    fprs = fp_changes_sorted

    np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
    np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
    pros = pro_changes_sorted

    # Merge (FPR, PRO) points that occur together at the same threshold.
    # For those points, only the final (FPR, PRO) point should be kept.
    # That is because that point is the one that takes all changes
    # to the FPR and the PRO at the respective threshold into account.
    # -> keep_mask is True if the subsequent score is different from the
    # score at the respective position.
    # anomaly_scores_sorted = [7, 4, 4, 4, 3, 1, 1]
    # ->          keep_mask = [T, F, F, T, T, F]
    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    del anomaly_scores_sorted

    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    del keep_mask

    # To mitigate the adding up of numerical errors during the np.cumsum calls,
    # make sure that the curve ends at (1, 1) and does not contain values > 1.
    np.clip(fprs, a_min=None, a_max=1., out=fprs)
    np.clip(pros, a_min=None, a_max=1., out=pros)

    # Make the fprs and pros start at 0 and end at 1.
    zero = np.array([0.])
    one = np.array([1.])

    return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one))


def mean_top1p(distances):
    if int(len(distances) * 0.01) == 0:
        return np.max(distances)
    else:
        return np.mean(sorted(distances.flatten(), reverse = True)[:int(len(distances) * 0.01)])


def eval_segmentation(gt_filenames, prediction_filenames, pro_integration_limit=0.3, delete_tiff_files=False):
    # Read all ground truth and anomaly images.
    ground_truth = []
    predictions = []

    print("Read ground truth files and corresponding predictions...")
    for (gt_name, pred_name) in tqdm(zip(gt_filenames, prediction_filenames),
                                     total=len(gt_filenames)):
        prediction = read_tiff(pred_name)
        predictions.append(prediction)

        if gt_name is not None:
            ground_truth.append(np.asarray(Image.open(gt_name)))
        else:
            ground_truth.append(np.zeros(prediction.shape))

    # Compute the PRO curve.
    pro_curve = compute_pro(
        anomaly_maps=predictions,
        ground_truth_maps=ground_truth)

    # Compute the area under the PRO curve.
    au_pro = trapezoid(
        pro_curve[0], pro_curve[1], x_max=pro_integration_limit)
    au_pro /= pro_integration_limit
    print(f"AU-PRO (FPR limit: {pro_integration_limit}): {au_pro}", end=" -- ")

    ground_truth = np.array(ground_truth).ravel()
    predictions = np.array(predictions).ravel()
    ground_truth = (ground_truth > 0).astype(np.uint8)

    # Compute pixel-level AUROC
    auroc_px = roc_auc_score(ground_truth, predictions)
    print(f"AUROC (pixel-level): {auroc_px}", end=" -- ")

    # # Compute pixel-level Average Precision (AP)
    precisions, recalls, thresholds = precision_recall_curve(ground_truth, predictions)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_px = np.max(f1_scores[np.isfinite(f1_scores)])

    print(f"F1 (pixel-level): {f1_px}")

    # optionally delete all tiff files in the anomaly_maps_dir to save disk space
    if delete_tiff_files:
        for pred_name in prediction_filenames:
            os.remove(pred_name + '.tiff')

    return au_pro, auroc_px, f1_px


def eval_classification(gt_filenames, prediction_filenames):
    # Read all ground truth and anomaly images.
    ground_truth = []
    predictions = []

    print("Read ground truth files and corresponding predictions...")
    for (gt_name, pred_name) in tqdm(zip(gt_filenames, prediction_filenames),
                                     total=len(gt_filenames)):
        prediction = np.load(pred_name + '.npy') 
        predictions.append(prediction)

        if gt_name is not None:
            ground_truth.append(np.asarray(Image.open(gt_name)))
        else:
            ground_truth.append(np.zeros(prediction.shape))
    
    # derive binary labels (0 = anomaly free, 1 = anomalous)
    binary_labels = [int(np.any(x > 0)) for x in ground_truth]
    del ground_truth

    predictions = [mean_top1p(dist.flatten()) for dist in predictions]

    # Compute image-level AUROC
    auroc_clf = roc_auc_score(binary_labels, predictions)
    print(f"AUROC (image-level): {auroc_clf}", end=" -- ")

    # Compute image-level Average Precision (AP)
    ap_clf = average_precision_score(binary_labels, predictions)
    print(f"Average Precision (image-level): {ap_clf}", end=" -- ")
  
    # Compute image_level F1 score
    precisions, recalls, thresholds = precision_recall_curve(binary_labels, predictions)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_clf = np.max(f1_scores[np.isfinite(f1_scores)])
    
    print(f"F1 (image-level): {f1_clf}")
    return auroc_clf, ap_clf, f1_clf


def get_objects_from_dataset(dataset):
    if dataset == "MVTec":
        objects = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
    elif dataset == "VisA":
        objects = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
    return objects
    

def eval_finished_run(dataset, dataset_base_dir, anomaly_maps_dir, output_dir, seed = None, pro_integration_limit = 0.3, eval_clf = True, eval_segm = False, delete_tiff_files = True):
    """
    Evaluate the results of a finished run on the MVTec AD dataset.
    Arguments:
    - dataset: Name of the dataset to evaluate (either "MVTec" or "VisA").
    - dataset_base_dir: Base directory of the MVTec AD dataset.
    - anomaly_maps_dir: Base directory where anomaly maps are located.
    - output_dir: Directory where to store the evaluation results.
    - seed: Seed of the run to evaluate.
    - pro_integration_limit: Limit of the false positive rate for the PRO curve.
    - eval_clf: Whether to evaluate classification performance.
    - eval_segm: Whether to evaluate segmentation performance.
    - delete_tiff_files: Whether tiff files are deleted after evaluation (default True)
    """

    # Parse the filenames of all ground truth and corresponding anomaly
    objects = get_objects_from_dataset(dataset)
    # Store evaluation results in this dictionary.
    evaluation_dict = dict()

    # Keep track of the mean performance measures.
    au_pro_ls = []
    auroc_px_ls = []
    f1_px_ls = []

    auroc_clf_ls = []
    ap_clf_ls = []
    f1_clf_ls = []

    # Evaluate each dataset object separately.
    for obj in objects:
        print(f"=== Evaluate {obj} ===")
        evaluation_dict[obj] = dict()

        # Parse the filenames of all ground truth and corresponding anomaly
        # images for this object.
        gt_filenames, prediction_filenames = \
            parse_dataset_files(
                object_name=obj,
                dataset_base_dir=dataset_base_dir,
                anomaly_maps_dir=anomaly_maps_dir,
                dataset=dataset)
        
        if eval_segm:
            # Evaluate segmentation performance
            au_pro, auroc_px, f1_px = \
                eval_segmentation(
                    gt_filenames,
                    prediction_filenames,
                    pro_integration_limit=pro_integration_limit,
                    delete_tiff_files=delete_tiff_files)

            evaluation_dict[obj]['seg_AUPRO'] = au_pro
            evaluation_dict[obj]['seg_AUROC'] = auroc_px
            evaluation_dict[obj]['seg_F1'] = f1_px

            # Keep track of the mean performance measures.
            au_pro_ls.append(au_pro)
            auroc_px_ls.append(auroc_px)
            f1_px_ls.append(f1_px)
            
        if eval_clf:
        # Evaluate classification performance
            auroc_clf, ap_clf, f1_clf = \
                eval_classification(
                    gt_filenames,
                    prediction_filenames)

            evaluation_dict[obj]['classification_AUROC'] = auroc_clf
            evaluation_dict[obj]['classification_AP'] = ap_clf
            evaluation_dict[obj]['classification_F1'] = f1_clf

            # Keep track of the mean performance measures.
            auroc_clf_ls.append(auroc_clf)
            ap_clf_ls.append(ap_clf)
            f1_clf_ls.append(f1_clf)

    # Compute the mean of the performance measures.
    if eval_segm:
        evaluation_dict['mean_au_pro'] = np.mean(au_pro_ls).item()
        evaluation_dict['mean_segmentation_au_roc'] = np.mean(auroc_px_ls).item()
        evaluation_dict['mean_segmentation_f1'] = np.mean(f1_px_ls).item()
    if eval_clf:
        evaluation_dict['mean_classification_au_roc'] = np.mean(auroc_clf_ls).item()
        evaluation_dict['mean_classification_ap'] = np.mean(ap_clf_ls).item()
        evaluation_dict['mean_classification_f1'] = np.mean(f1_clf_ls).item()
    
    if seed is not None:
        metric_file = f"metrics_seed={seed}.json"
    else:
        metric_file = "metrics.json"

    # Write the evaluation results to file
    if output_dir is not None:
        makedirs(output_dir, exist_ok=True)

        with open(path.join(output_dir, metric_file), 'w') as file:
            json.dump(evaluation_dict, file, indent=4)

        print(f"Wrote metrics to {path.join(output_dir, metric_file)}")