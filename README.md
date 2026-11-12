# [WACV2025] AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2 <img align="right" src="media/AnomalyDINO.png" style="height: 84px; max-width: 100%;">

*Simon Damm, Mike Laszkiewicz, Johannes Lederer, Asja Fischer*

This is the official code to reproduce the experiments in the paper [AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2](https://arxiv.org/abs/2405.14529), accepted at IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2025).

## Updates
- **Nov 2026:** [DINOv3](https://github.com/facebookresearch/dinov3) backbones included (`--model_name dinov3_vits16`, also in the batched zero-shot script). With the default settings, DINOv2 seems to stay the better choice, see the [comparison below](#dinov2-vs-dinov3).

## Prerequisites

1. Create a virtual environment (e.g., `python -m venv .venvAnomalyDINO`), activate it (e.g., `source .venvAnomalyDINO/bin/activate`) and install the required dependencies for AnomalyDINO:
    ```shell
    pip install -r requirements.txt
    ```
    Info: The requirements install `faiss-gpu-cu12` for GPU-accelerated similarity search (CUDA 12, Python 3.10–3.13). Alternatively, `faiss-gpu` can be installed via conda (officially supported, see [here](https://github.com/facebookresearch/faiss/wiki/Installing-Faiss); note that the conda packages currently require `numpy<2`). To perform similarity search on CPU, install `faiss-cpu` instead and set the additional flag `--faiss_on_cpu`.

2. Download and prepare the datasets [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and [VisA](https://github.com/amazon-science/spot-diff) from their official sources.
For VisA, follow the instruction in the official repo to organize the data in the official 1-class splits. 
The default data roots are `data/mvtec_anomaly_detection` for MVTec-AD, and `data/VisA_pytorch/1cls/` for VisA. 
Please adapt the function calls below if necessary. 
Alternatively, prepare your own dataset accordingly:
    ```
    your_data_root
    ├── object1
    │   ├── ground_truth        # anomaly annotations per anomaly type (optional)
    │   │   ├── anomaly_type1
    │   │   ├── ...
    │   ├── test                # test images per anomaly type & 'good' (if applicable)
    │   │   ├── anomaly_type1    
    │   │   ├── ...
    │   │   └── good
    │   └── train               # train/reference images (without anomalies)
    │       └── good
    ├── object2
    │   ├── ...
    ```
When no 'good' test set is available, just inference is performed (no evaluation of detection/segmentation metrics possible).
For custom datasets (any `--dataset` name other than MVTec/VisA), test images can be in common image formats and ground truth masks are expected as `<image name>_mask.png` or `<image name>.png`. Without masks, detection metrics are still evaluated (segmentation requires masks).

## Usage

### Short Demo
Get started with the minimal demo to perform few-shot anomaly detection (`demo_AnomalyDINO.ipynb`).

### Few-shot anomaly detection

For the full evaluation, run the script `run_anomalydino.py` on the selected dataset for a given number of shots and repetitions (seeds).
The preprocessing for your dataset can be specified in `src/utils.py` in `get_dataset_info`, default is "agnostic" (apply masking whenever PCA-based masking works well & augment reference samples by rotations, see the paper).

The results for the default setting, i.e., all considered shots, three repetitions, and agnostic preprocessing, can be reproduced by calling:
```shell
python run_anomalydino.py --dataset MVTec --shots 1 2 4 8 16 --num_seeds 3 --preprocess agnostic --data_root data/mvtec_anomaly_detection
```

```shell
python run_anomalydino.py --dataset VisA --shots 1 2 4 8 16 --num_seeds 3 --preprocess agnostic --data_root data/VisA_pytorch/1cls/
```

For a faster inspection use, e.g.,
```shell
python run_anomalydino.py --dataset MVTec --shots 1 --num_seeds 1 --preprocess informed --data_root data/mvtec_anomaly_detection
```

The script automatically creates some example plots, plots some anomaly maps for each object, and automatically evaluates each run (activate evaluation of segmentation with `--eval_segm` if applicable).

Evaluation results are saved in the respective results directory as `metrics_seed={seed}.json` for each seed.

Other backbones can be selected with `--model_name`, e.g., `dinov2_vitb14` or DINOv3 (`dinov3_vits16`, `dinov3_vitb16`, `dinov3_vitl16`, ...; loaded via `timm`). 
Note that zero-shot masking is only supported for DINOv2, for all other backbones masking is deactivated.
By default, patch features are taken from the last transformer block (as in the paper), other layers can be selected with `--layer`.

### Tests
Some quick sanity checks (no datasets, model downloads or GPU required) can be run with `pytest tests` (`pytest` is included in `requirements.txt`).

### Batched-Zero-Shot Anomaly Detection
To reproduce the results in the *batched* zero-shot scenario, run `run_anomalydino_batched.py` with appropriate arguments:

```shell
python run_anomalydino_batched.py --dataset MVTec --data_root data/mvtec_anomaly_detection
```
```shell
python run_anomalydino_batched.py --dataset VisA --data_root data/VisA_pytorch/1cls/
```
Other backbones can be selected with `--model_name` (e.g., `dinov3_vits16`), the resolution with `--resolution`.

---

### DINOv2 vs. DINOv3
Image-level AUROC (%) with the default settings (agnostic preprocessing, last block), mean ± std over 3 seeds:

**MVTec-AD**

| Backbone (res.) | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| DINOv2-S/14 (448) | 96.5 <sub>±0.5</sub> | 96.7 <sub>±1.0</sub> | 97.6 <sub>±0.1</sub> | 98.0 <sub>±0.1</sub> | 98.3 <sub>±0.1</sub> |
| DINOv2-S/14 (672) | **96.6** <sub>±0.5</sub> | **96.9** <sub>±0.8</sub> | **97.7** <sub>±0.2</sub> | **98.2** <sub>±0.2</sub> | **98.4** <sub>±0.1</sub> |
| DINOv3-S/16 (512) | 93.0 <sub>±1.1</sub> | 93.9 <sub>±1.2</sub> | 94.8 <sub>±0.3</sub> | 96.0 <sub>±0.6</sub> | 97.1 <sub>±0.3</sub> |
| DINOv3-S/16 (768) | 92.2 <sub>±1.2</sub> | 93.4 <sub>±1.0</sub> | 94.5 <sub>±0.8</sub> | 95.8 <sub>±0.9</sub> | 97.1 <sub>±0.6</sub> |

**VisA**

| Backbone (res.) | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| DINOv2-S/14 (448) | 85.6 <sub>±1.8</sub> | 88.3 <sub>±2.1</sub> | 91.2 <sub>±0.9</sub> | 92.5 <sub>±0.1</sub> | 93.8 <sub>±0.1</sub> |
| DINOv2-S/14 (672) | **87.4** <sub>±1.5</sub> | **89.7** <sub>±1.6</sub> | **92.5** <sub>±1.1</sub> | **93.8** <sub>±0.4</sub> | **94.8** <sub>±0.2</sub> |
| DINOv3-S/16 (512) | 82.9 <sub>±0.9</sub> | 85.5 <sub>±0.5</sub> | 87.4 <sub>±0.3</sub> | 88.4 <sub>±0.0</sub> | 89.2 <sub>±0.2</sub> |
| DINOv3-S/16 (768) | 82.7 <sub>±0.8</sub> | 85.2 <sub>±0.9</sub> | 86.8 <sub>±0.1</sub> | 87.9 <sub>±0.2</sub> | 88.9 <sub>±0.2</sub> |

Resolutions are chosen so that both backbones see a comparable number of patch tokens per image (448/14 = 512/16 = 32 and 672/14 = 768/16 = 48 patches per side).
Zero-shot masking is only applied for DINOv2 (not calibrated for DINOv3, helps mainly on VisA).
Measured with updated dependencies (so numbers differ slightly from the paper results).

---

This work uses the following resources and datasets:
- [DINOv2](https://github.com/facebookresearch/dinov2), code and model available under Apache 2.0 license.
- [DINOv3](https://github.com/facebookresearch/dinov3) (optional backbone), code and model available under the DINOv3 License.
- The [MVTec-AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad), available under the CC BY-NC-SA 4.0 license.
- The [VisA dataset](https://github.com/amazon-science/spot-diff), available under the CC BY 4.0 license.

---

If you find this repository useful in your research/project, please consider citing the paper:

```
@inproceedings{damm2024anomalydino,
      title={AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2}, 
      author={Simon Damm and Mike Laszkiewicz and Johannes Lederer and Asja Fischer},
      booktitle={Proceedings of the Winter Conference on Applications of Computer Vision (WACV 2025)},
      year={2025},
      url={https://arxiv.org/abs/2405.14529}, 
}
```