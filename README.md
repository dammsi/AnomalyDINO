# AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2 <img align="right" src="media/AnomalyDINO.png" style="height: 84px; max-width: 100%;">

*Simon Damm, Mike Laszkiewicz, Johannes Lederer, Asja Fischer*

This is the official code to reproduce the experiments in the paper [AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2](https://arxiv.org/abs/2405.14529).

## Prerequisits

1. Create a virtual environment (e.g., `python -m venv .venvAnomalyDINO`), activate it (e.g., `source .venvAnomalyDINO/bin/activate`) and install the required dependencies for AnomalyDINO:
    ```shell
    pip install -r requirements.txt
    ```
    Info: If you want to use `faiss` with GPU-acceleration we recommend setting up a conda environment with the required packages instead (only conda installation is supported, see, e.g., [here](https://github.com/facebookresearch/faiss/wiki/Installing-Faiss#why-dont-you-support-installing-via-xxx-)).

2. Download and prepare the datasets [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and [VisA](https://github.com/amazon-science/spot-diff) from their official sources.
For VisA, follow the instruction in the official repo to organize the data in the official 1-class splits. 
The default data roots are `data/mvtec_anomaly_detection` for MVTec-AD, and `data/VisA_pytorch/1cls/` for VisA. 
Please adapt the function calls below if necessary. 
Alternatively, prepare your own dataset accordingly:
    ```
    your_data_root
    ├── object1
    │   ├── ground_truth        # anomaly annotaions per anomaly type
    │   │   ├── anomaly_type1
    │   │   ├── ...
    │   ├── test                # test images per anomaly type & 'good'
    │   │   ├── anomaly_type1    
    │   │   ├── ...
    │   │   └── good
    │   └── train               # train/reference images (without anomalies)
    │       └── good
    ├── object2
    │   ├── ...
    ```


## Usage

### Short Demo
Get started with the minimal demo to perform few-shot anomaly detection (`demo_AD_DINO.ipynb`).

### Few-shot anomaly detection

For the full evaluation, run the script `run_anomalydino.py` on the selected dataset for a given number of shots and repetitions (seeds).
The preprocessing to your dataset can be specified in `src/utils.py` in `get_dataset_info`, default is "agnostic" (apply masking whenever PCA-based masking works well & augment reference samples by rotations, see the paper).

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

The script automatically creates some example plots, plots some anomaly maps for each object, and automatically evaluates each run (activate evaluation of segementation with `--eval_segm True` if applicable). 

Evaluation results are saved in the respective results directory as `metrics_seed={seed}.json` for each seed.


### Batched-Zero-Shot Anomay Detection
To reproduce the results in the *batched* zero-shot scenario, run `run_anomalydino_batched.py` with appropriate arguments:

```shell
python run_anomalydino_batched.py --dataset MVTec --data_root data/mvtec_anomaly_detection
```
```shell
python run_anomalydino_batched.py --dataset VisA --data_root data/VisA_pytorch/1cls/
```

---

This work builds on the following ressources:
- [DINOv2](https://github.com/facebookresearch/dinov2), code and model available under Apache 2.0 license.
- The [MVTec-AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad), available under the CC BY-NC-SA 4.0 license.
- The [VisA dataset](https://github.com/amazon-science/spot-diff), available under the CC BY 4.0 license.

---

If you find this repository useful in your research/project, please consider citing the paper:

```
@misc{damm2024anomalydino,
      title={AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2}, 
      author={Simon Damm and Mike Laszkiewicz and Johannes Lederer and Asja Fischer},
      year={2024},
      eprint={2405.14529},
      url={https://arxiv.org/abs/2405.14529}, 
}
```