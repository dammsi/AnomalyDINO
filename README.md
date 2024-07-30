# AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2

<!-- 
This is the official code to reproduce the experiments in the paper [AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2](https://arxiv.org/abs/2405.14529). -->

<div style="display: flex; flex-wrap: wrap;">
  <div style="flex: 8; padding-right: 20px;">

*Simon Damm, Mike Laszkiewicz, Johannes Lederer, Asja Fischer*

This is the official code to reproduce the experiments in the paper [AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2](https://arxiv.org/abs/2405.14529).

</div>
<div style="flex: 1;">
<img src="media/AnomalyDINO_transparent.png" alt="AnomalyDINO" style="height: 64px; max-width: 100%; height: auto;">
</div>
</div>

## Prerequisits

1. Download and prepare the datasets [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and [VisA](https://github.com/amazon-science/spot-diff) from their official sources. 
For VisA, follow the instruction in the official repo to organize the data in the official 1-class splits. 
The default folder are `data/mvtec_anomaly_detection` for MVTec-AD, and `data/VisA_pytorch/1cls/` for VisA. 
Please adapt the function calls below if necessary. 

2. Create a virtual environment (e.g., `python -m venv .venvAnomalyDINO`), activate it ( `source .venvAnomalyDINO/bin/activate`) and install the required dependencies for AnomalyDINO:
    ```shell
    pip install -r requirements.txt
    ```

## Usage

### Short Demo
Test the minimal demo to perform few-shot anomaly detection (`demo_AD_DINO.ipynb`).

### Few-shot anomaly detection

Run the script `run_anomalydino.py` on the selected dataset for a given number of shots and repetitions (seeds).
The preprocessing to your dataset can be specified in `src/utils.py` in `get_dataset_info`, default is "agnostic" (apply masking whenever PCA-based masking works well & augment reference samples by rotations).

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

The script automatically creates some example plots, plots some anomaly maps for each object, and automatically evaluates each run (deactivate evaluation of segementation with `--eval_segm False` if applicable). 
<!-- The evaluation script is partly based on the official MVTec evaluation (also available [here](https://www.mvtec.com/company/research/datasets/mvtec-ad)).  -->

Evaluation results are saved in the respective results directory as `metrics_seed={seed}.json` for each seed.


### Batched-Zero-Shot Anomay Detection
We also generalized the proposed method to the *batched* zero-shot scenario.
The results can be reproduced with: 

```shell
python run_anomalydino_batched.py --dataset MVTec --data_root data/mvtec_anomaly_detection
```
```shell
python run_anomalydino_batched.py --dataset VisA --data_root data/VisA_pytorch/1cls/
```