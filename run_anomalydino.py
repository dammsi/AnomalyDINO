import argparse
import csv
import os

import yaml
from tqdm import trange

from src.utils import get_dataset_info, list_images, limit_cpu_threads, PREPROCESS_OPTIONS
from src.detection import run_anomaly_detection
from src.post_eval import eval_finished_run
from src.visualize import create_sample_plots
from src.backbones import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot anomaly detection with AnomalyDINO.")
    parser.add_argument("--dataset", type=str, default="MVTec",
                        help="Dataset name: 'MVTec', 'VisA', or any other name for a custom dataset (objects are then inferred from --data_root).")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14",
                        help="Name of the backbone model, e.g., 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16', or 'vit_b_16'.")
    parser.add_argument("--data_root", type=str, default="data/mvtec_anomaly_detection",
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--preprocess", type=str, default="agnostic", choices=PREPROCESS_OPTIONS,
                        help="Preprocessing (masking & rotation) setting, see 'get_dataset_info' in src/utils.py.")
    parser.add_argument("--resolution", type=int, default=448,
                        help="Size of the smaller image edge (in pixels) after resizing.")
    parser.add_argument("--layer", nargs='+', type=int, default=None,
                        help="Transformer block(s) to take the patch features from (0-based, negative counts from the end, e.g., -1 = last block). "
                             "Several blocks are averaged after L2 normalization. Default: last block (as in the paper). "
                             "Masking always uses the last block. DINOv2/DINOv3 only.")
    parser.add_argument("--knn_metric", type=str, default="L2_normalized", choices=["L2", "L2_normalized"],
                        help="Distance for the nearest neighbor search ('L2_normalized' is equivalent to the cosine distance).")
    parser.add_argument("--k_neighbors", type=int, default=1,
                        help="Number of nearest neighbors to consider.")
    parser.add_argument("--faiss_on_cpu", default=False, action=argparse.BooleanOptionalAction,
                        help="Run the FAISS kNN search on CPU instead of GPU (use if faiss-gpu is not available; faiss-gpu is recommended to be installed via conda).")
    parser.add_argument("--shots", nargs='+', type=int, default=[1],
                        help="List of shots to evaluate. Full-shot scenario is -1.")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Number of repetitions (seeds) per shot, each seed uses different reference samples.")
    parser.add_argument("--mask_ref_images", default=False, action=argparse.BooleanOptionalAction,
                        help="Also apply masking to the reference images (default: only test images are masked).")
    parser.add_argument("--just_seed", type=int, default=None,
                        help="Run only this seed (overrides --num_seeds).")
    parser.add_argument('--save_examples', default=True, action=argparse.BooleanOptionalAction, help="Save example plots (reference/test examples and per-seed sample anomaly maps). --no-save_examples skips all plotting.")
    parser.add_argument("--eval_clf", default=True, action=argparse.BooleanOptionalAction, help="Evaluate anomaly detection performance.")
    parser.add_argument("--eval_segm", default=False, action=argparse.BooleanOptionalAction, help="Evaluate anomaly segmentation performance.")
    parser.add_argument("--device", default='cuda:0', help="CUDA device to run on, e.g., 'cuda:0'.")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Max. CPU threads per library (PyTorch, NumPy/scikit-learn, OpenCV, faiss); 0 = no limit (one per core). "
                             "Limiting is much faster when several runs share a machine.")
    parser.add_argument("--warmup_iters", type=int, default=25, help="Number of warmup iterations, relevant when benchmarking inference time.")

    parser.add_argument("--tag", help="Optional tag for the saving directory.")

    args = parser.parse_args()
    return args


if __name__=="__main__":

    args = parse_args()
    limit_cpu_threads(args.num_threads)
    
    print(f"Requested to run {len(args.shots)} (different) shot(s):", args.shots)
    if args.just_seed is not None:
        print(f"Requested to run only seed {args.just_seed}.")
    else:
        print(f"Requested to repeat the experiments {args.num_seeds} time(s).")

    objects, object_anomalies, masking_default, rotation_default = get_dataset_info(args.dataset, args.preprocess, data_path=args.data_root)

    # set CUDA device (e.g., 'cuda:1' -> only GPU 1 is visible, then addressed as 'cuda')
    if args.device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]
    model = get_model(args.model_name, 'cuda', smaller_edge_size=args.resolution, layers=args.layer)

    if not args.model_name.startswith("dinov2"):
        masking_default = {o: False for o in objects}
        print("Caution: Only DINOv2 supports 0-shot masking (for now)!")

    if args.just_seed is not None:
        seeds = [args.just_seed]
    else:
        seeds = range(args.num_seeds)
    
    for shot in list(args.shots):
        save_examples = args.save_examples

        model_dir = f"{args.model_name}_{args.resolution}" + (f"_layer{'+'.join(map(str, args.layer))}" if args.layer is not None else "")
        results_dir = f"results_{args.dataset}/{model_dir}/{shot}-shot_preprocess={args.preprocess}"
        
        if args.tag is not None:
            results_dir += "_" + args.tag
        plots_dir = results_dir
        os.makedirs(f"{results_dir}", exist_ok=True)
        
        # save preprocessing setups (masking and rotation) to file
        with open(f"{results_dir}/preprocess.yaml", "w") as f:
            yaml.dump({"masking": masking_default, "rotation": rotation_default}, f)

        # save arguments to file
        with open(f"{results_dir}/args.yaml", "w") as f:
            yaml.dump(vars(args), f)

        if args.faiss_on_cpu:
            print("Warning: Running similarity search on CPU. Consider using faiss-gpu for faster inference.")
        
        print("Results will be saved to", results_dir)
    
        for seed in seeds:
            print(f"=========== Shot = {shot}, Seed = {seed} ===========")
            
            if os.path.exists(f"{results_dir}/metrics_seed={seed}.json"):
                print(f"Results for shot {shot}, seed {seed} already exist. Skipping.")
                continue
            else:
                measurements_file = results_dir + f"/measurements_seed={seed}.csv"
                with open(measurements_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Object", "Sample", "Anomaly_Score", "MemoryBank_Time", "Inference_Time"])

                    for object_name in objects:
                        
                        if save_examples:
                            os.makedirs(f"{plots_dir}/{object_name}", exist_ok=True)
                            os.makedirs(f"{plots_dir}/{object_name}/examples", exist_ok=True)

                        # CUDA warmup (the image is loaded once, only the forward pass is repeated)
                        if args.warmup_iters > 0:
                            warmup_image = list_images(f"{args.data_root}/{object_name}/train/good")[0]
                            img_tensor, _ = model.prepare_image(f"{args.data_root}/{object_name}/train/good/{warmup_image}")
                            for _ in trange(args.warmup_iters, desc="CUDA warmup", leave=False):
                                model.extract_features(img_tensor)
                                         
                        anomaly_scores, time_memorybank, time_inference = run_anomaly_detection(
                                                                                model,
                                                                                object_name,
                                                                                data_root = args.data_root, 
                                                                                n_ref_samples = shot,
                                                                                object_anomalies = object_anomalies,
                                                                                plots_dir = plots_dir,
                                                                                save_examples = save_examples,
                                                                                knn_metric = args.knn_metric,
                                                                                knn_neighbors = args.k_neighbors,
                                                                                faiss_on_cpu = args.faiss_on_cpu,
                                                                                masking = masking_default[object_name],
                                                                                mask_ref_images = args.mask_ref_images,
                                                                                rotation = rotation_default[object_name],
                                                                                seed = seed,
                                                                                save_patch_dists = args.eval_clf, # save patch distances for detection evaluation
                                                                                save_tiffs = args.eval_segm)      # save anomaly maps as tiffs for segmentation evaluation
                        
                        # write anomaly scores and inference times to file
                        for sample, anomaly_score in anomaly_scores.items():
                            inference_time = time_inference[sample]
                            writer.writerow([object_name, sample, f"{anomaly_score:.5f}", f"{time_memorybank:.5f}", f"{inference_time:.5f}"])

                # read inference times from file
                with open(measurements_file, 'r') as file:
                    reader = csv.reader(file)
                    next(reader)
                    inference_times = [float(row[4]) for row in reader]
                print(f"Finished AD for {len(objects)} objects (seed {seed}), mean inference time: {sum(inference_times)/len(inference_times):.5f} s/sample = {len(inference_times)/(sum(inference_times)+1e-10):.2f} samples/s")

                # check wheter 'good' folder exists for testing
                for object_name in objects:
                    good_folder = f"{args.data_root}/{object_name}/test/good/"
                    if not os.path.exists(good_folder):
                        print(f"Warning: 'good' folder not found for {object_name}! No evaluation will be performed for seed {seed}.")
                        print("Finished AD without evaluation, inference results saved to", results_dir)
                        break
                else:
                    # evaluate all finished runs and create sample anomaly maps for inspection
                    print(f"=========== Evaluate seed = {seed} ===========")
                    eval_finished_run(args.dataset, 
                                    args.data_root, 
                                    anomaly_maps_dir = results_dir + f"/anomaly_maps/seed={seed}", 
                                    output_dir = results_dir,
                                    seed = seed,
                                    pro_integration_limit = 0.3,
                                    eval_clf = args.eval_clf,
                                    eval_segm = args.eval_segm,
                                    objects = objects)
                    
                    # sample anomaly maps are created from the saved patch distances (only saved with --eval_clf);
                    # skipped with --no-save_examples (no plotting at all)
                    if args.eval_clf and args.save_examples:
                        create_sample_plots(results_dir,
                                            anomaly_maps_dir = results_dir + f"/anomaly_maps/seed={seed}",
                                            seed = seed,
                                            dataset = args.dataset,
                                            data_root = args.data_root)
                
                    # deactivate creation of examples for the next seeds...
                    save_examples = False 

    print("Finished all runs!")