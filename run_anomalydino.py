import argparse
import os
import time
from argparse import ArgumentParser, Action 
import yaml
from tqdm import trange

import csv

# from src.dino import DINOv2
from src.utils import get_dataset_info 
from src.detection import run_anomaly_detection
from src.post_eval import eval_finished_run
from src.visualize import create_sample_plots
from src.backbones import get_model


class IntListAction(Action):
    """
    Define a custom action to always return a list. 
    This allows --shots 1 to be treated as a list of one element [1]. 
    """
    def __call__(self, namespace, values):
        if not isinstance(values, list):
            values = [values]
        setattr(namespace, self.dest, values)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--model_name", type=str, default="dinov2_vits14", help="Name of the backbone model. Choose from ['dinov2_vits14', 'dino_vitb24', 'dino_vitl14', 'dino_vitg14', 'vit_b_16'].")
    parser.add_argument("--data_root", type=str, default="data/mvtec_anomaly_detection",
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--preprocess", type=str, default="agnostic",
                        help="Preprocessing method. Choose from ['agnostic', 'informed', 'masking_only'].")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--knn_metric", type=str, default="L2_normalized")
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument("--faiss_on_cpu", default=False, action=argparse.BooleanOptionalAction, help="Use GPU for FAISS kNN search. (Conda install faiss-gpu recommended, does usually not work with pip install.)")
    parser.add_argument("--shots", nargs='+', type=int, default=[1], #action=IntListAction,
                        help="List of shots to evaluate. Full-shot scenario is -1.")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--just_seed", type=int, default=None)
    parser.add_argument('--save_examples', default=True, action=argparse.BooleanOptionalAction, help="Save example plots.")
    parser.add_argument("--eval_clf", default=True, action=argparse.BooleanOptionalAction, help="Evaluate anomaly detection performance.")
    parser.add_argument("--eval_segm", default=False, action=argparse.BooleanOptionalAction, help="Evaluate anomaly segmentation performance.")
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--warmup_iters", type=int, default=25, help="Number of warmup iterations, relevant when benchmarking inference time.")

    parser.add_argument("--tag", help="Optional tag for the saving directory.")

    args = parser.parse_args()
    return args


if __name__=="__main__":

    args = parse_args()
    
    print(f"Requested to run {len(args.shots)} (different) shot(s):", args.shots)
    print(f"Requested to repeat the experiments {args.num_seeds} time(s).")

    objects, object_anomalies, masking_default, rotation_default = get_dataset_info(args.dataset, args.preprocess)

    model = get_model(args.model_name, args.device, smaller_edge_size=args.resolution)

    if not args.model_name.startswith("dinov2"):
        masking_default = {o: False for o in objects}
        print("Caution: Only DINOv2 supports 0-shot masking (for now)!")    

    if args.just_seed != None:
        seeds = [args.just_seed]
    else:
        seeds = range(args.num_seeds)
    
    for shot in list(args.shots):
        save_examples = args.save_examples

        results_dir = f"results_{args.dataset}/{args.model_name}_{args.resolution}/{shot}-shot_preprocess={args.preprocess}"
        
        if args.tag != None:
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
                timeit_file = results_dir + "/time_measurements.csv"
                with open(timeit_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Object", "Sample", "Anomaly_Score", "MemoryBank_Time", "Inference_Time"])

                    for object_name in objects:
                        
                        if save_examples:
                            os.makedirs(f"{plots_dir}/{object_name}", exist_ok=True)
                            os.makedirs(f"{plots_dir}/{object_name}/examples", exist_ok=True)

                        # CUDA warmup
                        for _ in trange(args.warmup_iters, desc="CUDA warmup", leave=False):
                            img_tensor, grid_size = model.prepare_image(f"{args.data_root}/{object_name}/train/good/000.png")
                            features = model.extract_features(img_tensor)
                                         
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
                                                                                rotation = rotation_default[object_name],
                                                                                seed = seed,
                                                                                save_patch_dists = args.eval_clf, # save patch distances for detection evaluation
                                                                                save_tiffs = args.eval_segm)      # save anomaly maps as tiffs for segmentation evaluation
                        
                        # write anomaly scores and inference times to file
                        for counter, sample in enumerate(anomaly_scores.keys()):
                            anomaly_score = anomaly_scores[sample]
                            inference_time = time_inference[sample]
                            writer.writerow([object_name, sample, f"{anomaly_score:.5f}", f"{time_memorybank:.5f}", f"{inference_time:.5f}"])
                        # print(f"Mean inference time ({object_name}): {sum(time_inference.values())/len(time_inference):.5f} s/sample")                        

                # read inference times from file
                with open(timeit_file, 'r') as file:
                    reader = csv.reader(file)
                    next(reader)
                    inference_times = [float(row[4]) for row in reader]
                print(f"Finished AD for {len(objects)} objects (seed {seed}), mean inference time: {sum(inference_times)/len(inference_times):.5f} s/sample")

                # deactivate creation of examples for the next seeds...
                save_examples = False 
            
        # evaluate all finished runs and create sample anomaly maps for inspection
        for seed in seeds:
            print(f"=========== Evaluate seed = {seed} ===========")
            
            eval_finished_run(args.dataset, 
                              args.data_root, 
                              anomaly_maps_dir = results_dir + f"/anomaly_maps/seed={seed}", 
                              output_dir = results_dir,
                              seed = seed,
                              pro_integration_limit = 0.3,
                              eval_clf = args.eval_clf,
                              eval_segm = args.eval_segm)
            
            create_sample_plots(results_dir, 
                                anomaly_maps_dir = results_dir + f"/anomaly_maps/seed={seed}", 
                                seed = seed,
                                dataset = args.dataset, 
                                data_root = args.data_root)

    print("Finished and evaluated all runs!")