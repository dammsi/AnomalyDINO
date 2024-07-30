import os
import time
from argparse import ArgumentParser, Action 
import yaml

from src.dino import DINOv2
from src.utils import get_dataset_info 
from src.detection import run_anomaly_detection
from src.post_eval import eval_finished_run
from src.visualize import create_sample_plots


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
    parser.add_argument("--data_root", type=str, default="data/mvtec_anomaly_detection",
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--model_size", type=str, default="s",
                        help="Size of the DINOv2 ViT. Choose from ['s', 'm', 'l', 'g'].")
    parser.add_argument("--preprocess", type=str, default="agnostic",
                        help="Preprocessing method. Choose from ['agnostic', 'informed'].")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--knn_metric", type=str, default="L2_normalized")
    parser.add_argument("--k_neighbors", type=int, default=1)
    parser.add_argument("--shots", nargs='+', type=int, default=[1], action=IntListAction, 
                        help="List of shots to evaluate. Full-shot scenario is -1.")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--just_seed", type=int, default=None)
    parser.add_argument("--save_examples", default=True)
    parser.add_argument("--save_anomaly_map", default=True, help="Save anomaly map as tiff image (might consume some disk space).")
    parser.add_argument("--eval_clf", type=bool, default=True) 
    parser.add_argument("--eval_segm", type=bool, default=False)
    parser.add_argument("--device", default='cuda:0')

    parser.add_argument("--tag", help="Optional tag for the saving directory.")

    args = parser.parse_args()
    return args


if __name__=="__main__":

    args = parse_args()
    
    print(f"Requested to run {len(args.shots)} (different) shot(s):", args.shots)
    print(f"Requested to repeat the experiments {args.num_seeds} time(s).")

    args.model_name = "dinov2_vit" + args.model_size.lower() + "14"
    dm = DINOv2(model_name=args.model_name, smaller_edge_size=args.resolution, device=args.device)
    objects, object_anomalies, masking_default, rotation_default = get_dataset_info(args.dataset, args.preprocess)

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

        print("Results will be saved to", results_dir)

        elapsed_times = []

        for seed in seeds:
            print(f"=========================================================== Shot = {shot}, Seed = {seed} ===========================================================")
            
            if os.path.exists(f"{results_dir}/metrics_seed={seed}.json"):
                print(f"Results for shot {shot}, seed {seed} already exist. Skipping.")
                continue
            else:
                start_time = time.time() # start measuring time
                
                for object_name in objects:
                    
                    if save_examples:
                        os.makedirs(f"{plots_dir}/{object_name}", exist_ok=True)
                        os.makedirs(f"{plots_dir}/{object_name}/examples", exist_ok=True)
                    
                    run_anomaly_detection(dm,
                                        object_name,
                                        data_root = args.data_root, 
                                        n_ref_samples = shot,
                                        object_anomalies = object_anomalies,
                                        plots_dir = plots_dir,
                                        save_examples = save_examples,
                                        knn_metric = args.knn_metric,
                                        knn_neighbors = args.k_neighbors,
                                        masking = masking_default[object_name],
                                        rotation = rotation_default[object_name],
                                        seed = seed,
                                        save_patch_dists = args.eval_clf, # save the patch distances for detection evaluation
                                        save_tiffs = args.eval_segm)      # save the anomaly maps as tiff images for segmentation evaluation

                end_time = time.time()  # end measuring time
                elapsed_time = end_time - start_time
                elapsed_times.append(elapsed_time)
                with open(f"{results_dir}/elapsed_times.txt", "a") as file:
                    file.write(f"Seed: {seed}, Elapsed Time: {elapsed_time:.6f} seconds\n")

                # save arguments to file
                with open(f"{results_dir}/args.yaml", "w") as f:
                    yaml.dump(vars(args), f)
                
                print(f"Finished AD for {len(objects)} objects (seed {seed}). Elapsed time: {elapsed_time:.1f} seconds.")

                # deactivate creation of examples for the next seeds...
                save_examples = False 

        
        if len(elapsed_times) == 0:
            print("No new runs were executed.")
        else:
            # write the mean elapsed time to the file
            with open(f"{results_dir}/elapsed_times.txt", "a") as file:
                file.write(f"\nMean Elapsed Time: {sum(elapsed_times) / len(elapsed_times):.6f} seconds\n")
            
        # evaluate the finished runs and create sample anomaly maps for inspection
        for seed in seeds:

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