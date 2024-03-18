import pandas as pd
import numpy as np
import sklearn.metrics as skmetrics
import os
from h5py import File
import pickle
import argparse
from tqdm import tqdm


def calculate_metric(y_true, y_score, metric='auc'):
    if metric == "auc":
        return skmetrics.roc_auc_score(y_true, y_score)
    elif metric == "ap":
        return skmetrics.average_precision_score(y_true, y_score)
    elif metric == "precision":
        return skmetrics.precision_score(y_true, y_score)
    elif metric == "recall":
        return skmetrics.recall_score(y_true, y_score)
    elif metric == "f1":
        return skmetrics.f1_score(y_true, y_score)
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented.")


def generate_tile_labels(slide_id, h5_folder, gt_folder):
    h5_file = os.path.join(h5_folder, f"{slide_id}.h5")
    with File(h5_file, 'r') as f:
        labels = np.zeros(len(f["features"]))
    gt_idx_file = os.path.join(gt_folder, f"{slide_id}.pkl")
    if os.path.exists(gt_idx_file):
        with open(gt_idx_file, 'rb') as f:
            tum_idx = pickle.load(f)
        labels[tum_idx] = 1

    return labels


def get_predictions(pred_file, pred_folder, pred_type="soft"):
    pred_array = pd.read_csv(pred_file).to_numpy()
    if pred_type == "soft":
        return pred_array[:, 1]
    elif pred_type == "hard":
        return np.argmax(pred_array, axis=1)
    else:
        raise ValueError(
            f"Argument pred_type must be either 'soft' or 'hard'."
        )


def main(args):
    if args.task == "camelyon16":
        gt_folder = "./data/camelyon16/gt_patches_indexes"
        h5_folder = "./data/camelyon16/features/h5_files"

    elif args.task == "digestpath2019":
        # extracted patches were 128x128 pixels in the original work
        gt_folder = "./data/digestpath2019/gt_patches_indexes"
        h5_folder = "./data/digestpath2019/features/h5_files"

    exp_name = os.path.basename(args.experiment.rstrip('/'))

    metrics = {'auc': [], 'ap': [], 'precision': [], 'recall': [], 'f1': []}
    for fold in range(5):
        pred_folder = os.path.join(args.experiment, f"fold_{fold}")
        tile_scores = {"y_score": [], "y_pred": []}
        y_true = []
        total = len(os.listdir(pred_folder))
        for pred_file in tqdm(os.scandir(pred_folder), total=total):
            slide_id = os.path.splitext(pred_file.name)[0]
            tile_scores["y_score"].append(
                get_predictions(pred_file.path, pred_folder)
            )
            tile_scores["y_pred"].append(
                get_predictions(pred_file.path, pred_folder, pred_type='hard')
            )
            y_true.append(generate_tile_labels(slide_id, h5_folder, gt_folder))
        tile_scores["y_score"] = np.hstack(tile_scores["y_score"])
        tile_scores["y_pred"] = np.hstack(tile_scores["y_pred"])
        y_true = np.hstack(y_true)
        for metric in metrics.keys():
            if metric in ('auc', 'ap'):
                metrics[metric].append(
                    calculate_metric(y_true, tile_scores["y_score"], metric)
                )
            else:
                metrics[metric].append(
                    calculate_metric(y_true, tile_scores["y_pred"], metric)
                )
    for metric in metrics.keys():
        metrics[metric].append(np.mean(metrics[metric]))
        metrics[metric].append(np.std(metrics[metric], ddof=1))

    savepath = os.path.join(args.savedir, exp_name)
    os.makedirs(savepath, exist_ok=True)
    index = pd.Index([0, 1, 2, 3, 4, 'mean', 'std'], name='fold')
    metrics = pd.DataFrame(metrics, index=index)
    metrics.to_csv(os.path.join(savepath, "tile_metrics.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculates tile-level metrics for any experiment."
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Path to the tile predictions folder."
    )
    parser.add_argument(
        "--task", type=str, choices=["camelyon16", "digestpath2019"],
        help="Name of the classification task."
    )
    parser.add_argument("--savedir", required=True)
    args = parser.parse_args()
    if not os.path.exists(args.experiment):
        raise FileNotFoundError(f"{args.experiment} does not exist.")
    main(args)
