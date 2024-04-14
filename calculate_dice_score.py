import os
import numpy as np
# import multiprocessing
import argparse
import pandas as pd
import cv2
import re
import logging
from tqdm import tqdm


description = '''Calculates dice score for predicted binary masks on
pathological slides, and fall-out or false positive rate on
non-pathological slides.'''


def get_threshold(mask_path):
    return float(re.search("th-[0-9].[0-9]", mask_path).group(0).lstrip("th-"))


def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError(
            "Shape mismatch: im1 and im2 must have the same shape."
            f"im1 has shape {im1.shape} and im2 has shape {im2.shape}."
        )

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def calculate_metrics(slides, subdir, tile_probs_dir, thresh):
    '''
    Calculates either the Dice score for tumorous slides, or the
    specificity for normal slides.
    '''
    dice_scores = {}
    spec_scores = {}

    for slide in tqdm(slides):
        slide = SLIDES.loc[slide]
        if slide["label"] == "tumor_tissue":
            # calculate dice score
            pred_mask_path = os.path.join(
                subdir, slide.name + "-tile_level_mask.png"
            )
            ref_mask_path = os.path.join(REF_MASKS, f'{slide.name}.png')
            if not os.path.exists(pred_mask_path) or \
                    not os.path.exists(ref_mask_path):
                print(f'A mask is missing for {slide.name}. Skipping...')
                continue
            pred_mask = cv2.imread(pred_mask_path)
            ref_mask = cv2.imread(ref_mask_path)
            dice_scores[slide.name] = dice(pred_mask, ref_mask)
        else:
            # calculate false positive rate
            df = pd.read_csv(os.path.join(
                tile_probs_dir, slide.name + ".csv"))
            total_tiles = len(df)
            prob_tum = df.prob_1
            prob_tum_th = prob_tum[prob_tum > thresh]
            false_pos = len(prob_tum_th)
            # specificity = 1 - FPR = 1 - FP / N
            spec_scores[slide.name] = 1 - false_pos / total_tiles

    dice_scores = pd.Series(dice_scores).to_frame(name='value')
    dice_scores['metric'] = ['dice' for _ in dice_scores.index]
    spec_scores = pd.Series(spec_scores).to_frame(name='value')
    spec_scores['metric'] = ['specificity' for _ in spec_scores.index]
    metrics = pd.concat((dice_scores, spec_scores)).sort_index()

    return metrics


def main(args):
    # init logger
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    slides = [
        os.path.splitext(f)[0]
        for f in os.listdir(args.tile_predictions_path)
    ]
    subdir = args.predicted_masks_path
    if os.path.exists(subdir):
        tile_probs_dir = args.tile_predictions_path
        metrics = calculate_metrics(
            slides, subdir, tile_probs_dir, THRESH
        )
        metrics.to_csv(
            os.path.join(args.predicted_masks_path, f"metrics.csv"),
            header=False
        )

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--predicted_masks_path",
        required=True, help="Path to the predicted masks."
    )
    parser.add_argument(
        "--tile_predictions_path",
        required=True, help="Path to the tile predictions."
    )
    parser.add_argument(
        '--dataset',
        required=True, help='Path to the csv file corresponding to the dataset'
    )
    parser.add_argument(
        '--reference_masks',
        required=True,
        help="Path to the directory containing the reference masks as images."
    )
    args = parser.parse_args()
    print()

    if not os.path.exists(args.predicted_masks_path):
        raise FileNotFoundError(f"{args.predicted_masks_path} does not exist.")
    if not os.path.exists(args.tile_predictions_path):
        raise FileNotFoundError(
            f"{args.tile_predictions_path} does not exist."
        )

    THRESH = get_threshold(args.predicted_masks_path)
    SLIDES = pd.read_csv(args.dataset, index_col='slide_id')

    # Define all directories, including task-specific ones
    REF_MASKS = args.reference_masks

    main(args)
