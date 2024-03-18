import h5py
import openslide
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd
import os
import logging
import sys
import multiprocessing
from functools import partial
import argparse


def compute_attention_map(slide_id, a_path, save_dir, psize=256):
    # Init. scaler
    scaler = MinMaxScaler()
    # Load attention scores
    att_scores = pd.read_csv(os.path.join(a_path, f"{slide_id}.csv"))
    if "a_k_1" in att_scores.columns:
        att_scores = np.asarray(att_scores["a_k_1"])
    else:
        att_scores = np.asarray(att_scores["a_k_0"])
    att_scores = np.expand_dims(att_scores, axis=1)
    att_scores_scaled = scaler.fit_transform(att_scores)
    # Init. colormap
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    colors = cmap(att_scores_scaled).squeeze()[..., :3]
    colors = np.rint(colors * 255).astype(np.int32)

    # Load image
    if os.path.exists(os.path.join(SLIDE_DIR, f"{slide_id}.tif")):
        slide = openslide.open_slide(
            os.path.join(SLIDE_DIR, f"{slide_id}.tif")
        )
        thumbnail = slide.read_region(
            (0, 0), 5, slide.level_dimensions[5]).convert("RGB")
        thumbnail = np.asarray(thumbnail)
        psize = psize // 2**5
    canvas = np.zeros_like(thumbnail)

    with h5py.File(os.path.join(H5_PATH, f"{slide_id}.h5"), "r") as f:
        coords = f["coords"][:]

    for i in range(len(coords)):
        x, y = coords[i] // 2**5
        color = tuple(int(c) for c in colors[i])
        canvas = cv2.rectangle(
            canvas, (x, y),
            (x + psize, y + psize),
            color=color, thickness=-1)
    result = cv2.addWeighted(thumbnail, 0.5, canvas, 0.5, 0)
    cv2.imwrite(
        os.path.join(save_dir, f"{slide_id}.png"),
        cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        os.path.join(save_dir, f"{slide_id}_canvas.png"),
        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    )


def main(args):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    slide_ids = [
        os.path.splitext(f)[0]
        for f in os.listdir(args.att_scores_path)
    ]
    a_path = args.att_scores_path
    save_dir = args.dst_dir
    os.makedirs(save_dir, exist_ok=True)
    with multiprocessing.Pool() as pool:
        func = partial(
            compute_attention_map, a_path=a_path, save_dir=save_dir,
            psize=args.psize
        )
        pool.map(func, slide_ids)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save attention maps."
    )
    parser.add_argument(
        "--att_scores_path", required=True,
        help="Path to the attention scores."
    )
    parser.add_argument(
        "--dst_dir", required=True, help="Path to the destination folder."
    )
    parser.add_argument(
        "--slide_dir", required=True,
        help='Path to the directory containing the slides.'
    )
    parser.add_argument(
        "--h5_files", required=True,
        help="Path to the directory containing the h5 files "
        "corresponding to the slides. These files should contain "
        "the coordinates of the tiles"
    )
    parser.add_argument("--psize", type=int, default=256, help="Patch size.")
    args = parser.parse_args()

    SLIDE_DIR = args.slide_dir
    H5_PATH = args.h5_files
    os.makedirs(args.dst_dir, exist_ok=True)
    main(args)
    sys.exit()
