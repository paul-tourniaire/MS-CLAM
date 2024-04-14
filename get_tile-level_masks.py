import os
import numpy as np
import h5py
import cv2
import openslide
import pandas as pd
import multiprocessing
import argparse
from tqdm import tqdm


description = '''Computes slide masks based on tile-level predictions
made by instance classifiers'''

def update_pbar(*a):
    PBAR.update()


def createTileMasks(slide_list, subdir, save_dir, thresh=0.5, overwrite=False):
    '''
    For each slide in slide list, computes a binary mask at a lower mag
    (1/5) based on tile classification results.
    '''
    for i, slide in enumerate(slide_list):
        slide_name = os.path.splitext(slide)[0]
        save_path = os.path.join(save_dir, slide_name + "-tile_level_mask.png")

        if not os.path.exists(save_path) or overwrite:
            # features file to retrieve tile coords
            features_file = os.path.join(
                args.h5_files, f'{slide_name}.h5'
            )
            with h5py.File(features_file, "r") as infile:
                coordinates = infile["coords"][:]
            # check slide image extension, get slide width and height
            if os.path.exists(os.path.join(SLIDE_DIR, f"{slide_name}.tif")):
                slide_path = os.path.join(SLIDE_DIR, f"{slide_name}.tif")
                wsi = openslide.open_slide(slide_path)
                width, height = wsi.level_dimensions[5]
                RF = 32
                patch_size = PATCH_SIZE // RF
            # load tile predictions file in dataframe
            df = pd.read_csv(os.path.join(subdir, f"{slide_name}.csv"))
            # scale coords to current thumbnails dims
            df["start_w"] = (coordinates[:, 0] // RF).astype("int")
            df["start_h"] = (coordinates[:, 1] // RF).astype("int")
            # get only tumorous tiles based on thresh
            df = df[df["prob_1"] >= thresh].reset_index(drop=True)
            # create empty mask canva
            score_map = np.zeros((height, width))
            # if no tumorous tiles, skip to result
            if not df.empty:
                # try:
                # get image indexes to change to 1
                idx = np.vstack([
                    np.mgrid[
                        df.iloc[i]["start_h"]:df.iloc[i]["start_h"] + patch_size,
                        df.iloc[i]["start_w"]:df.iloc[i]["start_w"] + patch_size
                    ].reshape(2,-1).T for i in df.index
                ]).astype("int")
                # Remove border tiles that are not exactly of dimensions
                # (patch_size, patch_size)
                idx = idx[idx[:, 0] < (height - patch_size)]
                idx = idx[idx[:, 1] < (width - patch_size)]
                score_map[idx[:, 0], idx[:, 1]] = 255
                # except ValueError:
            score_map = np.stack(
                (score_map, score_map, score_map), axis=2
            )

            cv2.imwrite(save_path, score_map)

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--tile_path", required=True, help="Path to the tile predictions."
    )
    parser.add_argument(
        "--dst_dir", required=True, help="Path to the destination directory."
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
    parser.add_argument(
        "--thresh", type=float, default=0.5,
        help="Probability threshold for the computation of the binary mask"
    )
    parser.add_argument(
        "--slide_id",
        help="Make the operation for a single slide. .csv extension"
        " must be explicit."
    )
    parser.add_argument(
        "--patch_size", type=int, default=256,
        help="Original size of the patches"
    )
    parser.add_argument(
        "--overwrite_existing", action="store_true",
        help="Overwrite masks even if they exist"
    )
    args = parser.parse_args()
    print()

    PATCH_SIZE = args.patch_size
    SLIDE_DIR = args.slide_dir

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)
    if os.path.exists(args.tile_path):
        slide_list = os.listdir(args.tile_path)
        print(f"Number of slides to process: {len(slide_list)}")
        save_dir = os.path.join(
            args.dst_dir,
            f"th-{args.thresh}"
        )
        os.makedirs(save_dir, exist_ok=True)

        if not args.slide_id:
            num_slides = len(slide_list)
            if num_processes > num_slides:
                num_processes = num_slides
            images_per_process = num_slides / num_processes
            tasks = []
            for num_process in range(1, num_processes + 1):
                start_index = \
                    (num_process - 1) * images_per_process + 1
                end_index = num_process * images_per_process
                start_index = int(start_index)
                end_index = int(end_index)
                sublist = slide_list[start_index - 1:end_index]
                tasks.append((
                    sublist, args.tile_path, save_dir,
                    args.thresh,
                    args.overwrite_existing
                ))
            # start tasks
            PBAR = tqdm(total=len(tasks))
            results = []
            for t in tasks:
                results.append(
                    pool.apply_async(
                        createTileMasks, t,
                        callback=update_pbar
                    )
                )
            for result in results:
                result.get()
        else:
            createTileMasks(
                [args.slide_id], args.tile_path, save_dir, args.thresh,
                args.overwrite_existing
            )
    else:
        print()
