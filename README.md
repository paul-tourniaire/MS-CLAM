# MS-CLAM

Code associated to the [article](https://www.sciencedirect.com/science/article/abs/pii/S1361841523000245): *MS-CLAM: Mixed supervision for the classification and localization of tumors in Whole Slide Images*, published in **Medical Image Analysis**, Volume 85, April 2023.


MS-CLAM is a mixedly-supervised model for digital pathology. It integrates mixed supervision, i.e., *the concurrent use of slide- and tile-level supervision for whole-slide image tumor classification and localization*. It is based on the well-known [CLAM model](https://github.com/mahmoodlab/CLAM), a weakly-supervised model for interpretable WSI classification. The code allows the user to choose which amount of tile-level labeled slides they wish to use during training, either prespecified or selected randomly.

In this repository, one should find all the necessary elements to:

- train the MS-CLAM model with a specific amount of tile-level labeled slides
- generate attention maps, which map the attention scores of the tiles to a color map showing in red the highest ones, and in blue the lowest.
- generate tumor masks, which are binary masks where each tile that is predicted as tumorous appears in white

## Notes

- The slides are expected to be saved as .h5 or .pt files (see the [CLAM](https://github.com/mahmoodlab/CLAM) repo to see how it can be done).
- The h5 files should contain two keys:
  1. `coords`, where an array of size Nx2 should be stored, with the coordinates for all the tiles (openslide format, level 0)
  2. `features`, where an array of size Nxd should be stored, and each row is the latent space representation of a tile in the slide. N is the number of tiles in the slide, and d is the dimension of the embeddings. For instance, if an Imagenet pretrained Resnet-50 is used to extract features, d=1024 (using the implementation of the authors of CLAM). This array is also contained in the .pt files (for faster and easier training).
- The pickle files that contain the labeled tiles indexes should have the '.pkl' extension. These files contain lists of indexes that match the ones in the .pt or .h5 files. Each index in the list corresponds to a tumorous tile.
- Examples of such files are located in the `data` folder.
- If using the `--tile_labels_predefined` flag, then the `splits` directory should contain a subdirectory for the dataset, and another one for the dataset with only the annotated slides in the training set. The structure of the `splits` directory then reads (ratio defines the percentage of annotated slides):

```bash
splits/
  ├── dataset_name
        ├── splits_0.csv
        ├── splits_1.csv
        └── ...
  ├── dataset_name_<ratio>
        ├── splits_0.csv
        ├── splits_1.csv
        └── ...
  └── ...
```

## Virtual environment
The file msclam.yml contains the necessary packages for this repository. Simply create a conda virtual environment with:

```shell
conda env create -f msclam.yml
```
Then, activate it with:

```shell
conda activate msclam
```

## Training
To train the model, simply launch `./main.sh` after you have activated the conda virtual environment. If you wish to use predefined tile labels instead of randomly chosen ones, simply change the `--tile_labels_at_random` flag to `--tile_labels_predefined`.

## Inference

After the `main.py` program finishes, the directory given to the `--results_dir` parameter will contain two csv files: one for the attention scores, and one for the tile-level predictions. The first one is used by the `attention_maps.py` script to generate and save attention maps. The second one is used by the `calculate_dice_score.py` and `get_tile-level_masks.py` scripts to obtain the tile-level tumor maps and tile-level metrics (Dice score, specificity).

The following commands are given assuming that the experiment name in the `main.sh` file has not been changed.

### Attention maps

<img src="attention_maps.png" height="250" />

The following command will generate attention_maps for the available slides

```shell
python attention_maps.py --att_scores_path ./results/attention_scores/fold_0 --dst_dir ./results/camelyon16/msclam_exp_0_s1/attention_maps/fold_0 --slide-dir ./data/camelyon16/slides/ --h5-files ./data/camelyon16/features/h5_files
```

### Tile-level metrics and maps

First, the tile-level maps should be created before the metrics are computed:

```shell
python get_tile-level_masks.py --tile_path ./results/camelyon16/msclam_exp_0_s1/tile_predictions/fold_0 --dst_dir ./results/camelyon16/predicted_masks/fold_0 --slide-dir ./data/camelyon16/slides --thresh 0.5 --patch_size 256
```

Then, the Dice score and the specificity are obtained thanks to:

```shell
python calculate_dice_score.py --predicted-masks-path ./results/camelyon16/msclam_exp_0_s1/predicted_masks/fold_0/th-0.5/ --tile_predictions_path ./results/camelyon16/msclam_exp_0_s1/tile_predictions/fold_0 --dataset ./dataset_csv/camelyon16.csv --reference_masks ./data/camelyon16/reference_masks --tile-mask-gt
```

The `--tile-mask-gt` flag indicates that the reference mask should be the tile-accurate mask, instead of the pixel-accurate one. In the tile-accurate-mask, entire tiles are labeled True or False whether they contain tumor or not.
