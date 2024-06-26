#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir ./data/camelyon16 \
--max_epochs 90 \
--lr 2e-4 \
--k 5 \
--results_dir ./results/camelyon16/ \
--split_dir camelyon16 \
--drop_out \
--bag_loss ce \
--exp_code msclam_exp_0 \
--task camelyon16 \
--inst_loss ce \
--bag_weight 0.6 \
--B 8 \
--ms_clam \
--use_tile_labels \
--B_gt 1024 \
--use_att_loss total \
--att_weight 1.0 \
--exp_weighted_sample \
--sampler_weight_decay 0.90 \
--labeled_weights_init_val 90 \
--double_loader \
--tile_labels_at_random 10 \
--to_exclude ./dataset_csv/camelyon16_incomplete_annotations.csv \
--gt_dir ./data/camelyon16/gt_patches_indexes
