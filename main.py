import os
import pprint
import random
import sys

import numpy as np
import pandas as pd
import torch

from datasets.dataset_generic import Generic_MIL_Dataset
from parser import Parser
from utils.file_utils import save_pkl
from utils.utils import f1_score
from utils.training_utils import get_training_results


def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    folds = np.arange(start, end)
    final_metrics = {
        "folds": folds,
        "test_auc": [],
        "val_auc": [],
        "test_acc": [],
        "val_acc": [],
        "test_recall": [],
        "test_precision": [],
        "test_f1": [],
        "test_a_loss_tum": [],
        "test_a_loss_norm": []
    }
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, csv_path=f'{args.split_dir}/splits_{i}.csv'
        )
        datasets = (train_dataset, val_dataset, test_dataset)
        results = get_training_results(datasets, i, args, final_metrics)
        # add f1-score based on precision and recall results
        final_metrics["test_f1"].append(f1_score(
            final_metrics["test_precision"][-1],
            final_metrics["test_recall"][-1]
        ))
        # write results to pkl
        filename = os.path.join(args.results_dir, f'split_{i}_results.pkl')
        save_pkl(filename, results)

    final_df = pd.DataFrame(final_metrics)
    mean_df = final_df.mean(axis=0).to_frame().T.drop("folds", axis=1)
    std_df = final_df.std(axis=0).to_frame().T.drop("folds", axis=1)
    final_mean_std = pd.concat([mean_df, std_df], ignore_index=True)

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    final_mean_std.to_csv(
        os.path.join(args.results_dir, "summary_avg.csv"), index=False
    )


if __name__ == "__main__":
    description = (
        "Run MS-CLAM on 5 different splits of a tumor "
        "classification dataset."
    )
    parser = Parser(description)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    encoding_size = 1024
    settings = {
        'num_splits': args.k,
        'k_start': args.k_start,
        'k_end': args.k_end,
        'task': args.task,
        'max_epochs': args.max_epochs,
        'results_dir': args.results_dir,
        'lr': args.lr,
        'experiment': args.exp_code,
        'reg': args.reg,
        'label_frac': args.label_frac,
        'bag_loss': args.bag_loss,
        'seed': args.seed,
        'model_size': args.model_size,
        "use_drop_out": args.drop_out,
        'weighted_sample': args.weighted_sample,
        'opt': args.opt,
        'bag_weight': args.bag_weight,
        'inst_loss': args.inst_loss,
        'B': args.B,
        'B_gt': args.B_gt,
        'use_tile_labels': args.use_tile_labels,
        'use_att_loss': args.use_att_loss,
        'att_weight': args.att_weight,
        'tile_labels_predefined': args.tile_labels_predefined,
        'to_exclude': args.to_exclude,
        'exp_weighted_sample': args.exp_weighted_sample,
        'sampler_weight_decay': args.sampler_weight_decay,
        'labeled_weights_init_val': args.labeled_weights_init_val,
        'ms_clam': args.ms_clam,
        "accumulate_gradient": args.accumulate_gradient,
        "double_loader": args.double_loader,
        "tile_labels_at_random": args.tile_labels_at_random,
        "inst_weighted_ce": args.inst_weighted_ce
    }

    print('\nLoad Dataset', flush=True)

    if args.task == 'task_1_tumor_vs_normal':
        args.n_classes = 2
        dataset = Generic_MIL_Dataset(
            csv_path='dataset_csv/tumor_vs_normal_dummy_clean.csv',
            data_dir=os.path.join(args.data_root_dir,
                                  'tumor_vs_normal_resnet_features'),
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
            patient_strat=False,
            ignore=[])

    elif args.task == 'camelyon16':
        args.n_classes = 2
        dataset = Generic_MIL_Dataset(
            csv_path=f'dataset_csv/camelyon16.csv',
            data_dir=os.path.join(args.data_root_dir, 'features'),
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
            patient_strat=False,
            ignore=[]
        )

    else:
        raise NotImplementedError

    args.results_dir = os.path.join(
        args.results_dir, str(args.exp_code) + f'_s{args.seed}'
    )
    os.makedirs(args.results_dir, exist_ok=True)

    if args.split_dir is None:
        args.split_dir = os.path.join(
            'splits', args.task + '_{}'.format(int(args.label_frac * 100)))
    else:
        args.split_dir = os.path.join('splits', args.split_dir)

    print('split_dir: ', args.split_dir, flush=True)
    assert os.path.isdir(args.split_dir)

    settings.update({'split_dir': args.split_dir})

    with open(args.results_dir + f'/experiment_{args.exp_code}.txt', 'w') \
            as f:
        pprint.pprint(settings, stream=f)

    print("################# Settings ###################", flush=True)
    for key, val in settings.items():
        print("{}:  {}".format(key, val), flush=True)

    results = main(args)
    print("finished!", flush=True)
    print("end script", flush=True)
    sys.exit(0)
