import argparse


class Parser(argparse.ArgumentParser):
    '''Parse the command line arguments.'''
    def __init__(self, description):
        super().__init__(description)
        # Generic training settings
        self.add_argument(
            '--data_root_dir', type=str, default=None, help='data directory'
        )
        self.add_argument(
            '--max_epochs', type=int, default=200,
            help='maximum number of epochs to train (default: 200)'
        )
        self.add_argument(
            '--lr', type=float, default=1e-4,
            help='learning rate (default: 0.0001)'
        )
        self.add_argument(
            '--label_frac', type=float, default=1.0,
            help='fraction of training labels (default: 1.0)'
        )
        self.add_argument(
            '--reg', type=float, default=1e-5,
            help='weight decay (default: 1e-5)'
        )
        self.add_argument(
            '--seed', type=int, default=1,
            help='random seed for reproducible experiment (default: 1)'
        )
        self.add_argument(
            '--k', type=int, default=10, help='number of folds (default: 10)'
        )
        self.add_argument(
            '--k_start', type=int, default=-1,
            help='start fold (default: -1, last fold)')
        self.add_argument(
            '--k_end', type=int, default=-1,
            help='end fold (default: -1, first fold)'
        )
        self.add_argument(
            '--results_dir', default='./results',
            help='results directory (default: ./results)'
        )
        self.add_argument(
            '--split_dir', type=str, default=None,
            help="manually specify the set of splits to use, "
            " instead of infering from the task and "
            " label_frac argument (default: None)"
        )
        self.add_argument(
            '--log_data', action='store_true', default=False,
            help='log data using tensorboard'
        )
        self.add_argument(
            '--testing', action='store_true', default=False,
            help='debugging tool'
        )
        self.add_argument(
            '--early_stopping', action='store_true', default=False,
            help='enable early stopping'
        )
        self.add_argument(
            '--opt', type=str, choices=['adam', 'sgd'], default='adam'
        )
        self.add_argument(
            '--drop_out', action='store_true', default=False,
            help='enable dropout (p=0.25)'
        )
        self.add_argument(
            '--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
            help='slide-level classification loss function (default: ce)'
        )
        self.add_argument(
            '--exp_code', type=str, help='experiment code for saving results'
        )
        self.add_argument(
            '--weighted_sample', action='store_true', default=False,
            help='enable weighted sampling'
        )
        self.add_argument(
            '--model_size', type=str, choices=['small', 'big', 'tiny'],
            default='small',
            help='Size of the model (number of neurons per layer)'
        )
        self.add_argument(
            '--task', type=str,
            choices=['task_1_tumor_vs_normal', 'camelyon16']
        )
        self.add_argument(
            '--inst_loss', type=str, choices=['svm', 'ce', 'bce', None],
            default=None,
            help='instance-level clustering loss function (default: None)'
        )
        self.add_argument(
            '--bag_weight', type=float, default=0.7,
            help='clam: weight coefficient for bag-level loss (default: 0.7)'
        )
        self.add_argument(
            '--B', type=int, default=8,
            help='number of positive/negative patches to sample for clam'
        )
        self.add_argument(
            '--ms-clam', action="store_true", default=False,
            help="MS-CLAM setting: no negative pseudo-labels for normal slides"
        )
        self.add_argument(
            '--use-tile-labels', action='store_true', default=False,
            help='Use tile ground truth labels when available'
        )
        self.add_argument(
            "--B_gt", type=int, default=128,
            help="number of patches to sample from ground truth regions"
        )
        self.add_argument(
            "--gt-dir",
            help="Path to the files containing the labels of the tiles. The "
            "file should be a pickle file containing a list of indexes which "
            "correspond to the indexes of the tiles in the .pt or .h5 file."
        )
        self.add_argument(
            '--use_att_loss', type=str, default=None,
            choices=["total", "partial"],
            help="Use entropy loss on attention weights. "
            "When using `partial`, the attention loss will only be used for "
            "negative bags."
        )
        self.add_argument(
            "--att_weight", type=float, default=1.0,
            help="attention loss coefficient"
        )
        self.add_argument(
            "--tile-labels-predefined", type=str,
            help="Use tile labels from a predefined selection. "
            "Under the `split_dir` directory, a folder corresponding "
            "to the dataset with the desired amount of labeled slides "
            "should be available."
        )
        self.add_argument(
            "--tile-labels-at-random", type=int,
            help="Select randomly among the slides the ones that are "
            "tile-labeled, as opposed to what tile-labels-predefined does. "
            "The quantity represents the percentage of labeled slides."
        )
        self.add_argument(
            "--to-exclude",
            help="The path to a csv file indicating slide ids which should "
            "not be considered when sampling tile-labeled slides."
        )
        # Exponential weighted sampling arguments
        self.add_argument(
            "--exp_weighted_sample",
            action="store_true",
            default=False,
            help="Use ExponentialWeightedSampler (only for mixed supervision)."
        )
        self.add_argument(
            "--sampler_weight_decay", type=float, default=0.9,
            help="Decay value of the weights for the exponential "
            "weighted sampler."
        )
        self.add_argument(
            "--labeled_weights_init_val", type=float, default=100.,
            help="Initial weight values of the slides with annotations "
            "(when using ExponentialWeightedSampler)."
        )
        self.add_argument(
            "--accumulate-gradient", type=int, default=1,
            help="Defines the number of steps when accumulating gradient. "
            "Default: 1 (no accumulation)."
        )
        self.add_argument(
            "--double-loader", action="store_true", default=False,
            help="Use two loaders for the slides instead of one. "
            "Tumorous and normal slides are loaded separately."
        )
        self.add_argument(
            "--inst-weighted-ce", nargs='*', type=float, default=[1., 1.],
            help="Use weights in the instance loss function."
        )


if __name__ == '__main__':
    description = (
        "Run MS-CLAM on 5 different splits of a tumor "
        "classification dataset."
    )
    parser = Parser(description)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f'{k}: {v}')