import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import (
    DataLoader, Sampler, WeightedRandomSampler,
    RandomSampler, SequentialSampler, sampler)
import torch.optim as optim
import math
from itertools import islice
import collections
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SubsetSequentialSampler(Sampler):
    """
    Samples elements sequentially from a given list of indices,
    without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ExponentialWeightedRandomSampler(Sampler):
    r"""Derived from the :class:`~torch.utils.data.WeightedRandomSampler`.
    The weights of the multinomial law sampling the data are updated
    after each epoch by a factor decay.

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up
            to one
        num_samples (int): number of samples to draw
        idx_list (array-like): indexes of the weights that have to be changed
            after each epoch.
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
        decay (bool): factor by which weights at indexes idx_list are multiplied.
    """
    num_samples: int
    replacement: bool

    def __init__(self, weights, num_samples: int, idx_list,
                 replacement: bool = True, generator=None, decay=0.9) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             f"value, but got num_samples={num_samples}"
                )
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.idx_list = idx_list
        self.decay = decay

    def __iter__(self):
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement,
            generator=self.generator
        )
        self.update_weights()
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def update_weights(self):
        self.weights[self.idx_list] *= self.decay
        if torch.sum(self.weights[self.idx_list]) <= len(self.idx_list):
            self.weights[self.idx_list] = 1.
            self.decay = 1.

def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    # slide_id = torch.tensor([item[2] for item in batch])
    slide_id = np.array([item[2] for item in batch])
    return [img, label, slide_id]


def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
    kwargs = (
        {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers}
        if device.type == "cuda" else {})
    loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=sampler.SequentialSampler(dataset),
        collate_fn=collate_MIL, **kwargs)
    return loader


def get_split_loader(
        split_dataset, training=False, testing=False, weighted=False,
        exp_weighted=False, idx_list=None, weight_decay=0.9,
        init_weights_val=100., double_loader=False):
    """
        return either the validation loader or training loader
        Args:
            split_dataset (Dataset): dataset from which to load the data.
            training (bool): if ``True`` selects a random-like Sampler.
            testing (bool): debugging parameter.
            weighted (bool): if ``True``, sets the sampler to
                WeightedRandomSampler.
            exp_weighted (bool): if ``True``, sets the sampler to
                ExponentialWeightedRandomSampler
            idx_list (array_like): specifies indexes of which weights
                are updated when exp_weighted is set to ``True``.
            weight_decay (float): sets the weights decay for the sampler
                when exp_weighted is set to ``True``.
            init_weights_val (float): initial value of the weights for
                the slides with annotations.
            double_loader (bool): if True, returns two loaders instead of one,
                corresponding to each class.

        Returns:
            loader (DataLoader): pytorch DataLoader of the split_dataset.

    """
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(
                    split_dataset)
                loader = DataLoader(
                    split_dataset, batch_size=1,
                    sampler=WeightedRandomSampler(weights, len(weights)),
                    collate_fn=collate_MIL, **kwargs)
            elif exp_weighted:
                if double_loader:
                    weights_normal, weights_tum = \
                        make_weights_for_double_loader(split_dataset)
                    weights_tum[idx_list] = init_weights_val
                    sampler_normal = WeightedRandomSampler(
                        weights=weights_normal,
                        num_samples=np.count_nonzero(weights_normal),
                        replacement=False
                    )
                    sampler_tum = ExponentialWeightedRandomSampler(
                        weights=weights_tum,
                        num_samples=np.count_nonzero(weights_normal),
                        idx_list=idx_list,
                        decay=weight_decay,
                        replacement=True
                    )
                    loader_normal = DataLoader(
                        split_dataset, sampler=sampler_normal,
                        collate_fn=collate_MIL, **kwargs
                    )
                    loader_tum = DataLoader(
                        split_dataset, sampler=sampler_tum,
                        collate_fn=collate_MIL, **kwargs
                    )
                    loader = (loader_normal, loader_tum)
                else:
                    weights = np.ones((len(split_dataset,)))
                    weights[idx_list] = init_weights_val
                    sampler = ExponentialWeightedRandomSampler(
                        weights, len(weights), idx_list, decay=weight_decay
                    )
                    loader = DataLoader(
                        split_dataset, batch_size=1,
                        sampler=sampler,
                        collate_fn=collate_MIL, **kwargs)
            else:
                loader = DataLoader(
                    split_dataset, batch_size=1,
                    sampler=RandomSampler(split_dataset),
                    collate_fn=collate_MIL, **kwargs)
        else:
            loader = DataLoader(
                split_dataset, batch_size=1,
                sampler=SequentialSampler(split_dataset),
                collate_fn=collate_MIL, **kwargs)

    else:
        ids = np.random.choice(
            np.arange(len(split_dataset), int(len(split_dataset) * 0.1)),
            replace=False)
        loader = DataLoader(
            split_dataset, batch_size=1,
            sampler=SubsetSequentialSampler(ids),
            collate_fn=collate_MIL, **kwargs)

    return loader


def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net, flush=True)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params, flush=True)
    print(
        'Total number of trainable parameters: %d' % num_params_train,
        flush=True)


def generate_split(
        cls_ids, val_num, test_num, samples, n_splits=5,
        seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)

    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None:  # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)  # all indices of this class
            val_ids = np.random.choice(possible_indices, val_num[c], replace=False)  # validation ids

            remaining_ids = np.setdiff1d(possible_indices, val_ids)  # indices of this class left after validation
            all_val_ids.extend(val_ids)

            if custom_test_ids is None:  # sample test split

                test_ids = np.random.choice(
                    remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)

            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def calculate_error(Y_hat, Y):
    if Y_hat.size(1) == 1:  # multiple values
        Y_hat = Y_hat.squeeze()
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [
        N / len(dataset.slide_cls_ids[c])
        for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)


def make_weights_for_double_loader(dataset):
    weights_tum = np.zeros((len(dataset),))
    for idx in range(len(dataset)):
        if dataset.getlabel(idx) == 1:
            weights_tum[idx] = 1.
    weights_normal = 1. - weights_tum
    return torch.DoubleTensor(weights_normal), torch.DoubleTensor(weights_tum)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def f1_score(prec, rec):
    return 2 * prec * rec / (prec + rec)
