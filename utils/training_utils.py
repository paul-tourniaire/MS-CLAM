import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import torch
import torch.nn as nn
import torch.nn.functional as F
from topk import SmoothTop1SVM

from models.model_clam import CLAM_SB
from datasets.dataset_generic import save_splits
from utils.utils import (
    get_split_loader, print_network, get_optim, calculate_error
)


class Accuracy_Logger(object):
    """
    Tracks class-wise accuracy at either level (slide or tile).
    """

    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.probs = {"y_true": [], "y_prob": []}

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += \
                (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def log_batch_prob(self, Y_prob, Y):
        """For now, only works for binary classification."""
        self.probs["y_prob"].append(np.array(Y_prob)[:, 1])
        self.probs["y_true"].append(np.array(Y))

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count

    def get_recall(self):
        recall, _, _ = self.get_summary(1)
        return recall

    def get_specificity(self):
        specificity, _, _ = self.get_summary(0)
        return specificity

    def get_precision(self):
        tp = self.data[1]["correct"]
        fp = self.data[0]["count"] - self.data[0]["correct"]
        if tp + fp:
            precision = tp / (tp + fp)
        else:
            precision = 0.

        return precision

    def get_auc(self):
        y_prob = np.hstack(self.probs["y_prob"])
        y_true = np.hstack(self.probs["y_true"])
        return roc_auc_score(y_true, y_prob)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given
    patience.
    """

    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        args:
            patience (int): How long to wait after last time validation loss
                            improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss
                            improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter}"
                f" out of {self.patience}")
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f}"
                f" --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def get_training_results(datasets, cur, args, test_metrics):
    """
    Train the model for a single fold.

    args:
        datasets: tuple
            The three sets (train, validation, test) as
            `Generic_MIL_Dataset` objects (see datasets.dataset_generic).
        cur: int
            Fold index.
        args: Namespace
            Contains all arguments passed to the main.py script.
        test_metrics: dict
            Metrics dictionnary to fill during inference.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val/test splits...', end=' ', flush=True)
    train_split, val_split, test_split = datasets
    save_splits(
        datasets, ['train', 'val', 'test'],
        os.path.join(args.results_dir, 'splits_{}.csv'.format(cur))
    )
    print('Done!', flush=True)
    print("Training on {} samples".format(len(train_split)), flush=True)
    print("Validating on {} samples".format(len(val_split)), flush=True)
    print("Testing on {} samples".format(len(test_split)), flush=True)

    print('\nInit loss function...', end=' ', flush=True)
    if args.bag_loss == 'svm':
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!', flush=True)

    print('\nInit Model...', end=' ', flush=True)
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_size is not None:
        model_dict.update({"size_arg": args.model_size})

    if args.B > 0:
        model_dict.update({'k_sample': args.B})
    if args.B_gt > 0:
        model_dict.update({"gt_sample": args.B_gt})
    model_dict.update({'ms_clam': args.ms_clam})

    if args.inst_loss == 'svm':
        instance_loss_fn = SmoothTop1SVM(n_classes=2)
        if device.type == 'cuda':
            instance_loss_fn = instance_loss_fn.cuda()
    elif args.inst_loss == 'bce':
        instance_loss_fn = nn.BCELoss()
    else:
        reduction = "sum" if args.double_loader else "mean"
        weight = torch.tensor(args.inst_weighted_ce)
        instance_loss_fn = nn.CrossEntropyLoss(
            weight=weight, reduction=reduction
        )
        instance_loss_fn = instance_loss_fn.cuda()

    model_dict.update({"gt_dir": args.gt_dir})

    # load list of slides with incomplete annotations.
    # these slides are not sampled to be used as labeled.
    incomplete_annot = \
        pd.read_csv("./incomplete_annotations.csv")["slide_id"]

    if args.tile_labels_predefined:
        partial_split_file = os.path.join(
            "splits",
            f"{args.task}_{args.tile_labels_predefined}",
            f"splits_{cur}.csv"
        )
        partial_split = pd.read_csv(partial_split_file)["train"].dropna()
        if args.double_loader:
            has_annot = (
                train_split.slide_data["slide_id"].isin(partial_split)
                & train_split.slide_data["label"] == 1
            )
        else:
            has_annot = \
                train_split.slide_data["slide_id"].isin(partial_split)
        with_annot = train_split.slide_data[has_annot]["slide_id"]
        idx_list = list(with_annot.index)
        with_annot_train = with_annot.to_list()
    elif args.tile_labels_at_random:
        if args.task == "camelyon16":
            total_tum = 111
        else:
            raise NotImplementedError(
                f"The task {args.task} is not implemented!"
            )
        tum_slides_mask = (
            (train_split.slide_data["label"] == 1)
            & (~train_split.slide_data["slide_id"].isin(incomplete_annot))
        )
        tum_slides = train_split.slide_data[tum_slides_mask]["slide_id"]
        frac = args.tile_labels_at_random / 100
        n = round(frac * total_tum)
        if n >= len(tum_slides):
            with_annot = tum_slides
        else:
            with_annot = tum_slides.sample(n=n, random_state=args.seed)
        with_annot.to_csv(
            os.path.join(args.results_dir, f"with_annot_{cur}.csv"),
            header=["slide_id"],
            index=False
        )
        idx_list = list(with_annot.index)
        with_annot_train = with_annot.to_list()
    else:
        idx_list = []
        with_annot_train = []

    model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)

    # remove partially annotated slides from the available ones
    with_annot_val = val_split.slide_data[
        ~val_split.slide_data["slide_id"].isin(incomplete_annot)
    ]["slide_id"].to_list()

    # same for test set
    with_annot_test = test_split.slide_data[
        ~test_split.slide_data["slide_id"].isin(incomplete_annot)
    ]["slide_id"].to_list()

    model.relocate()
    print('Done!', flush=True)
    print_network(model)

    print('\nInit optimizer ...', end=' ', flush=True)
    optimizer = get_optim(model, args)
    print('Done!', flush=True)

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(
        train_split, training=True,
        testing=args.testing, weighted=args.weighted_sample,
        exp_weighted=args.exp_weighted_sample, idx_list=idx_list,
        weight_decay=args.sampler_weight_decay,
        init_weights_val=args.labeled_weights_init_val,
        double_loader=args.double_loader
    )
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)
    print('Done!', flush=True)

    if args.early_stopping:
        print('\nSetup EarlyStopping...', end=' ', flush=True)
        early_stopping = EarlyStopping(
            patience=20, stop_epoch=50, verbose=True
        )
        print('Done!', flush=True)

    else:
        early_stopping = None
    print('\n')

    train_metrics = {
        "total_loss": [], "slide_loss": [], "inst_loss": [],
        "att_loss_norm": [], "att_loss_tum": [], "error": [],
        "inst_recall": [], "inst_precision": [], "inst_specificity": [],
        "inst_auc": []
    }
    valid_metrics = {
        "total_loss": [], "slide_loss": [], "inst_loss": [],
        "att_loss_norm": [], "att_loss_tum": [], "error": [],
        "inst_recall": [], "inst_precision": [], "inst_specificity": [],
        "inst_auc": []
    }
    for epoch in range(args.max_epochs):
        if args.double_loader:
            train_loop_double_loader(
                epoch, model, train_loader, optimizer, args.n_classes,
                args.bag_weight, loss_fn,
                use_tile_labels=args.use_tile_labels,
                use_att_loss=args.use_att_loss,
                avail_annot=with_annot_train, att_weight=args.att_weight,
                metrics_dict=train_metrics,
            )
        else:
            train_loop_clam(
                epoch, model, train_loader, optimizer, args.n_classes,
                args.bag_weight, loss_fn,
                use_tile_labels=args.use_tile_labels,
                use_att_loss=args.use_att_loss,
                avail_annot=with_annot_train, att_weight=args.att_weight,
                metrics_dict=train_metrics
            )
        stop = validate_clam(
            cur, epoch, model, val_loader, args.n_classes, early_stopping,
            loss_fn, args.results_dir,
            use_tile_labels=args.use_tile_labels,
            bag_weight=args.bag_weight, avail_annot=with_annot_val,
            att_weight=args.att_weight, metrics_dict=valid_metrics
        )
        print('\n')

        if stop:
            break

    pd.DataFrame(train_metrics).to_csv(
        os.path.join(args.results_dir, f"train_metrics_fold_{cur}.csv"),
        index=False
    )
    pd.DataFrame(valid_metrics).to_csv(
        os.path.join(args.results_dir, f"valid_metrics_fold_{cur}.csv"),
        index=False
    )

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(
            args.results_dir, f"s_{cur}_checkpoint.pt"
        )))
    else:
        torch.save(
            model.state_dict(),
            os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")
        )

    print(f"\n+{'-' * 20} Summary {'-' * 20}+\n")

    _, val_error, val_auc, _, _ = summary(
        model, val_loader, args.n_classes,
        use_tile_labels=args.use_tile_labels, avail_annot=with_annot_val
    )
    print(
        f'Val error: {val_error:.4f}, ROC AUC: {val_auc:.4f}',
        flush=True
    )

    results_dict, test_error, test_auc, acc_logger, test_att_loss = summary(
        model, test_loader, args.n_classes,
        use_tile_labels=args.use_tile_labels, avail_annot=with_annot_test
    )
    print(
        f'Test error: {test_error:.4f}, ROC AUC: {test_auc:.4f}',
        flush=True
    )

    test_recall = acc_logger.get_recall()
    test_precision = acc_logger.get_precision()

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(
            f'class {i}: acc {acc:.4f}, correct {correct}/{count}',
            flush=True
        )

    att_save_path = os.path.join(
        args.results_dir, 'attention_scores', f'fold_{cur}'
    )
    inst_save_path = os.path.join(
        args.results_dir, 'tile-predictions', f'fold_{cur}'
    )
    os.makedirs(att_save_path, exist_ok=True)
    os.makedirs(inst_save_path, exist_ok=True)
    get_tile_predictions(model, test_loader, inst_save_path, att_save_path)

    print(f"\n+{'-' * 49}+\n")

    test_metrics["test_auc"].append(test_auc)
    test_metrics["test_acc"].append(1 - test_error)
    test_metrics["test_precision"].append(test_precision)
    test_metrics["test_recall"].append(test_recall)
    test_metrics["val_auc"].append(val_auc)
    test_metrics["val_acc"].append(1 - val_error)
    test_metrics["test_a_loss_norm"].append(test_att_loss[0])
    test_metrics["test_a_loss_tum"].append(test_att_loss[1])

    return results_dict


def train_loop_double_loader(
        epoch, model, loader, optimizer, n_classes, bag_weight, loss_fn=None,
        use_tile_labels=False, use_att_loss=False, avail_annot=None,
        att_weight=1.0, metrics_dict=None,
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    train_att_loss = {0: 0., 1: 0.}  # separate between normal and tumor
    train_total_loss = 0.
    inst_count = 0
    # count the slide with available attention loss for each class
    slide_w_a_loss_count = {0 : 0, 1: 0}

    for batch_idx, (batch_normal, batch_tum) in enumerate(zip(*loader)):
        data_n, label_n, slide_id_n = batch_normal
        data_t, label_t, slide_id_t = batch_tum
        data_n, data_t, label_n, label_t = (
            data_n.to(device), data_t.to(device),
            label_n.to(device), label_t.to(device)
        )
        slide_id_n, slide_id_t = slide_id_n.item(), slide_id_t.item()
        use_tile_labels_ = slide_id_t in avail_annot

        logits_t, Y_prob_t, Y_hat_t, _, instance_dict_t = model(
            data_t, label=label_t, instance_eval=True,
            use_tile_labels=use_tile_labels_, slide_id=slide_id_t
        )
        remain_to_sample = (
            np.count_nonzero(instance_dict_t["inst_labels"])
            - np.count_nonzero(instance_dict_t["inst_labels"] == 0)
            )
        logits_n, Y_prob_n, Y_hat_n, _, instance_dict_n = model(
            data_n, label=label_n, instance_eval=True,
            slide_id=slide_id_n,
            k_sample=remain_to_sample
        )
        logits = torch.cat((logits_n, logits_t))
        label = torch.cat((label_n, label_t))
        Y_hat = torch.cat((Y_hat_n, Y_hat_t))
        loss = loss_fn(logits, label)  # slide-level loss
        loss_value = loss.item()
        acc_logger.log(Y_hat_n, label_n)
        acc_logger.log(Y_hat_t, label_t)

        instance_loss = (
            instance_dict_n['instance_loss']
            + instance_dict_t['instance_loss']
        )
        inst_batch_size = (
            len(instance_dict_n["inst_preds"])
            + len(instance_dict_t["inst_preds"])
        )
        # reduction = 'sum', so CE must be averaged
        if not isinstance(model.instance_loss_fn, SmoothTop1SVM):
            instance_loss /= inst_batch_size
        inst_count += 2
        train_inst_loss += instance_loss.item()

        att_loss_n = instance_dict_n['att_loss']
        train_att_loss[label_n.item()] += att_loss_n.item()
        slide_w_a_loss_count[label_n.item()] += 1
        att_loss = att_loss_n
        att_loss_t = instance_dict_t["att_loss"]
        if att_loss_t is not None:
            train_att_loss[label_t.item()] += att_loss_t.item()
            slide_w_a_loss_count[label_t.item()] += 1
            att_loss += att_loss_t

        total_loss = (
            bag_weight * loss
            + (1 - bag_weight) * instance_loss
            + att_weight * att_loss
        )

        inst_preds = np.concatenate((
            instance_dict_n['inst_preds'],
            instance_dict_t['inst_preds']
        ))
        inst_labels = np.concatenate((
            instance_dict_n['inst_labels'],
            instance_dict_t['inst_labels']
        ))
        if instance_dict_n["inst_probs"].size == 0:
            print(slide_id_n)
            print(slide_id_t)
        inst_probs = np.concatenate((
            instance_dict_n['inst_probs'],
            instance_dict_t['inst_probs']
        ))
        inst_logger.log_batch(inst_preds, inst_labels)
        inst_logger.log_batch_prob(inst_probs, inst_labels)

        train_loss += loss_value
        train_total_loss += total_loss.item()
        train_error += calculate_error(Y_hat, label)

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(loader[0])
    train_error /= len(loader[0])
    train_total_loss /= len(loader[0])
    for i in train_att_loss.keys():
        train_att_loss[i] /= slide_w_a_loss_count[i] + 1e-10

    if inst_count > 0:
        train_inst_loss /= inst_count

    print(
        f"Epoch: {epoch} | Train -- total loss: {train_total_loss:.4f},"
        f" slide_loss: {train_loss:.4f},"
        f" clustering_loss: {train_inst_loss:.4f},"
        f" att_loss_norm: {train_att_loss[0]:.4f},"
        f" att_loss_tum: {train_att_loss[1]:.4f},"
        f" error: {train_error:.4f}",
        flush=True
    )

    for i in range(2):
        acc, correct, count = inst_logger.get_summary(i)
        print(
            f'class {i} clustering acc {acc}: correct {correct}/{count}',
            flush=True
        )

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f'class {i}: acc {acc}, correct {correct}/{count}', flush=True)

    metrics_dict["total_loss"].append(train_total_loss)
    metrics_dict["slide_loss"].append(train_loss)
    metrics_dict["inst_loss"].append(train_inst_loss)
    metrics_dict["att_loss_norm"].append(train_att_loss[0])
    metrics_dict["att_loss_tum"].append(train_att_loss[1])
    metrics_dict["error"].append(train_error)
    metrics_dict["inst_recall"].append(inst_logger.get_recall())
    metrics_dict["inst_precision"].append(inst_logger.get_precision())
    metrics_dict["inst_specificity"].append(inst_logger.get_specificity())
    metrics_dict["inst_auc"].append(inst_logger.get_auc())


def train_loop_clam(
        epoch, model, loader, optimizer, n_classes, bag_weight, loss_fn=None,
        use_tile_labels=False, use_att_loss=False, avail_annot=None,
        att_weight=1.0, metrics_dict=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    train_att_loss = {0: 0., 1: 0.}  # separate between normal and tumor
    train_total_loss = 0.
    inst_count = 0
    # count the slide with available attention loss for each class
    slide_w_a_loss_count = {0 : 0, 1: 1}

    for batch_idx, (data, label, slide_id) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_id[-1]
        # use_tile_labels_ is True if slide is tumorous and slide has
        # tile labels
        use_tile_labels_ = all(
            (use_tile_labels, label.item() == 1, (slide_id in avail_annot))
        )
        if use_att_loss == "total":
            add_att_loss = any((
                label.item() == 1 and slide_id in avail_annot,
                label.item() == 0
            ))
        elif use_att_loss == "partial":
            add_att_loss = label.item() == 0
        else:
            add_att_loss = False

        logits, Y_prob, Y_hat, _, instance_dict = model(
            data, label=label, instance_eval=True,
            use_tile_labels=use_tile_labels_, slide_id=slide_id
        )

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)  # slide-level loss
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        train_inst_loss += instance_loss.item()

        att_loss = instance_dict['att_loss']
        if att_loss is not None:
            att_loss_value = att_loss.item()
            train_att_loss[label.item()] += att_loss_value
            slide_w_a_loss_count[label.item()] += 1

        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss
        if add_att_loss:
            total_loss += att_weight * att_loss

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_probs = instance_dict['inst_probs']
        inst_logger.log_batch(inst_preds, inst_labels)
        inst_logger.log_batch_prob(inst_probs, inst_labels)

        train_loss += loss_value
        train_total_loss += total_loss.item()
        error = calculate_error(Y_hat, label)
        train_error += error

        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(loader)
    train_error /= len(loader)
    train_total_loss /= len(loader)
    for i in (0, 1):
        train_att_loss[i] /= slide_w_a_loss_count[i]

    if inst_count > 0:
        train_inst_loss /= inst_count

    print(
        f"Epoch: {epoch} | Train -- total loss: {train_total_loss:.4f},"
        f" slide_loss: {train_loss:.4f},"
        f" clustering_loss: {train_inst_loss:.4f},"
        f" att_loss_norm: {train_att_loss[0]:.4f},"
        f" att_loss_tum: {train_att_loss[1]:.4f},"
        f" error: {train_error:.4f}",
        flush=True
    )

    for i in range(2):
        acc, correct, count = inst_logger.get_summary(i)
        print(
            f'class {i} clustering acc {acc}: correct {correct}/{count}',
            flush=True
        )

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f'class {i}: acc {acc}, correct {correct}/{count}', flush=True)

    metrics_dict["total_loss"].append(train_total_loss)
    metrics_dict["slide_loss"].append(train_loss)
    metrics_dict["inst_loss"].append(train_inst_loss)
    metrics_dict["att_loss_norm"].append(train_att_loss[0])
    metrics_dict["att_loss_tum"].append(train_att_loss[1])
    metrics_dict["error"].append(train_error)
    metrics_dict["inst_recall"].append(inst_logger.get_recall())
    metrics_dict["inst_precision"].append(inst_logger.get_precision())
    metrics_dict["inst_specificity"].append(inst_logger.get_specificity())
    metrics_dict["inst_auc"].append(inst_logger.get_auc())


def validate_clam(
        cur, epoch, model, loader, n_classes, early_stopping=None,
        loss_fn=None, results_dir=None, use_tile_labels=False,
        bag_weight=0., avail_annot=[], att_weight=1.0, metrics_dict=None
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_slide_loss = 0.
    val_att_loss = {0: 0., 1: 0.}  # one loss per class
    inst_count = 0
    slide_w_a_loss_count = {0 : 0, 1: 0}  # count the slide per class

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (data, label, slide_id) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            slide_id = slide_id[-1]
            # use_tile_labels_ is True if label is 1 and slide has tile
            # labels
            use_tile_labels_ = all(
                (use_tile_labels, label.item() == 1, (slide_id in avail_annot))
            )

            logits, Y_prob, Y_hat, _, instance_dict = model(
                data, label=label, instance_eval=True,
                use_tile_labels=use_tile_labels_,
                slide_id=slide_id,
            )
            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)
            val_slide_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            if isinstance(model.instance_loss_fn, nn.CrossEntropyLoss):
                if model.instance_loss_fn.reduction == "sum":
                    instance_loss /= len(instance_dict["inst_labels"])
            att_loss = instance_dict['att_loss']
            att_loss_value = 0 if att_loss is None else att_loss.item()

            val_loss += loss.item()

            inst_count += 1
            val_inst_loss += instance_loss.item()
            if slide_id in avail_annot:
                val_att_loss[label.item()] += att_loss_value
                slide_w_a_loss_count[label.item()] += 1

            if instance_loss:
                inst_preds = instance_dict['inst_preds']
                inst_labels = instance_dict['inst_labels']
                inst_probs = instance_dict['inst_probs']
                inst_logger.log_batch(inst_preds, inst_labels)
                inst_logger.log_batch_prob(inst_probs, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)
    val_slide_loss /= len(loader)
    for i in (0, 1):
        val_att_loss[i] /= slide_w_a_loss_count[i] + 1e-10

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(
            labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(
                    binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print(
        f"\nValidation -- total loss: {val_loss:.4f},"
        f" slide_loss: {val_slide_loss:.4f},"
        f" val_inst_loss: {val_inst_loss / inst_count:.4f},"
        f" val_att_loss_norm: {val_att_loss[0]:.4f},"
        f" val_att_loss_tum: {val_att_loss[1]:.4f},"
        f" val_error: {val_error:.4f}, auc: {auc:.4f}")

    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print(f'class {i} clustering acc {acc}: correct {correct}/{count}')

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    metrics_dict["total_loss"].append(val_loss)
    metrics_dict["slide_loss"].append(val_loss)
    metrics_dict["inst_loss"].append(val_inst_loss)
    metrics_dict["att_loss_norm"].append(val_att_loss[0])
    metrics_dict["att_loss_tum"].append(val_att_loss[1])
    metrics_dict["error"].append(val_error)
    metrics_dict["inst_recall"].append(inst_logger.get_recall())
    metrics_dict["inst_precision"].append(inst_logger.get_precision())
    metrics_dict["inst_specificity"].append(inst_logger.get_specificity())
    metrics_dict["inst_auc"].append(inst_logger.get_auc())

    if early_stopping:
        assert results_dir
        early_stopping(
            epoch, val_loss, model,
            ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(
        model, loader, n_classes, use_tile_labels=False, avail_annot=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    # test_loss = 0.
    test_error = 0.
    att_loss = {0: 0., 1: 0.}
    slide_w_a_loss_count = {0: 0, 1: 0}

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, _) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        # use_tile_labels_ is True if label is 1 and slide has tile labels
        use_tile_labels_ = all(
            (use_tile_labels, label.item() == 1, (slide_id in avail_annot))
        )
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, instance_dict = model(
                data, label=label, instance_eval=True,
                use_tile_labels=use_tile_labels_, slide_id=slide_id
            )

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {
            'slide_id': np.array(slide_id),
            'prob': probs,
            'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

        if slide_id in avail_annot:
            att_loss_ = instance_dict["att_loss"]
            att_loss_value = 0 if att_loss is None else att_loss_.item()
            att_loss[label.item()] += att_loss_value
            slide_w_a_loss_count[label.item()] += 1

    test_error /= len(loader)
    for i in (0, 1):
        att_loss[i] /= slide_w_a_loss_count[i]

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(
            all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(
                    binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger, att_loss


def get_tile_predictions(model, loader, inst_save_path, att_save_path):
    '''
    At the end of the training, saves tile predictions and attention scores.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slide_ids = loader.dataset.slide_data['slide_id']
    for batch_idx, (data, label, _) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            A, h = model.attention_net(data)
            A = torch.transpose(A, 1, 0)
            A_raw = A
            A = F.softmax(A, dim=1)
            tile_logits = model.instance_classifiers(h)

            probs = F.softmax(tile_logits, dim=1)
            probs = probs.detach().cpu().numpy()
            df = pd.DataFrame(probs, columns=["prob_0", "prob_1"])
            df.to_csv(os.path.join(
                inst_save_path, f"{slide_id}.csv"),
                index=False
            )

            A_cat = torch.cat((A, A_raw), dim=0).transpose(1, 0)
            A_cat = A_cat.detach().cpu().numpy()
            columns = (
                [f"a_k_{i}" for i in range(A.size(0))]
                + [f"a_k_{i}_logits" for i in range(A.size(0))]
            )
            df_att = pd.DataFrame(A_cat, columns=columns)
            df_att.to_csv(
                os.path.join(att_save_path, f"{slide_id}.csv")
            )
