import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
import pickle
import os
from utils.entropy_utils import shannon_entropy


class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class TileClassifier(nn.Module):
    def __init__(self, dropout, in_dim=512):
        super(TileClassifier, self).__init__()
        network = [nn.Linear(in_dim, 2)]
        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)


class CLAM_SB(nn.Module):
    """
    Args
    -------------
    gate : bool
        Whether to use gated attention network.
    size_arg : str
        Config for network size.
    dropout : bool, float
        Whether to use dropout (p = 0.25).
    k_sample : int
        Number of positive/neg patches to sample for instance-level training.
    n_classes : int
        Number of classes.
    gt_dir : str
        Path to the instance ground-truth labels.
    instance_loss_fn : nn.Module
        Loss function to supervise instance-level training.
    gt_sample : int
        Number of positive/neg patches to sample for instance-level
        training when using instance-level supervision.
    ms_clam : bool
        If True, uses the Mixedly-Supervised CLAM implementation of the model.
    """

    def __init__(
            self,
            gate=True,
            size_arg="small",
            dropout=False,
            k_sample=8,
            n_classes=2,
            gt_dir=None,
            instance_loss_fn=nn.CrossEntropyLoss(),
            gt_sample=128,
            ms_clam=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {
            "small": [1024, 512, 256],
            "big": [1024, 512, 384],
            "tiny": [512, 256, 128]  # for 512-dim input features
        }
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(
                L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.instance_classifiers = TileClassifier(
            0.25, size[1]
        )
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes

        if gt_dir:  # path to the ground truth instance indexes
            self.gt_dir = gt_dir
        self.gt_sample = gt_sample
        self.ms_clam = ms_clam
        self.f_entropy = shannon_entropy

        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()

    def inst_eval(
            self,
            A,
            h,
            classifier,
            slide_id=None,
            only_top=False,
            use_tile_labels=False,
            A_logit=None,
            k_sample=None):
        """
        Instance (or tile) classification function. If use_tile_labels is
        False, samples the `self.k_sample` tiles with the highest attention
        scores and the `self.k_sample` ones with the lowest within the slide to
        classify them. When using ms-clam, `self.k_sample` / 2 normal tiles are
        sampled in both tumorous and non tumorous slides, to compensate for
        class imbalance. If use_tile_labels, selects tiles based on
        ground-truth classes.

        Args
        -------------
        A : Tensor
            The tensor containing the tile attention scores.
        h : Tensor
            The tile features.
        classifier : nn.Module
            The tile classifier.
        slide_id : str
            The corresponding slide id.
        only_top : bool
            When True, samples only `self.k_sample` // 2 normal tiles.
        use_tile_labels : Use tiles ground-truth labels instead of pseudo ones.
        A_logit : Tensor
            The tensor containing the attention logits, before softmax
            normalization.
        k_sample : int
            Number of instances to sample within the slide.
            Defaults to `self.k_sample`.

        Returns
        -------------
        instance_loss : Tensor
            The loss calculated on the tile predictions.
        all_preds : Tensor
            The tile predictions computed by the classifier.
        all_targets : Tensor
            The tile labels.
        probs : Tensor
            The tile prediction probabilities.
        """

        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        if use_tile_labels:
            with open(os.path.join(self.gt_dir, slide_id + ".pkl"), "rb") \
                    as fi:
                index_list = pickle.load(fi)
            # Selected instances are the ground truth patches
            num_instances_tum = min(self.gt_sample, len(index_list))
            top_p_ids = np.random.choice(
                index_list, num_instances_tum, replace=False
            )
            p_targets = self.create_positive_targets(num_instances_tum, device)
            top_p_ids = torch.tensor(
                top_p_ids, dtype=torch.int64, device=device
            )
            top_p = torch.index_select(h, dim=0, index=top_p_ids)

            # Normal patches are taken outside of the index_list
            all_indexes = np.arange(A.size(-1))
            tumor_indexes = np.array(index_list)
            top_n_ids = all_indexes[~np.isin(all_indexes, tumor_indexes)]
            # Twice as fewer neg. instances as pos. ones
            num_instances_normal = min(
                num_instances_tum // 2,
                len(top_n_ids)
            )
            top_n_ids = np.random.choice(
                top_n_ids, num_instances_normal, replace=False
            )
            top_n_ids = torch.tensor(
                top_n_ids, dtype=torch.int64, device=device
            )
            top_n = torch.index_select(h, dim=0, index=top_n_ids)
            n_targets = self.create_negative_targets(
                num_instances_normal, device
            )
        else:
            if k_sample is None:
                k_sample = min(self.k_sample, torch.numel(A) // 2)
            else:
                k_sample = min(k_sample, torch.numel(A))

            top_p_ids = torch.topk(A, k_sample)[1][-1]
            top_p = torch.index_select(h, dim=0, index=top_p_ids)
            if only_top:
                top_n_ids = torch.topk(-A, k_sample, dim=1)[1][-1]
            else:
                top_n_ids = torch.topk(
                    -A, k_sample // (1 + int(self.ms_clam)), dim=1)[1][-1]
            top_n = torch.index_select(h, dim=0, index=top_n_ids)
            p_targets = self.create_positive_targets(len(top_p), device)
            n_targets = self.create_negative_targets(len(top_n), device)
        if only_top:
            all_targets = n_targets
            all_instances = top_p
        else:
            all_targets = torch.cat([p_targets, n_targets], dim=0)
            all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        probs = nn.Softmax(dim=1)(logits)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets, probs

    def attention_loss(self, A, slide_id):
        """
        Calculates entropy-based loss on the attention weights to force
        higher weights on tumorous patches.
        """
        device = A.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        try:
            with open(os.path.join(self.gt_dir, slide_id + ".pkl"), "rb") \
                    as fi:
                index_list = pickle.load(fi)

            tumor_indexes = np.array(index_list)
            all_indexes = np.arange(A.size(-1))
            idx_normal = all_indexes[~np.isin(all_indexes, tumor_indexes)]
            idx_normal = torch.tensor(
                idx_normal, dtype=torch.int64, device=device
            )
            idx_tumor = torch.tensor(
                index_list, dtype=torch.int64, device=device
            )
            A_norm = torch.index_select(A, dim=1, index=idx_normal)
            A_tum = torch.index_select(A, dim=1, index=idx_tumor)
            if A_tum.size(1) != 1:
                att_loss = (
                    torch.sum(A_norm)
                    - self.f_entropy(A_tum)
                    - torch.sum(A_tum)
                )
            else:
                # in case of single pos. inst.
                att_loss = torch.sum(A_norm) - A_tum.squeeze()
        except FileNotFoundError:
            att_loss = None
        return att_loss

    def forward(
            self,
            h,
            label=None,
            instance_eval=False,
            return_features=False,
            attention_only=False,
            use_tile_labels=False,
            slide_id=None,
            k_sample=None):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if label == 0:  # att_loss = - H(A)
            att_loss = -self.f_entropy(A)
        else:
            if use_tile_labels:
                att_loss = self.attention_loss(A, slide_id)
            else:
                att_loss = None

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            all_probs = []
            inst_labels = F.one_hot(
                label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(self.n_classes):
                inst_label = inst_labels[i].item()
                if self.ms_clam:
                    classifier = self.instance_classifiers
                else:
                    classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    only_top = label == 0 and self.ms_clam
                    instance_loss, preds, targets, probs = self.inst_eval(
                        A, h, classifier, slide_id, only_top=only_top,
                        use_tile_labels=use_tile_labels, A_logit=A_raw,
                        k_sample=k_sample
                    )
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.detach().cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    continue
                total_inst_loss += instance_loss

        M = torch.mm(A, h)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds),
                'inst_probs': np.array(all_probs),
                'att_loss': att_loss}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
