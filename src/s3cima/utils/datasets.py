"""Utils specific to the S3-CIMA Dataset classes.
"""

# Imports
import os
import argparse
import pickle
import random
import time
import json
from typing import Union
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from scipy.spatial import cKDTree


# Functions ---------------------------------------------------------------

# TODO add a calculation of an outlier score - this will help once we
# implement the outlier sampling.

class CIMADataset(Dataset):
    """
    One sample is one anchor cell + its K nearest neighbours' intensities.

    Parameters
    ----------
    anchor      : str   — anchor cell type label
    image       : (N,) array of image IDs
    pat         : (N,) array of patient IDs
    intensity   : (N, M) float array of marker intensities
    ct          : (N,) array of cell type strings
    x, y        : (N,) float arrays of spatial coordinates
    cellid      : (N,) array of cell IDs
    K           : int  — number of nearest neighbours
    ncell       : int - number of K neighbours to subsample (CellCNN input)
    label       : (N,) array of labels
    random_ctrl : bool — if True, expose random-anchor controls as well
    task        : str  — "classification" or "regression", inferred from label type
    seed        : int  — random seed
    """

    def __init__(
        self,
        anchor: str,
        image, pat, intensity, ct, x, y, cellid, label,
        K: int,
        ncell : int,
        random_ctrl: bool = False,
        task: str = "classification",
        seed: int = 12345,
    ):
        # Task input check
        assert task in ("classification", "regression"), \
            "task must be 'classification' or 'regression'"
        
        super().__init__()
        self.K = K
        self.ncell = ncell
        self.label = label
        self.random_ctrl = random_ctrl
        self.rng = np.random.default_rng(seed)

        # resolve label type
        self.task = task
        print(self.task)

        # --- collect anchor indices per image, build KD-trees once --------
        self.samples = []

        # Set control label
        if random_ctrl:
            ctrl_label = len(set(self.label))
            encoded_ctrl = self._encode_label(ctrl_label)
            print(encoded_ctrl)


        for img_id in np.unique(image):
            mask = image == img_id
            ix   = np.where(mask)[0]

            img_x      = x[ix]
            img_y      = y[ix]
            img_int    = intensity[ix]
            img_ct     = ct[ix]
            img_cellid = cellid[ix]
            img_pat    = np.unique(pat[ix])[0]
            img_labels = label[ix]

            coords = np.stack([img_x, img_y], axis=1)
            tree   = cKDTree(coords)

            anchor_idx = np.where(img_ct == anchor)[0]
            all_idx    = np.arange(len(img_ct))
            non_anchor = np.delete(all_idx, anchor_idx)
            ctrl_idx   = self.rng.choice(
                non_anchor, size=len(anchor_idx), replace=False
            )
            # Selecting the actual anchor cells
            for idx in anchor_idx:
                knn_int, knn_ids = self._query(idx, tree, img_int, img_cellid)
                self.samples.append({
                    "intensity": torch.tensor(knn_int, dtype=torch.float32),
                    "cellids":   knn_ids,
                    "pat":       img_pat,
                    "label":     self._encode_label(img_labels[idx]),
                    "task":      self.task,
                    "is_ctrl":   False,
                })
            # Background control cells (aka non-anchor cells)
            if random_ctrl:
                for idx in ctrl_idx:
                    knn_int, knn_ids = self._query(idx, tree, img_int, img_cellid)
                    self.samples.append({
                        "intensity": torch.tensor(knn_int, dtype=torch.float32),
                        "cellids":   knn_ids,
                        "pat":       img_pat,
                        "label":     encoded_ctrl,
                        "task":      self.task,
                        "is_ctrl":   True,
                    })
                


    # label query and encoding
    def _encode_label(self, val) -> torch.Tensor:
        if self.task == "classification":
            # expects val to already be an integer class index,
            # or a string pre-mapped to an int by the user upstream
            return torch.tensor(int(val), dtype=torch.long)
        else:
            return torch.tensor(float(val), dtype=torch.float32)
        

    def _query(self, idx, tree, img_int, img_cellid):
        _, nn_ix = tree.query(tree.data[idx], k=self.K + 1)
        nn_ix = nn_ix[1:]

        assert self.ncell <= self.K, (
        f"ncell ({self.ncell}) must be <= K ({self.K}): "
        f"cannot sample more cells than the number of nearest neighbours retrieved."
    )
        nn_ix = self.rng.choice(nn_ix, size=self.ncell, replace=False)

        return img_int[nn_ix], img_cellid[nn_ix].tolist()
    

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        return self.samples[idx]


class NormalizedCIMASubset(Dataset):
    def __init__(self, subset: Subset, mean: torch.Tensor, std: torch.Tensor):
        """
        Wraps a Subset of CIMADataset to apply Z-score normalization.
        """
        self.subset = subset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Retrieve the dictionary from the underlying CIMADataset
        sample = self.subset[idx]
        x = sample["intensity"].clone()
        x = (x - self.mean) / self.std
        
        # Return the updated dictionary
        return {
            "intensity": x,
            "label":     sample["label"],
            "pat":       sample["pat"],
            "task":      sample["task"],
            "is_ctrl":   sample["is_ctrl"],
            "cellids":   sample["cellids"]
        }


def spatial_collate_fn(batch):
    return {
        "intensity": torch.stack([s["intensity"] for s in batch]),  # (B, K, M)
        "cellids":   [s["cellids"] for s in batch],
        "pat":       [s["pat"]     for s in batch],
        "label":     torch.stack([s["label"]     for s in batch]),  # (B,) long or float32
        "is_ctrl":   [s["is_ctrl"] for s in batch],
    }


def get_norm_stats(dataset: CIMADataset, train_idx: list):
    """
    Calculates mean and std of intensities across all training samples.
    """
    # Collect intensities from the specified training indices
    train_intensities = torch.stack([dataset.samples[i]["intensity"] for i in train_idx])
    
    # Calculate marker-wise stats
    flat_intensities = train_intensities.view(-1, train_intensities.shape[-1])
    
    mean = flat_intensities.mean(dim=0)
    std = flat_intensities.std(dim=0)
    
    # safety
    std[std == 0] = 1.0
    
    return mean, std


def make_patient_stratified_splits(
    dataset: CIMADataset,
    n_val_folds: int = 5,
    seed: int = 12345,
):
    """
    Build patient-stratified train/val/test splits with no sample-level leakage,
    stratified by patient-level label.

    Strategy
    --------
    - 1 patient per label class is the held-out test set
    - Remaining patients split into n_val_folds folds:
        each fold uses 1 patient per label class as validation, rest as training

    Parameters
    ----------
    dataset     : CIMADataset
    n_val_folds : int - number of train/val splits
    seed        : int

    Returns
    -------
    test_subset : Subset
    folds       : list of (train_Subset, val_Subset) tuples
    """
    rng = np.random.default_rng(seed)

    # derive patient level + labels
    # exclude control samples (is_ctrl=True) to get the real label per patient
    pat_to_label = {}
    for s in dataset.samples:
        if not s["is_ctrl"]:
            p = s["pat"]
            if p not in pat_to_label:
                pat_to_label[p] = s["label"].item()   # scalar from tensor

    # group patients by their label
    label_to_pats = defaultdict(list)
    for p, lbl in pat_to_label.items():
        label_to_pats[lbl].append(p)

    for lbl in label_to_pats:
        rng.shuffle(label_to_pats[lbl])
        assert len(label_to_pats[lbl]) >= n_val_folds + 1, (
            f"Label '{lbl}' has only {len(label_to_pats[lbl])} patients — "
            f"need at least n_val_folds+1={n_val_folds + 1}"
        )

    # test hold-out
    test_pats = set()
    remaining_pats = {}                        # label --> list of remaining patients
    for lbl, pats in label_to_pats.items():
        test_pats.add(pats[0])
        remaining_pats[lbl] = pats[1:]

    sample_pats = np.array([s["pat"] for s in dataset.samples])
    test_idx    = [i for i, p in enumerate(sample_pats) if p in test_pats]
    m_test, s_test = get_norm_stats(dataset, test_idx)
    test_subset = NormalizedCIMASubset(Subset(dataset, test_idx), m_test, s_test)
    print(f"Test : {len(test_idx)} samples ({len(test_pats)} patients, "
          f"1 per label: {list(label_to_pats.keys())})")

    # build the splits
    folds = []

    # Check input number of folds is correct
    min_pats_per_label = min(len(pats) for pats in remaining_pats.values())
    min_label = min(remaining_pats, key=lambda l: len(remaining_pats[l]))
    assert n_val_folds <= min_pats_per_label, (
        f"n_val_folds={n_val_folds} exceeds the number of remaining patients for label "
        f"'{min_label}' ({min_pats_per_label} patients after test holdout). "
        f"Reduce n_val_folds to at most {min_pats_per_label}."
    )

    # TODO - improve the fold formation.
    for fold in range(n_val_folds):
        val_pats = set()
        train_pats = set()
        for lbl, pats in remaining_pats.items():
            val_pat = pats[fold % len(pats)]
            val_pats.add(val_pat)
            train_pats.update(p for p in pats if p != val_pat)

        val_idx = [i for i, p in enumerate(sample_pats) if p in val_pats]
        train_idx = [i for i, p in enumerate(sample_pats) if p in train_pats]

        m, s = get_norm_stats(dataset, train_idx + val_idx)

        train_subset = NormalizedCIMASubset(Subset(dataset, train_idx), m, s)
        val_subset = NormalizedCIMASubset(Subset(dataset, val_idx), m, s)
        folds.append((train_subset, val_subset))
        print(
            f"Fold {fold}: "
            f"train={len(train_idx)} samples ({len(train_pats)} patients) | "
            f"val={len(val_idx)} samples ({len(val_pats)} patients)"
        )

    return test_subset, folds