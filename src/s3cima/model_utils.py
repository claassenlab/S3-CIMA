"""Utils specific to the S3-CIMA Model classes.
"""

# Imports

import os
import argparse
import pickle
import random
import time
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, cdist

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset



# Functions ---------------------------------------------------------------

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"[INFO]: Early stopping counter - {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('[INFO]: Early stopping')
                self.early_stop = True


def select_top_pool(x: torch.Tensor, k: int, selection_type: str = "mean") -> torch.Tensor:
    """
    Pools the top-k activated cells per filter.

    Parameters
    ----------
    x              : (B, nfilter, ncell) — output of Conv1d after ReLU
    k              : number of top cells to select per filter
    selection_type : 'mean' or 'max'

    Returns
    -------
    (B, nfilter) pooled representation
    """
    # topk over the cell dimension (dim=2)
    topk_vals, _ = torch.topk(x, k=k, dim=2)   # (B, nfilter, k)
    if selection_type == "mean":
        return topk_vals.mean(dim=2)             # (B, nfilter)
    elif selection_type == "max":
        return topk_vals[:, :, 0]               # (B, nfilter)
    else:
        raise ValueError(f"selection_type must be 'mean' or 'max', got '{selection_type}'")