"""Main function to run S3-CIMA. See README for usage instructions"""

# Imports

import json
from typing import Union

import numpy as np
import pandas as pd


from utils.dataset_utils import CIMADataset, NormalizedCIMASubset
from utils.dataset_utils import spatial_collate_fn, get_norm_stats
from utils.dataset_utils import make_patient_stratified_splits

from utils.model_utils import EarlyStopping, set_seed
from utils.model_utils import get_discriminative_filters, select_top_pool
from utils.model_utils import train, validate, test, fit

# Function ---------------------------------------------------------------

def run_s3cima(
    anchor: str,
    image, pat, intensity, genes, ct, x, y, cellid, label,
    K: int = 50,
    ncell : int = 30,
    random_ctrl: bool = True,
    task: str = "classification",
    nruns: int = 5, 
    dendrogram_cutoff: float = 0.4,
    n_val_folds: int = 3,              
    maxpool_percentages : list = [0.01, 1, 5, 20, 100],
    nfilter_selection: list = [3, 4, 5, 6, 7, 8, 9, 10],
    background: bool = False,
    batch_size = 256,
    lr: float = 0.01,
    epochs: int = 20, 
    num_workers: int = 0,
    early_stopping = True,
    patience: int = 3,
    seed: int = 420,
    save_path: str = ".",
    ):  
    """Runs an S3-CIMA analysis

    Args
    ----

    Returns
    -------
    """
    # Set seed
    set_seed(seed)

    # Create dataset
    dataset = CIMADataset()
