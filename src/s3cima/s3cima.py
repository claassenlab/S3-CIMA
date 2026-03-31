"""Main function to run S3-CIMA. See README for usage instructions"""

# Imports

import json
from typing import Union

import numpy as np
import pandas as pd


from s3cima.utils.datasets import CIMADataset
from s3cima.utils.model import fit, set_seed

# Function ---------------------------------------------------------------

def run_s3cima(
    anchor: str,
    image_id, sample_id, intensity, genes, cell_type, x, y, cell_id, label, label_map,
    K: int = 50,
    ncell : int = 30,
    random_ctrl: bool = True,
    task: str = "classification",
    nruns: int = 5, 
    dendrogram_cutoff: float = 0.4,
    n_val_folds: int = 3,              
    maxpool_percentages : list = [0.01, 1, 5, 20, 100],
    nfilter_selection: list = [3, 4, 5, 6, 7, 8, 9, 10],
    batch_size = 256,
    lr: float = 0.1,
    epochs: int = 20, 
    num_workers: int = 0,
    early_stopping = True,
    patience: int = 3,
    seed: int = 420,
    save_path: str = ".",
    ):  
    """Runs an S3-CIMA analysis TODO

    Args
    ----

    Returns
    -------
    """
    # Set seed
    set_seed(seed)

    # Create dataset from processed csv
    dataset = CIMADataset(
        anchor = anchor,
        image = image_id,
        pat = sample_id,
        intensity = intensity,
        ct = cell_type,
        x = x,
        y = y,
        cellid = cell_id,
        K  = K,
        ncell = ncell,
        label = label,
        random_ctrl = random_ctrl,
        task = task,
    )

    # Fit the CIMA model from this dataset
    preds, loss, val_loss, ba, val_ba, test_samples = fit(dataset,
        anchor = anchor,
        K = K,
        ncell = ncell,
        genes = genes,
        n_val_folds=n_val_folds,
        nruns=nruns,
        background=random_ctrl,
        batch_size = batch_size,
        lr = lr,
        epochs=epochs,
        dendrogram_cutoff = dendrogram_cutoff,
        maxpool_percentages = maxpool_percentages,
        nfilter_selection = nfilter_selection,
        early_stopping = early_stopping,
        patience = patience,
        num_workers = num_workers,
        save_loc = save_path,
        seed=seed)
    
    # Plot TODO
