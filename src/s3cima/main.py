"""Main function to run S3-CIMA. See README for usage instructions"""

# Imports

import json
from typing import Union

import numpy as np
import pandas as pd




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