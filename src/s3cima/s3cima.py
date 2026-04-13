"""Main function to run S3-CIMA. See README for usage instructions"""

# Imports

import json
from typing import Union

import numpy as np
import pandas as pd


from s3cima.utils.datasets import CIMADataset
from s3cima.utils.model import fit, set_seed
import s3cima.utils.plot as s3cima_plot 

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
    filter_threshold: float = 0.9,
    dropout: bool = True,
    dropout_p: float = 0.5,
    bg_sets: int = 500,
    batch_size = 256,
    lr: float = 0.1,
    l2: float = 1e-4,
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
        bg_sets=bg_sets,
        seed = seed,
    )

    # Fit the CIMA model from this dataset
    results, loss, val_loss, ba, val_ba, test_samples, train_folds, res_save_path = fit(dataset,
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
        dropout = dropout,
        dropout_p = dropout_p,
        l2 = l2,
        early_stopping = early_stopping,
        patience = patience,
        num_workers = num_workers,
        save_loc = save_path,
        seed=seed)
    
    # Calculate cell filter response - results is the final metadata dict
    fig_save_path = s3cima_plot.plot_filter_weights(results,
                                    show=False, 
                                    save_path=res_save_path)   
    res_test = s3cima_plot.get_high_response_cells_test(results, 
                                            test_samples, 
                                            genes, 
                                            filter_threshold=filter_threshold)
    res_train = s3cima_plot.get_high_response_cells_train(results, 
                                              train_folds, 
                                              genes, 
                                              filter_threshold=filter_threshold)
    
    # Calculate enrichment and plots
    test_fig_save_path = s3cima_plot.save_stats(res_test,
                                    x, y, cell_id, cell_type, sample_id, label, label_map,
                                    save_path = fig_save_path,
                                    test = True)
    train_fig_save_path = s3cima_plot.save_stats(res_train,
                                    x, y, cell_id, cell_type, sample_id, label, label_map,
                                    save_path = fig_save_path,
                                    test = False)
    

    # s3cima_plot.enrichment_summary(test_fig_save_path,
    #                                test=True)
    # s3cima_plot.enrichment_summary(train_fig_save_path,
    #                                test=False)

