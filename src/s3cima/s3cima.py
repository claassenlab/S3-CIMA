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
    maxpool_percentages : list = (0.01, 1, 5, 20, 100),
    nfilter_selection: list = (3, 4, 5, 6, 7, 8, 9, 10),
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
    """Run a full S3-CIMA analysis.

    Builds multi-cell input sets from the spatial neighborhoods of an anchor
    cell type, trains the S3-CIMA weakly supervised convolutional neural
    network to predict the sample-level outcome, and identifies the cell
    populations (filters) whose neighborhood composition is associated with
    that outcome. Filter weights, high-response cells, enrichment statistics
    and summary plots are written to disk for both the training and test sets.

    Parameters
    ----------
    anchor : str
        Cell type used as the anchor. Neighborhoods are built around cells
        of this type.
    image_id : array-like of shape (n_cells,)
        Image (region of interest) identifier for each cell.
    sample_id : array-like of shape (n_cells,)
        Sample or patient identifier for each cell. Outcome labels are
        defined at this level.
    intensity : array-like of shape (n_cells, n_markers)
        Marker expression intensities for each cell.
    genes : list of str
        Marker names, in the same order as the columns of ``intensity``.
    cell_type : array-like of shape (n_cells,)
        Cell type annotation for each cell.
    x : array-like of shape (n_cells,)
        Spatial x coordinate of each cell.
    y : array-like of shape (n_cells,)
        Spatial y coordinate of each cell.
    cell_id : array-like of shape (n_cells,)
        Unique identifier for each cell.
    label : array-like of shape (n_cells,)
        Outcome label (e.g. disease condition) of the sample each cell
        belongs to.
    label_map : dict
        Mapping from label values to human-readable class names, used in
        plots and enrichment outputs.
    K : int, default=50
        Number of nearest neighbors defining the spatial neighborhood around
        each anchor cell.
    ncell : int, default=30
        Number of cells in each multi-cell input set passed to the network.
    random_ctrl : bool, default=True
        Whether to generate random background (control) cell sets to compare
        against the anchor neighborhoods.
    task : str, default="classification"
        Learning task, e.g. ``"classification"`` for categorical outcomes or
        regression for contiuous outcomes ("regression" mode not yet supported).
    nruns : int, default=5
        Number of independent training runs with different random
        initializations.
    dendrogram_cutoff : float, default=0.4
        Cutoff applied to the hierarchical clustering of learned filters
        across runs, used to group similar filters into consensus filters.
    n_val_folds : int, default=3
        Number of cross-validation folds used for model selection.
    maxpool_percentages : list of float, default=[0.01, 1, 5, 20, 100]
        Candidate percentages of top-responding cells pooled per filter,
        searched during hyperparameter selection.
    nfilter_selection : list of int, default=[3, 4, 5, 6, 7, 8, 9, 10]
        Candidate numbers of convolutional filters, searched during
        hyperparameter selection.
    filter_threshold : float, default=0.9
        Threshold on the filter response above which a cell is considered
        a high-response cell.
    dropout : bool, default=True
        Whether to apply dropout during training.
    dropout_p : float, default=0.5
        Dropout probability, used if ``dropout`` is True.
    bg_sets : int, default=500
        Number of random background cell sets generated when
        ``random_ctrl`` is True.
    batch_size : int, default=256
        Training batch size.
    lr : float, default=0.1
        Learning rate.
    l2 : float, default=1e-4
        L2 regularization strength (weight decay).
    epochs : int, default=20
        Maximum number of training epochs.
    num_workers : int, default=0
        Number of worker processes for data loading.
    early_stopping : bool, default=True
        Whether to stop training when the validation loss stops improving.
    patience : int, default=3
        Number of epochs without improvement before early stopping is
        triggered.
    seed : int, default=420
        Random seed for reproducibility.
    save_path : str, default="."
        Directory where the trained model results, figures and enrichment
        statistics are saved.

    Returns
    -------
    None
        All outputs (model results, filter weight plots, high-response cell
        statistics and enrichment summaries for the train and test sets)
        are written to ``save_path``.

    Examples
    --------
    >>> run_s3cima(
                "BG",
                image_id, 
                sample_id, 
                intensity, 
                genes, 
                cell_type, 
                x, 
                y, 
                cell_id, 
                labels,
                label_map,
                K=50,
                ncell = 30,
                random_ctrl=False,
                n_val_folds=3,
                bg_sets=1000,    # Number of random background sets to draw from each sample/patient
                num_workers=24,
                batch_size=128,
                lr = 0.001,
                nruns=10,
                save_path = save_path,
                seed=43
                )
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
    

    s3cima_plot.enrichment_summary(test_fig_save_path,
                                   cell_type, sample_id,
                                   test=True)
    s3cima_plot.enrichment_summary(train_fig_save_path,
                                   cell_type, sample_id,
                                   test=False)

