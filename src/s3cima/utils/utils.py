"""Generic utils for S3-CIMA"""

# Imports

import os
import random
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, cdist

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image



# Functions ---------------------------------------------------------------

def set_seed(seed = 42):
    """Fix the seeds for reproductible runs during training"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def check_cima_input_csv(
    meta: pd.DataFrame,
    markers: pd.DataFrame,
    x_col: str,
    y_col: str,
    cell_id_col: str,
    cell_type_col: str,
    sample_id_col: str,
    image_id_col: str,
    condition_col: str,
    filter_cell_types: list = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate and clean input DataFrames for the S3-CIMA pipeline.

    Returns
    -------
    meta    : pd.DataFrame — cleaned metadata
    markers : pd.DataFrame — cleaned marker intensities (rows aligned to meta)
    """

    # Length check
    if len(meta) != len(markers):
        raise ValueError(
            f"meta ({len(meta)} rows) and markers ({len(markers)} rows) must "
            "have the same number of rows — each row must correspond to the "
            "same cell in both DataFrames."
        )

    # Col existence in meta
    required_cols = [x_col, y_col, cell_id_col, cell_type_col,
                     sample_id_col, image_id_col, condition_col]
    missing_cols = [c for c in required_cols if c not in meta.columns]
    if missing_cols:
        raise ValueError(
            f"The following required columns are missing from the metadata: "
            f"{missing_cols}\n"
            f"Available columns: {list(meta.columns)}"
        )

    # NaNs
    meta_nan_mask    = meta.isna().any(axis=1)
    markers_nan_mask = markers.isna().any(axis=1)
    combined_nan     = meta_nan_mask | markers_nan_mask
    n_nan            = combined_nan.sum()

    if n_nan > 0:
        print(f"[WARNING] {n_nan} rows contain NaN values and will be removed.")
        print(f"  NaNs in meta:    {meta_nan_mask.sum()} rows")
        print(f"  NaNs in markers: {markers_nan_mask.sum()} rows")
        meta    = meta[~combined_nan].reset_index(drop=True)
        markers = markers[~combined_nan].reset_index(drop=True)
        print(f"  Remaining rows after removal: {len(meta)}")

    # Log norm ?
    marker_vals = markers.values.astype(float)

    has_negatives  = (marker_vals < 0).any()
    global_max     = marker_vals.max()
    global_min     = marker_vals.min()
    global_mean    = marker_vals.mean()

    if has_negatives:
        warnings.warn(
            f"[WARNING] Marker matrix contains negative values "
            f"(min={global_min:.4f}). Log-normalised expression values should "
            "be non-negative. Check whether your data has been correctly "
            "normalised.",
            UserWarning,
            stacklevel=2,
        )
    if global_max > 30:
        warnings.warn(
            f"[WARNING] Marker matrix contains very large values "
            f"(max={global_max:.4f}, mean={global_mean:.4f}). This may indicate "
            "the data is not log-normalised (expected range typically 0–10 for "
            "log1p-normalised counts). Consider applying log1p normalisation.",
            UserWarning,
            stacklevel=2,
        )
    if not has_negatives and global_max <= 30:
        print(f"[INFO] Marker value range: [{global_min:.4f}, {global_max:.4f}] "
              f"— consistent with log-normalised data.")

    # Filter cell types
    if filter_cell_types is not None:
        if not isinstance(filter_cell_types, Iterable) or isinstance(filter_cell_types, str):
            raise TypeError(
                f"filter_cell_types must be a list or iterable of cell type strings, "
                f"got {type(filter_cell_types)}."
            )

        available_types = meta[cell_type_col].unique().tolist()
        missing_types   = [ct for ct in filter_cell_types
                           if ct not in available_types]
        if missing_types:
            raise ValueError(
                f"The following cell types in filter_cell_types were not found "
                f"in the dataset: {missing_types}\n"
                f"Available cell types:\n  {sorted(available_types)}"
            )
        print(f"[INFO] filter_cell_types validated — "
              f"{len(filter_cell_types)} type(s) will be removed.")

    print("[INFO] All checks passed.")
    return meta, markers


def process_csv(
    markers_path: str,
    meta_path: str,
    x_col: str,
    y_col: str,
    cell_id_col: str,
    cell_type_col: str,
    sample_id_col: str,
    image_id_col: str,
    condition_col: str,
    filter_cell_types: list = None,
) -> tuple:
    """
    Validate, clean, and extract arrays from input CSVs for the S3-CIMA pipeline.

    Args
    ----
    markers_path      : path to CSV of marker intensities (cells × genes)
    meta_path         : path to CSV of cell metadata
    x_col             : column name for x coordinate
    y_col             : column name for y coordinate
    cell_id_col       : column name for cell ID
    cell_type_col     : column name for cell type label
    sample_id_col     : column name for patient/sample ID
    image_id_col      : column name for image/FOV ID
    condition_col     : column name for condition/phenotype label
    filter_cell_types : optional list of cell types to retain; None keeps all

    Returns
    -------
    intensity   : np.ndarray, shape (N, M) — marker expression matrix
    genes       : list of str             — gene/marker names (length M)
    x           : np.ndarray, shape (N,)  — x coordinates
    y           : np.ndarray, shape (N,)  — y coordinates
    cell_id     : np.ndarray, shape (N,)  — cell IDs
    cell_type   : np.ndarray, shape (N,)  — cell type labels
    sample_id   : np.ndarray, shape (N,)  — sample/patient IDs
    image_id    : np.ndarray, shape (N,)  — image IDs
    condition   : np.ndarray, shape (N,)  — condition labels
    """
    import os

    # Load
    print(f"[INFO] Loading metadata from   : {meta_path}")
    print(f"[INFO] Loading markers from    : {markers_path}")
    meta    = pd.read_csv(meta_path,    sep=',')
    markers = pd.read_csv(markers_path, sep=',')
    print(f"[INFO] Loaded {len(meta)} cells × {markers.shape[1]} markers.")

    # Input check
    meta, markers = check_cima_input_csv(
        meta = meta,
        markers = markers,
        x_col = x_col,
        y_col = y_col,
        cell_id_col = cell_id_col,
        cell_type_col = cell_type_col,
        sample_id_col = sample_id_col,
        image_id_col = image_id_col,
        condition_col = condition_col,
        filter_cell_types = filter_cell_types,
    )
 
    # Filter if specified
    if filter_cell_types is not None:
        n_before = len(meta)
        mask = ~meta[cell_type_col].isin(filter_cell_types)
        meta = meta[mask].reset_index(drop=True)
        markers = markers[mask].reset_index(drop=True)
        print(f"[INFO] Cell type filter applied: "
              f"{n_before} → {len(meta)} cells retained "
              f"({n_before - len(meta)} removed).")

    # Extract the arrays
    genes = list(markers.columns)
    intensity = markers.values.astype(np.float32)

    x = meta[x_col].values.astype(np.float32)
    y = meta[y_col].values.astype(np.float32)
    cell_id = meta[cell_id_col].values
    cell_type = meta[cell_type_col].values
    sample_id = meta[sample_id_col].values
    image_id = meta[image_id_col].values
    condition = meta[condition_col].values

    print(f"[INFO] Final output: {intensity.shape[0]} cells × "
          f"{intensity.shape[1]} markers.")

    return intensity, genes, x, y, cell_id, cell_type, sample_id, image_id, condition

