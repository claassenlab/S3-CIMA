"""Main function to run S3-CIMA. See README for usage instructions"""

# Imports

import json

import numpy as np
import pandas as pd




# Function ---------------------------------------------------------------

def process_csv(
    markers: pd.DataFrame,
    meta: pd.DataFrame,
    x_col: str,
    y_col: str,
    cell_id_col: str,
    cell_type_col: str,
    sample_id_col: str,
    image_id_col: str,
    condition_col: str,
    filter_cell_types: list = None,
    save_path: str = ".",
    ):  
    """Runs an S3-CIMA analysis"""