"""Utils specific to the S3-CIMA Model classes.
"""

# Imports

import os
import argparse
import pickle
import random
import time
import json
from datetime import datetime

from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, cdist

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from sklearn.metrics import f1_score, balanced_accuracy_score

from s3cima.utils.datasets import CIMADataset
from s3cima.utils.datasets import spatial_collate_fn, get_norm_stats
from s3cima.utils.datasets import make_patient_stratified_splits

from s3cima.model import CellCNN

# General functions ---------------------------------------------------------------

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


# Filter calculation functions ----------------------------------------------------

def _relu(x):
    return np.maximum(0, x)


def _single_filter_output(conv_w, bias, out_w, valid_samples, mp):
    """
    conv_w  : (nmark,)   — filter weights
    bias    : float      — filter bias
    out_w   : (n_classes,) — output weights for this filter
    mp      : float      — max-pooling percentage (0-100)
    """
    y_pred = np.zeros(len(valid_samples))
    for i, x in enumerate(valid_samples):
        # x shape: (ncell, nmark)
        g = _relu(x @ conv_w + bias)               # (ncell,)
        ntop = max(1, int(mp / 100. * x.shape[0]))
        gpool = np.mean(np.sort(g)[-ntop:])            # scalar
        y_pred[i] = gpool
    dominant_class = int(np.argmax(out_w))
    return y_pred, dominant_class


def _compute_filter_diff(consensus_conv, consensus_out, consensus_biases,
                          valid_samples, valid_phenotypes, mp):
    """
    Returns filter_diff array of shape (n_consensus_filters,).
    """
    y_true      = np.array(valid_phenotypes)
    filter_diff = np.zeros(len(consensus_conv))

    for i, (conv_w, out_w, bias) in enumerate(
            zip(consensus_conv, consensus_out, consensus_biases)):
        y_pred, filter_class = _single_filter_output(
            conv_w, bias, out_w, valid_samples, mp
        )
        in_class  = y_true == filter_class
        out_class = ~in_class
        if in_class.sum() == 0 or out_class.sum() == 0:
            filter_diff[i] = 0.0   # degenerate case
        else:
            filter_diff[i] = np.mean(y_pred[in_class]) - np.mean(y_pred[out_class])

    return filter_diff
    

def get_discriminative_filters(
    save_model: str,
    nruns: int,
    validation_subset: Subset,
    loss_fn,
    n_classes: int,
    nmarkers: int,
    lr: float,
    batch_size: int,
    num_workers: int,
    log_file: str,
    device: str,
    accur_thres: float = 0.85,
    dendrogram_cutoff: float = 0.4,
    filter_diff_threshold: float = 0.0,
    result_folder: str = None,
    genes: list = None,
    has_background: bool = True,
    classification: bool = True,
):
    """
    After all runs are complete:
      1. Load each saved run's best model
      2. Evaluate validation accuracy
      3. Collect filter weight vectors from runs passing accur_thres
         (minimum 3 models always kept regardless of threshold)
      4. Concatenate all passing filter weights into one matrix
      5. Cluster via hierarchical clustering (cosine distance)
      6. Select one consensus filter per cluster (medoid) and calculate filter diff

    Filter weights shape per run: (nfilter, nmark)
    Only filters connected to non-control classes are retained
    (i.e. output weight to ctrl class [n_classes-1] is NOT the dominant connection).
    """
    # Block if regression for the moment
    if not classification:
        raise Exception("Regression not implemented yet")


    # Validate genes if provided
    if genes is not None:
        genes = list(genes)
        if len(genes) != nmarkers:
            raise ValueError(
                f"Length of genes ({len(genes)}) must equal nmarkers ({nmarkers})."
            )
        
    # Get the sample information from the validation dataset

    t_loader = DataLoader(
        validation_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    accuracies = {}    
    f1_scores = {}       
    all_weights = {}      # run -> dict with 'conv' (nfilter, nmark) and 'out' (nfilter, n_classes)

    # ----------------------------------------------------------------
    # 1. Load each run, evaluate test accuracy, extract weights
    # ----------------------------------------------------------------
    for run in range(nruns):
        ckpt_path = f"{save_model}/best_model_run{run}.pth"
        if not os.path.exists(ckpt_path):
            print(f"[WARN] No checkpoint for run {run}, skipping.")
            continue

        params = torch.load(ckpt_path, map_location=device, weights_only=False)

        model = CellCNN(
            nmark        = params["nmarkers"],
            nfilter      = params["nfilter"],
            k            = params["k"],
            n_classes    = params["n_classes"],
            dropout      = params["dropout"],
            dropout_p    = params["dropout_p"],
            regression   = params["regression"],
            selection_type = params["selection_type"],
        )
        model.load_state_dict(params["model_state_dict"])
        model.to(device)

        test_loss, preds, trues = test(
            model, t_loader, loss_fn, 0, result_folder, log_file, device
        )

        ba = balanced_accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average = "macro")
        accuracies[run] = ba
        f1_scores[run] = f1
        print(f"[Run {run}] Validation balanced accuracy: {ba:.4f}")
        print(f"[Run {run}] Validation F1 Score: {f1:.4f}")

        # Extract conv filter weights: shape (nfilter, nmark)
        # Extract output layer weights: shape (nfilter, n_classes)
        # Careful of cellcnn names
        conv_w = model.conv.weight.detach().cpu().numpy()   # (nfilter, nmark, 1) for Conv1d
        conv_w = conv_w.squeeze(-1)                          # (nfilter, nmark)
        out_w  = model.fc.weight.detach().cpu().numpy()  # (n_classes, nfilter)
        out_w  = out_w.T                                     # (nfilter, n_classes)
        bias_w = model.conv.bias.detach().cpu().numpy()

        all_weights[run] = {"conv": conv_w, "out": out_w, "bias": bias_w}

        with open(log_file, "a") as log:
            log.write(f"\n[Run {run}] Validation balanced accuracy: {ba:.2f}, Validation Macro F1 Score: {f1:.2f}\n")

    # ----------------------------------------------------------------
    # 2. Select runs above threshold; always keep at least 3
    # ----------------------------------------------------------------
    sorted_runs = sorted(accuracies, key=accuracies.get, reverse=True)
    passing_runs = [r for r in sorted_runs if accuracies[r] >= accur_thres]

    default_selection = len(sorted_runs)
    if len(passing_runs) < 1:
        passing_runs = sorted_runs[:default_selection]
        print(f"[INFO] No run passed the accur_thres={accur_thres}. "
              f"Keeping top 3 by accuracy.")
        
        with open(log_file, "a") as log:
            log.write(f"[INFO] No run passed the accur_thres={accur_thres}.\n")
            log.write(f"[INFO] Keeping top {default_selection} runs by accuracy.\n")

    print(f"[INFO] Runs used for filter extraction: {passing_runs}")

    # ----------------------------------------------------------------
    # 3. Pool filter weight vectors from passing runs
    #    Keep only filters whose dominant output connection is NOT
    #    the control class (last class index = n_classes - 1)
    # ----------------------------------------------------------------
    ctrl_class = n_classes - 1 if has_background else None

    pooled_conv = []
    pooled_out = []
    pooled_bias = []
    filter_run_ids = []

    for run in passing_runs:
        conv_w = all_weights[run]["conv"]   # (nfilter, nmark)
        out_w  = all_weights[run]["out"]    # (nfilter, n_classes)
        bias_w = all_weights[run]["bias"]   # (nfilter,)

        for f_idx in range(conv_w.shape[0]):
            if has_background:
                if np.argmax(out_w[f_idx]) == ctrl_class:
                    continue # discard the filters pertaining to the background
            pooled_conv.append(conv_w[f_idx])
            pooled_out.append(out_w[f_idx])
            pooled_bias.append(bias_w[f_idx])
            filter_run_ids.append(run)

    if len(pooled_conv) == 0:
        print("[WARN] No non-control filters found across passing runs.")
        return None

    pooled_conv = np.array(pooled_conv)   # (N_filters, nmark)
    pooled_out  = np.array(pooled_out)    # (N_filters, n_classes)

    print(f"[INFO] Total non-control filters pooled: {pooled_conv.shape[0]}")

    # ----------------------------------------------------------------
    # 4. Hierarchical clustering of filter weight vectors
    #    using cosine distance
    # ----------------------------------------------------------------
    if pooled_conv.shape[0] == 1:
        # Edge case: only one filter, no clustering needed
        labels_clust = np.array([0])
    else:
        dist_matrix  = pdist(pooled_conv, metric="cosine")
        linkage_mat  = linkage(dist_matrix, method="average")
        labels_clust = fcluster(linkage_mat, t=dendrogram_cutoff, criterion="distance")

    n_clusters = len(np.unique(labels_clust))
    print(f"[INFO] Number of filter clusters: {n_clusters}")

    # ----------------------------------------------------------------
    # 5. Select consensus filter per cluster: the medoid
    #    (filter with minimum average cosine distance to all others
    #     in the same cluster)
    # ----------------------------------------------------------------
    consensus_conv = []
    consensus_out  = []
    consensus_meta = []
    consensus_bias = []

    for cluster_id in np.unique(labels_clust):
        member_idx = np.where(labels_clust == cluster_id)[0]
        members    = pooled_conv[member_idx]

        if len(member_idx) == 1:
            medoid_local = 0
        else:
            dist_mat     = cdist(members, members, metric="cosine")
            medoid_local = np.argmin(dist_mat.mean(axis=1))

        medoid_global = member_idx[medoid_local]
        medoid_weights   = pooled_conv[medoid_global]    # (nmark,)

        # Rank genes by absolute filter weight (descending)
        ranked_idx    = np.argsort(np.abs(medoid_weights))[::-1]
        ranked_weights = medoid_weights[ranked_idx].tolist()
        ranked_genes   = (
            [genes[i] for i in ranked_idx] if genes is not None
            else [f"marker_{i}" for i in ranked_idx]
        )

        consensus_conv.append(medoid_weights)
        consensus_out.append(pooled_out[medoid_global])
        consensus_bias.append(pooled_bias[medoid_global])
        consensus_meta.append({
            "cluster":            int(cluster_id),
            "medoid_filter_idx":  int(medoid_global),
            "source_run":         filter_run_ids[medoid_global],
            "val_ba":             round(accuracies[filter_run_ids[medoid_global]], 4),
            "n_members":          len(member_idx),
            "top_genes":          ranked_genes[:10], 
            "top_weights":        [round(w, 6) for w in ranked_weights[:10]],
            "all_gene_weights":   dict(zip(ranked_genes, [round(w, 6) for w in ranked_weights])),
        })

    consensus_conv = np.array(consensus_conv)  # (n_clusters, nmark)
    consensus_out  = np.array(consensus_out)   # (n_clusters, n_classes)
    consensus_bias = np.array(consensus_bias)  # (n_clusters,)

    # ----------------------------------------------------------------
    # 6a. Compute filter_diff if validation data provided
    # ----------------------------------------------------------------
    filter_dir = os.path.join(result_folder, "filters")
    os.makedirs(filter_dir, exist_ok=True)
    
    filter_diff = None
    valid_samples = [validation_subset.dataset.samples[i] for i in validation_subset.indices]
    
    vs = [
        s["intensity"].numpy() if isinstance(s, dict) else np.array(s)
        for s in valid_samples
    ]

    # Default to 100% (mean pooling) as a safe fallback
    mp = 100.0
    filter_diff = _compute_filter_diff(
        consensus_conv, consensus_out, consensus_bias, vs, trues, mp
    )

    print(f"[INFO] Filter discriminability scores: {np.round(filter_diff, 4)}")

    # Flag low-discriminability filters
    flagged = [i for i, fd in enumerate(filter_diff) if fd < filter_diff_threshold]
    if flagged:
        print(f"[WARN] Filters with filter_diff < {filter_diff_threshold}: "
                f"cluster IDs {[consensus_meta[i]['cluster'] for i in flagged]}")

    # Embed into metadata
    for i, m in enumerate(consensus_meta):
        m["filter_diff"] = round(float(filter_diff[i]), 6)
        m["discriminative"] = bool(filter_diff[i] >= filter_diff_threshold)

    # Save filter_diff as CSV
    df_diff = pd.DataFrame({
        "cluster":      [m["cluster"] for m in consensus_meta],
        "filter_diff":  filter_diff,
        "discriminative": [m["discriminative"] for m in consensus_meta],
    }).sort_values("filter_diff", ascending=False)
    df_diff.to_csv(f"{filter_dir}/filter_discriminability.csv", index=False)

    with open(log_file, "a") as log:
        log.write(f"\n[FILTER DIFF] Discriminability scores (threshold={filter_diff_threshold}):\n")
        for _, row in df_diff.iterrows():
            log.write(f"  Cluster {int(row['cluster'])}: "
                        f"filter_diff={row['filter_diff']:.4f} "
                        f"({'OK' if row['discriminative'] else 'FLAGGED'})\n")

    # ----------------------------------------------------------------
    # 7. Save everything
    # ----------------------------------------------------------------
    np.save(f"{filter_dir}/consensus_filter_weights.npy", consensus_conv)
    np.save(f"{filter_dir}/consensus_output_weights.npy", consensus_out)
    np.save(f"{filter_dir}/all_pooled_filter_weights.npy", pooled_conv)
    np.save(f"{filter_dir}/all_pooled_output_weights.npy", pooled_out)

    # Save per-cluster ranked gene weight CSVs
    col_label = "gene" if genes is not None else "marker"
    for m in consensus_meta:
        cluster_id = m["cluster"]
        df_filter = pd.DataFrame({
            col_label: list(m["all_gene_weights"].keys()),
            "weight":  list(m["all_gene_weights"].values()),
        })
        # Already sorted by |weight| descending from ranked_idx above
        df_filter.to_csv(
            f"{filter_dir}/cluster{cluster_id}_filter_weights.csv", index=False
        )

    with open(f"{filter_dir}/consensus_filter_meta.json", "w") as f:
        json.dump(consensus_meta, f, indent=2)

    with open(log_file, "a") as log:
        log.write(f"\n[FILTERS] {n_clusters} consensus filters saved to {filter_dir}\n")
        for m in consensus_meta:
            log.write(f"  Cluster {m['cluster']}: {m['n_members']} members, "
                      f"source run={m['source_run']}\n")

    print(f"[INFO] Consensus filters saved to {filter_dir}")
    return {
        "consensus_conv":   consensus_conv,
        "consensus_out":    consensus_out,
        "all_pooled_conv":  pooled_conv,
        "all_pooled_out":   pooled_out,
        "accuracies":       accuracies,
        "f1_scores":        f1_scores,
        "passing_runs":     passing_runs,
        "cluster_labels":   labels_clust,
        "meta":             consensus_meta,
    }
    
# Model training and fitting functions ---------------------------------------------------------

def train(model, 
          train_loader, 
          loss_fn, 
          optimizer, 
          epoch, 
          log_file, 
          device):
    '''Train function for the model. To evaluate the training, multiple different
    measures are used: Balanced Accuracy, Precision, Recall.

    Arguments
    ---
    model : nn.Module descendant
        Model through which training data is passed
    train_loader : DataLoader
        Training data with features and labels
    loss_fn : function
        Method of loss calculation
    optimizer : function (taken using torch.optim)
        Optimisation method
    epoch : int
        Number of passes through the network

    Returns
    ---
    epoch_loss : ?
        Information on loss (loss vector ?)
    '''
    model.train()

    train_loss = 0
    counter = 0

    all_true_classes = []
    all_pred_classes = []
    with tqdm(train_loader, total=len(train_loader), unit="batch") as tepoch:
        for batch in tepoch:
            counter += 1
            tepoch.set_description(f"Epoch {epoch}")

            # Send the input to the device
            X, y = batch['intensity'].to(device), batch['label'].to(device)

            # Compute prediction and loss
            pred = model(X)

            # Loss calculation
            loss = torch.sum(loss_fn(pred, y))

            # Backpropagation
            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
            # Batch metrics - first detach vectors
            pred = pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            class_preds = np.argmax(pred, axis = 1)

            # Save class information     
            all_true_classes.extend(y)
            all_pred_classes.extend(class_preds)
           
            # Progres pbar
            postfix = {}
            postfix["Train: loss"] = f"{train_loss / counter:.5f}"
            tepoch.set_postfix(postfix)
        
        # Loss and protein metrics
        epoch_loss = train_loss / counter
        ba = balanced_accuracy_score(all_true_classes, all_pred_classes)
        f1 = f1_score(all_true_classes, all_pred_classes, 
                      average = "macro")

        # Save to log
        with open(log_file, "a") as log:
            log.write(f"Epoch {epoch} train : Balanced Accuracy = {ba:.2f}, Macro F1-score = {f1:.2f} | Loss : {loss:.2f}\n")
    
    return epoch_loss, ba, f1


def validate(model, val_loader, loss_fn, epoch, log_file, device):
    '''Train function for the model. 

    Arguments
    ---
    model : nn.Module descendant
        Model through which training data is passed
    val_loader : DataLoader
        Validation data with features and labels
    loss_fn : function
        Method of loss calculation
    optimizer : function (taken using torch.optim)
        Optimisation method
    epoch : int
        Number of passes through the network

    Returns
    ---
    epoch_loss : ?
        Information on loss (loss vector ?)
    '''
    model.eval()

    val_loss = 0
    counter = 0

    all_true_classes = []
    all_pred_classes = []
    with torch.no_grad():
        for batch in val_loader:
            counter += 1

            #Send the input to the device
            X, y = batch['intensity'].to(device), batch['label'].to(device)

            #Compute prediction and loss
            pred = model(X)

            # Loss calculation
            loss = torch.sum(loss_fn(pred, y))
            val_loss += loss.item()
        
            # Batch metrics - first detach vectors
            pred = pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            class_preds = np.argmax(pred, axis = 1)

            # Save class information     
            all_true_classes.extend(y)
            all_pred_classes.extend(class_preds)
        
        #Loss and protein metrics
        epoch_loss = val_loss / counter

        ba = balanced_accuracy_score(all_true_classes, all_pred_classes)
        f1 = f1_score(all_true_classes, all_pred_classes, 
                      average = "macro")
        
        # Save to log
        with open(log_file, "a") as log:
            log.write(f"Epoch {epoch} validation : Balanced Accuracy = {ba:.2f}, Macro F1-score = {f1:.2f} | Loss : {loss:.2f}\n")

        #Prints
        print(f"Validation performance across samples :")
        print(f"Balanced Accuracy Score : {ba:.2f}")
        print(f"Weighted F1 score : {f1:.2f}")
        
    return epoch_loss, ba, f1


def test(model, test_loader, loss_fn, epoch, figure_path, log_file, device):
    '''Train function for the model. 

    Arguments
    ---
    model : nn.Module descendant
        Model through which training data is passed
    val_loader : DataLoader
        Validation data with features and labels
    loss_fn : function
        Method of loss calculation
    optimizer : function (taken using torch.optim)
        Optimisation method
    epoch : int
        Number of passes through the network

    Returns
    ---
    epoch_loss : ?
        Information on loss (loss vector ?)
    '''
    model.eval()

    test_loss = 0
    counter = 0

    all_true_classes = []
    all_pred_classes = []
    #samples = []
    with torch.no_grad():
        for batch in test_loader:
            counter += 1

            #Send the input to the device
            X, y = batch['intensity'].to(device), batch['label'].to(device)

            #Compute prediction and loss
            pred = model(X)

            # Loss calculation
            loss = torch.sum(loss_fn(pred, y))
            test_loss += loss.item()
        
            # Batch metrics - first detach vectors
            pred = pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            class_preds = np.argmax(pred, axis = 1)

            # Save class information     
            all_true_classes.extend(y)
            all_pred_classes.extend(class_preds)
            #samples.append(X)
        
        #Loss and protein metrics
        epoch_loss = test_loss / counter

        ba = balanced_accuracy_score(all_true_classes, all_pred_classes)
        f1 = f1_score(all_true_classes, all_pred_classes, 
                      average = "macro")

        #Prints
        print(f"\n Performance across samples : \n")
        print(f"Balanced Accuracy Score : {ba}")
        print(f"Weighted F1 score : {f1}")
    
    return epoch_loss, all_pred_classes, all_true_classes


def fit(dataset: CIMADataset,               # CIMA Args
        anchor: str,
        K: int,
        ncell: int,
        genes: list,
        save_loc: str,
        nruns: int = 5, 
        dendrogram_cutoff: float = 0.4,
        n_val_folds: int = 3,              
        maxpool_percentages : list = [1, 5, 10, 20, 50, 100],
        nfilter_selection: list = [3, 4, 5, 6, 7, 8, 9, 10],
        dropout: bool = True,
        dropout_p: float = 0.5,
        l2: float = 1e-4,
        background: bool = False,
        batch_size = 256,
        lr: float = 0.01,
        epochs: int = 20, 
        num_workers: int = 8,
        early_stopping = True,
        patience: int = 3,
        seed: int = 12345,
        classification: bool = True):
    '''Fit an S3CIMA Model to a dataset.

    Arguments
    ---

    Returns
    ---

    '''
    set_seed(seed)
    start = time.time()

    # Set nmarkers from len of genes
    nmarkers = len(genes)

    # Set device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}\n")

    # Results folder
    now = datetime.now()
    today = datetime.today()
    current_time = now.strftime("%H_%M_%S")
    current_day = today.strftime("%d_%m_%Y")

    result_folder = f"{save_loc}/results_Anchor{anchor}_K{K}"
    os.makedirs(result_folder, exist_ok=True)
         
    # Generate the datasets and set classes
    test_subset, folds = make_patient_stratified_splits(dataset, 
                                                        n_val_folds=n_val_folds,
                                                        seed=seed)
    if dataset.random_ctrl:
        n_classes = len(np.unique(dataset.label)) + 1
    else:
        n_classes = len(np.unique(dataset.label))
    
    # Initialise loss
    if classification:
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("Not yet implemented regression.")
    
    # Log model training
    log_file = f"{result_folder}/training_log_{current_day}_{current_time}.txt"
    with open(log_file, "w") as log:
        log.write(f""""[TRAINING LOG] - S3CIMA MODEL 
Reporting training results for {anchor} anchor, K = {K}, ncell = {ncell}. 
------------------------------------------------------------
\n
Classification - {classification} (regression if False)
Background samples - {dataset.random_ctrl} (if True, expect one more 'background' class)
Number of classes - {n_classes} 
\n
------------------------------------------------------------
                  """)     

    # ----------------------------------------------------------------
    # 1. Run the S3-CIMA training
    # ----------------------------------------------------------------
    max_ba = 0
    max_f1 = 0

    # Model save log
    save_model = f"{result_folder}/models"
    os.makedirs(save_model, exist_ok=True)\
    
    # Save loss vectors as well
    loss_vectors, val_loss_vectors = {}, {}
    accuracy_vectors, val_accuracy_vectors = {}, {}
    f1_vectors, val_f1_vectors = {}, {}

    for run in range(nruns):

        # Reset to calculate best model
        max_ba = 0
        max_f1 = 0

        # First select the parameters for this run
        nfilters = np.random.choice(nfilter_selection)
        print('Number of filters: %d' % nfilters)
        mp = maxpool_percentages[run % len(maxpool_percentages)]
        k = max(1, int(mp / 100. * ncell))
        print('Cells pooled: %d' % k)

        # Set mode + optimiser
        if classification:
            regression = False
        else:
            regression = True
        selection_type = "mean"

        # LOG
        with open(log_file, "a") as log:
            log.write(f"""
Currently on run {run} :

Number of filters - {nfilters}
Number of cells pooled (max pooling) - {k}

Batch Size - {batch_size}
Learning Rate - {lr}
Optimiser - Adam
""") 

        # Iterate over the train/validation folds
        count = 0
        for (train_data, val_data) in folds:

            # Instantiate model
            model = CellCNN(nmark=nmarkers,
                        nfilter=nfilters,
                        k=k,
                        n_classes=n_classes,
                        dropout = dropout ,
                        dropout_p = dropout_p ,
                        regression = regression,
                        selection_type = selection_type
                        )
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), 
                            lr=lr,amsgrad = False, 
                            weigth_decay=l2) 

            # Early stopping object
            if early_stopping:
                print("[INFO]: Initializing early stopping")
                early_stopping = EarlyStopping(patience=patience, min_delta=0)

            with open(log_file, "a") as log:
                log.write(f"""
Fold number {count} - {len(train_data)} train samples, {len(val_data)} validation samples
--------------------------------------------------------------------
""") 
            # Train model for set number of epochs
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            # Objects to save loss vectors
            loss_vector, val_loss_vector = [], []
            accuracy_vector, val_accuracy_vector = [], []
            f1_vector, val_f1_vector = [], []

            for epoch in range(epochs):

                # Train and validate functions
                train_loss, train_ba, train_f1 = train(model, train_loader, loss_fn, optimizer, epoch, log_file, device)
                val_loss, val_ba, val_f1 = validate(model, val_loader, loss_fn, epoch, log_file, device)

                loss_vector.append(train_loss)
                val_loss_vector.append(val_loss)
                accuracy_vector.append(train_ba)
                val_accuracy_vector.append(val_ba)
                f1_vector.append(train_f1)
                val_f1_vector.append(val_f1)

                if early_stopping:
                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        break

                # After epoch save best model weights
                if val_ba > max_ba and val_f1 > max_f1:
                    max_ba = train_ba
                    max_f1 = val_f1

                    # From this best model 
                    checkpoint = {
                        'run': run,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ba': val_ba,
                        'f1' : val_f1,
                        "nmarkers": nmarkers,
                        "nfilter": nfilters,
                        "maxpool_percentage": mp,
                        "k": k,
                        "n_classes": n_classes,
                        "dropout": dropout ,
                        "dropout_p": dropout_p ,
                        "regression": regression,
                        "selection_type": selection_type,
                    }
                    torch.save(checkpoint, f"{save_model}/best_model_run{run}.pth")
                
                loss_vectors[f"run{run}_fold{count}"] = loss_vector
                val_loss_vectors[f"run{run}_fold{count}"] = val_loss_vector
                accuracy_vectors[f"run{run}_fold{count}"] = accuracy_vector
                val_accuracy_vectors[f"run{run}_fold{count}"] = val_accuracy_vector
                f1_vectors[f"run{run}_fold{count}"] = f1_vector
                val_f1_vectors[f"run{run}_fold{count}"] = val_f1_vector
                
            # Fold number
            count += 1
    
    end = time.time()

    with open(log_file, "a") as log:
                log.write(f"""
--------------------------------------------------------------------
Training finished !
Time = {(end-start)/60:.3f} minutes
--------------------------------------------------------------------

Now re-running best models on a randomly chosen validation set to identify
the most discriminative filters across runs:
""")
    
    print(f"Training time: {(end-start)/60:.3f} minutes\n")
    # print(len(predictions_across_embds))
    print("Model complete")

    # ----------------------------------------------------------------
    # 2. Get discriminative filters
    # ----------------------------------------------------------------

    # Now identify discriminative filters on the validation set
    # For now, pick one of the validation sets randomly
    validation_dataset = random.choice([v[1]for v in folds])
    validation_dataset = validation_dataset.subset

    accur_thres = (n_classes - 1) / n_classes
    filter_results = get_discriminative_filters(
        save_model = save_model,
        nruns = nruns,
        validation_subset = validation_dataset,
        loss_fn = loss_fn,
        n_classes = n_classes,
        nmarkers = nmarkers,
        lr = lr,
        batch_size = batch_size,
        num_workers = num_workers,
        log_file = log_file,
        device = device,
        accur_thres = accur_thres,       
        dendrogram_cutoff = dendrogram_cutoff, 
        result_folder = result_folder,
        genes=genes,
        has_background=background,
        classification=classification,
    )

    with open(log_file, "a") as log:
                log.write(f"""
--------------------------------------------------------------------
Discriminative filters found 
A total of {len(filter_results['meta'])} filter clusters identified. 
--------------------------------------------------------------------

Running the models producing the top discriminative filters on test to 
report classification: 
""")
                
    # ----------------------------------------------------------------
    # 3. Validate on the test
    # ----------------------------------------------------------------

    # Now using these results, we can test the top filters on test
    # and report classification.
    filtered_meta = [m for m in filter_results["meta"] if m["discriminative"]]

    t_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    final_results = []
    counter = 0
    for r in filtered_meta:
        
        run_of_origin = r["source_run"]

        ckpt_path = f"{save_model}/best_model_run{run_of_origin}.pth"
        if not os.path.exists(ckpt_path):
            print(f"[WARN] No checkpoint for run {run_of_origin}, skipping.")
            continue

        params = torch.load(ckpt_path, map_location=device, weights_only=False)

        model = CellCNN(
            nmark = params["nmarkers"],
            nfilter = params["nfilter"],
            k = params["k"],
            n_classes = params["n_classes"],
            dropout = params["dropout"],
            dropout_p = params["dropout_p"],
            regression = params["regression"],
            selection_type = params["selection_type"],
        )
        model.load_state_dict(params["model_state_dict"])
        model.to(device)

        test_loss, preds, trues = test(
            model, t_loader, loss_fn, 0, result_folder, log_file, device
        )

        ba = balanced_accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average = "macro")
        print(f"Top model Test balanced accuracy: {ba:.4f}")
        print(f"Top model Test F1 Score: {f1:.4f}")

        if ba > accur_thres:
             r['test_ba'] = ba
             final_results.append(r)
             counter += 1

        with open(log_file, "a") as log:
            log.write(f"\n[Top model run {run_of_origin}, {r['filter_diff']:.2f} discriminative score] - "
                      f"Test balanced accuracy: {ba:.2f}, Test Macro F1 Score: {f1:.2f}\n")
            log.write(f'Found {counter} discriminative filters with test balanced accuracy above {accur_thres:.2f}\n')

    if len(final_results) == 0:
        print(f"[INFO] No discriminative filters had test balanced accuracy above {accur_thres:.2f}.")
        print(f"Returning all discriminative filters")

        with open(log_file, "a") as log:
            log.write(f"[INFO] No discriminative filters had test balanced accuracy above {accur_thres:.2f}.")

        final_results = filtered_meta

    return final_results, loss_vectors, val_loss_vectors, accuracy_vectors, val_accuracy_vectors, test_subset, folds, result_folder