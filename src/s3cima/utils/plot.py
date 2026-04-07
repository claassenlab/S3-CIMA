"""Plotting functions to for the S3-CIMA model outputs"""

# Imports

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


# Functions ---------------------------------------------------------------

def plot_filter_weights(
    meta: list,
    save_path: str = ".",
    filename: str = "filter_cluster_weights",
    show: bool = True,
    top_n_genes: int = None,
):
    """
    Plot a heatmap of gene weights across all filter clusters.

    Parameters
    ----------
    meta        : list of dicts — the 'meta' key from extract_and_save_filters output.
                  Each dict must contain 'cluster', 'dominant_class', 'all_gene_weights'.
    save_path   : str  — directory to save the HTML (and PNG) file.
    filename    : str  — base filename without extension.
    show        : bool — if True, calls fig.show() to display in browser/notebook.
    top_n_genes : int  — if provided, only plot the top N genes ranked by max
                         absolute weight across all clusters. Useful when nmarkers is large.
    """
    # Build Dataframe
    records = {}
    for m in meta:
        cluster_label = f"Cluster {m['cluster']} (Filter Diff: {m['filter_diff']:.2f})"
        records[cluster_label] = m["all_gene_weights"]  # {gene: weight}

    df = pd.DataFrame(records).T  # (n_clusters, n_genes)
    df = df.fillna(0.0)

    # Restrict top N genes
    if top_n_genes is not None and top_n_genes < df.shape[1]:
        max_abs = df.abs().max(axis=0)
        top_genes = max_abs.nlargest(top_n_genes).index
        df = df[top_genes]

    # Sort
    gene_order = df.abs().max(axis=0).sort_values(ascending=False).index
    df = df[gene_order]

    # Heatmap
    z        = df.values                          # (n_clusters, n_genes)
    x_labels = df.columns.tolist()                # gene names
    y_labels = df.index.tolist()                  # cluster labels

    # Symmetric color scale centred at 0
    abs_max = np.abs(z).max()

    fig = go.Figure(go.Heatmap(
        z           = z,
        x           = x_labels,
        y           = y_labels,
        colorscale  = "RdBu_r",
        zmid        = 0,
        zmin        = -abs_max,
        zmax        =  abs_max,
        colorbar    = dict(title="Weight"),
        hoverongaps = False,
        hovertemplate = "Gene: %{x}<br>Cluster: %{y}<br>Weight: %{z:.4f}<extra></extra>",
    ))

    n_genes    = len(x_labels)
    n_clusters = len(y_labels)

    fig.update_layout(
        title       = dict(
            text = (
                f"Filter Cluster Gene Weights "
                f"({n_clusters} clusters, {n_genes} genes)"
                f"<br><span style='font-size:14px;font-weight:normal;'>"
                f"Red = positive weight (activating) · Blue = negative (suppressing)"
                f"</span>"
            ),
            x    = 0.5,
            xanchor = "center",
        ),
        xaxis = dict(
            title_text   = "Gene / Marker",
            tickangle    = -45,
            tickfont     = dict(size=max(8, min(12, int(400 / max(n_genes, 1))))),
        ),
        yaxis = dict(
            title_text   = "Filter Cluster",
            autorange    = "reversed",   # cluster 0 at top
            tickfont     = dict(size=11),
        ),
        height = max(300, 120 * n_clusters),
    )

    # Save as html
    figure_path = f"{save_path}/plots"
    os.makedirs(figure_path, exist_ok=True)
    html_path = os.path.join(figure_path, f"{filename}.html")
    fig.write_html(html_path)
    print(f"[INFO] Heatmap saved to {html_path}")

    if show:
        fig.show()

    return fig


def get_high_response_cells_test(
    meta: list,
    dataset,
    genes: list,
    filter_threshold: float = 0.9,
) -> dict:
    """
    For each consensus filter cluster, compute the pooled filter response
    for every non-background neighbourhood in the test dataset, threshold at
    the filter_threshold quantile, and return the cell IDs of the
    high-responding neighbourhoods.

    Parameters
    ----------
    meta             : list of dicts — 'meta' key from extract_and_save_filters.
                       Each dict must contain 'all_gene_weights', 'cluster',
                       'dominant_class'. Optionally 'maxpool_percentage'.
    dataset          : NormalizedCIMASubset — samples must be
                       dicts with keys 'intensity', 'is_ctrl', 'cellids'.
    genes            : list of str — marker names in the same order as the
                       intensity columns used during training.
    filter_threshold : float in [0, 1) — responses below this quantile are
                       discarded. Default 0.9 retains the top 10%.

    Returns
    -------
    dict keyed by cluster_id, each value:
        {
          "cell_ids"                : list — unique cell IDs with high response,
          "responses"               : np.array — activations of high neighbourhoods,
          "all_responses"           : np.array — full activation distribution,
          "threshold_val"           : float — quantile cutoff used,
          "n_neighbourhoods_above"  : int,
          "n_cell_ids"              : int,
        }
    """
    assert 0.0 <= filter_threshold < 1.0, \
        "filter_threshold must be in [0, 1)"

    # Collect non-background samples
    anchor_intensities = []
    anchor_cellids = []
    anchor_labels = [] 
    anchor_samples = []

    for item in dataset:
        if item["is_ctrl"]:
            continue
        anchor_intensities.append(item["intensity"].numpy())  # (ncell, nmark)
        anchor_cellids.append(item["cellids"])                # list of cell IDs
        anchor_labels.append(int(item["label"].item()))
        anchor_samples.append(item["pat"].item())

    anchor_labels = np.array(anchor_labels)

    if len(anchor_intensities) == 0:
        raise ValueError(
            "No non-background samples found. Check the is_ctrl flag."
        )

    nmark = anchor_intensities[0].shape[1]
    if len(genes) != nmark:
        raise ValueError(
            f"len(genes)={len(genes)} does not match intensity nmark={nmark}. "
            "Ensure genes are provided in the same column order as the "
            "intensity matrix used during training."
        )

    print(f"[INFO] Computing filter responses over "
          f"{len(anchor_intensities)} anchor neighbourhoods "
          f"across {len(meta)} filter clusters.")

    # Neighbourhood responses
    results = {}

    for m in meta:
        filter_diff = m["filter_diff"]
        cluster_id = m["cluster"]
        gene_weights = m["all_gene_weights"]   # {gene: weight}, |w|-ranked order
        mp = m.get("maxpool_percentage", 100.0)
        ntop_frac = mp / 100.0

        # Reconstruct conv_w in original marker column order
        conv_w = np.array(
            [gene_weights[g] for g in genes], dtype=np.float32
        )

        # Compute pooled ReLU activation for each neighbourhood
        responses = np.zeros(len(anchor_intensities), dtype=np.float32)
        for i, x in enumerate(anchor_intensities):
            g  = np.maximum(0.0, x @ conv_w)
            ntop = max(1, int(ntop_frac * x.shape[0]))
            responses[i]  = np.mean(np.sort(g)[-ntop:])

        # Collect responsive cell IDs
        threshold_val = float(np.quantile(responses, filter_threshold))
        high_idx = np.where(responses >= threshold_val)[0]

        # Flatten and deduplicate cell IDs across high-responding neighbourhoods
        seen = set()
        unique_cell_ids = []
        for idx in high_idx:
            for cid in anchor_cellids[idx]:
                if cid not in seen:
                    seen.add(cid)
                    unique_cell_ids.append(cid)

        results[cluster_id] = {
            "cell_ids": unique_cell_ids,
            "responses": responses[high_idx],
            "all_responses": responses,
            "threshold_val": threshold_val,
            "n_neighbourhoods_above": len(high_idx),
            "n_cell_ids": len(unique_cell_ids),
            "filter_diff": filter_diff,
        }

        print(f"  [Cluster {cluster_id}]  "
              f"threshold={threshold_val:.4f} | "
              f"{len(unique_cell_ids)} unique cell IDs")
        
    # ALso save sample info
    results["samples"] = set(anchor_samples)

    return results


def get_high_response_cells_train(
    meta: list,
    validation_splits: list,
    genes: list,
    filter_threshold: float = 0.9,
) -> dict:
    """
    For each consensus filter cluster, compute the pooled filter response
    for every non-background neighbourhood in the whole dataset, threshold at
    the filter_threshold quantile, and return the cell IDs of the
    high-responding neighbourhoods.

    Parameters
    ----------
    meta             : list of dicts — 'meta' key from extract_and_save_filters.
                       Each dict must contain 'all_gene_weights', 'cluster',
                       'dominant_class'. Optionally 'maxpool_percentage'.
    validation_splits: list of tuples — each tuple contains (train_subset, val_subset)
    genes            : list of str — marker names in the same order as the
                       intensity columns used during training.
    filter_threshold : float in [0, 1) — responses below this quantile are
                       discarded. Default 0.9 retains the top 10%.

    Returns
    -------
    dict keyed by cluster_id, each value:
        {
          "cell_ids"                : list — unique cell IDs with high response,
          "responses"               : np.array — activations of high neighbourhoods,
          "all_responses"           : np.array — full activation distribution,
          "threshold_val"           : float — quantile cutoff used,
          "n_neighbourhoods_above"  : int,
          "n_cell_ids"              : int,
        }
    """
    assert 0.0 <= filter_threshold < 1.0, \
        "filter_threshold must be in [0, 1)"

    # Non-background samples
    anchor_intensities = []
    anchor_cellids = []
    anchor_labels = [] 
    anchor_samples = []

    # Take only the first validation split
    split = validation_splits[0]

    for dataset in split:
        for item in dataset:
            if item["is_ctrl"]:
                continue
            anchor_intensities.append(item["intensity"].numpy())  # (ncell, nmark)
            anchor_cellids.append(item["cellids"])                # list of cell IDs
            anchor_labels.append(int(item["label"].item()))
            anchor_samples.append(item["pat"].item())

    anchor_labels = np.array(anchor_labels)

    if len(anchor_intensities) == 0:
        raise ValueError(
            "No non-background samples found. Check the is_ctrl flag."
        )

    nmark = anchor_intensities[0].shape[1]
    if len(genes) != nmark:
        raise ValueError(
            f"len(genes)={len(genes)} does not match intensity nmark={nmark}. "
            "Ensure genes are provided in the same column order as the "
            "intensity matrix used during training."
        )

    print(f"[INFO] Computing filter responses over "
        f"{len(anchor_intensities)} anchor neighbourhoods "
        f"across {len(meta)} filter clusters.")

    # Response
    results = {}

    for m in meta:
        filter_diff = m["filter_diff"]
        cluster_id = m["cluster"]
        gene_weights = m["all_gene_weights"]   # {gene: weight}, |w|-ranked order
        mp = m.get("maxpool_percentage", 100.0)
        ntop_frac = mp / 100.0

        # Reconstruct conv_w in original marker column order
        conv_w = np.array(
            [gene_weights[g] for g in genes], dtype=np.float32
        )  # (nmark,)

        # Compute pooled ReLU activation for each neighbourhood
        responses = np.zeros(len(anchor_intensities), dtype=np.float32)
        for i, x in enumerate(anchor_intensities):
            g             = np.maximum(0.0, x @ conv_w)       # (ncell,)
            ntop          = max(1, int(ntop_frac * x.shape[0]))
            responses[i]  = np.mean(np.sort(g)[-ntop:])

        # Threshold
        threshold_val = float(np.quantile(responses, filter_threshold))
        high_idx      = np.where(responses >= threshold_val)[0]

        # Flatten and deduplicate cell IDs across high-responding neighbourhoods
        seen            = set()
        unique_cell_ids = []
        for idx in high_idx:
            for cid in anchor_cellids[idx]:
                if cid not in seen:
                    seen.add(cid)
                    unique_cell_ids.append(cid)

        results[cluster_id] = {
            "cell_ids": unique_cell_ids,
            "responses": responses[high_idx],
            "all_responses": responses,
            "threshold_val": threshold_val,
            "n_neighbourhoods_above": len(high_idx),
            "n_cell_ids": len(unique_cell_ids),
            "filter_diff": filter_diff
        }

        print(f"  [Cluster {cluster_id}]  "
              f"threshold={threshold_val:.4f} | "
              f"{len(unique_cell_ids)} unique cell IDs")
    
    # ALso save sample info
    results["samples"] = set(anchor_samples)
    return results


def save_high_response_stats(
    high_response: dict,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cell_id_col: str,
    cell_type_col: str,
    sample_id_col: str,
    save_path: str = ".",
    test: bool = True
):
    """
    For each consensus filter cluster and each sample, saves:
      a) selected_cells.csv     — full rows of high-response cells
      b) cell_type_counts.csv   — count and proportion of each cell type
                                  in the selected cells
      c) enrichment.csv         — per cell type enrichment score:
                                  prop_selected / prop_background
                                  (log2 fold-change style, with pseudocount)

    Parameters
    ----------
    high_response  : dict — output of get_high_response_cells, keyed by cluster_id
    cells_csv      : str  — path to cell metadata CSV
    x_col          : str  — column name for x coordinate
    y_col          : str  — column name for y coordinate
    cell_id_col    : str  — column name for cell ID
    cell_type_col  : str  — column name for cell type
    sample_id_col  : str  — column name for sample/image ID
    save_path      : str  — root directory for outputs
    test           : bool — if True, saves under <save_path>/test/; else under <save_path>/train/
    """
    if test:
        save_path = f"{save_path}/test"
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = f"{save_path}/train"
        os.makedirs(save_path, exist_ok=True)


    # Cell metadata
    required_cols = {x_col, y_col, cell_id_col, cell_type_col, sample_id_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    
    # Filter dataframe by sample ids and rm the key
    sample_ids = high_response["samples"]
    del high_response["samples"]
    df = df[df[sample_id_col].isin(sample_ids)].copy()
    if df.empty:
        raise ValueError("No rows left after filtering by test_sample_ids.")

    # Iterate over cluster then samples
    for cluster_id, res in high_response.items():
        high_ids = set(res["cell_ids"])
        filter_diff = res["filter_diff"]

        cluster_dir = os.path.join(save_path, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        # Accumulate per-sample enrichment rows for a combined summary
        all_enrichment = []

        for sample_id, sdf in df.groupby(sample_id_col):

            sdf = sdf.copy()
            sdf["_selected"] = sdf[cell_id_col].isin(high_ids)

            # Skip samples with no high-response cells
            n_selected = sdf["_selected"].sum()
            if n_selected == 0:
                print(f"  [Cluster {cluster_id} | Sample {sample_id}] "
                      f"0 selected cells — skipping.")
                continue

            sel_cells = sdf[sdf["_selected"]]
            bg_cells = sdf[~sdf["_selected"]]
            n_total = len(sdf)
            n_bg = len(bg_cells)

            sample_dir = os.path.join(cluster_dir, f"sample_{sample_id}")
            os.makedirs(sample_dir, exist_ok=True)

            # Create selected cells CSV per cluster
            sel_cells.drop(columns=["_selected"]).to_csv(
                os.path.join(sample_dir, "selected_cells.csv"), index=False
            )

            # Plot the cell filter response
            fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
            ax.scatter(bg_cells[x_col],  bg_cells[y_col],  c="#b0b0b0", s = 0.25)  # grey
            ax.scatter(sel_cells[x_col], sel_cells[y_col], c="#8b0000", s = 0.25)  # dark red
            fig.savefig(os.path.join(sample_dir, "spatial_plot.png"))
            plt.close(fig)

            # Celltype counts
            ct_counts = (
                sel_cells[cell_type_col]
                .value_counts()
                .rename_axis("cell_type")
                .reset_index(name="count_selected")
            )
            ct_counts["prop_selected"] = ct_counts["count_selected"] / n_selected
            ct_counts.to_csv(
                os.path.join(sample_dir, "cell_type_counts.csv"), index=False
            )

            # ------------------------------------------------------------
            # c) Enrichment score per cell type
            #    score = log2( (prop_selected + pseudo) /
            #                  (prop_background + pseudo) )
            #    pseudo = 1 / (n_selected + n_bg) — one-cell pseudocount
            # ------------------------------------------------------------
            all_cell_types = df[cell_type_col].unique()
            pseudo         = 1.0 / (n_selected + n_bg)

            enrichment_rows = []
            for ct in all_cell_types:
                count_sel = (sel_cells[cell_type_col] == ct).sum()
                count_bg = (bg_cells[cell_type_col]  == ct).sum()

                prop_sel = count_sel / n_selected if n_selected > 0 else 0.0
                prop_bg = count_bg  / n_bg       if n_bg       > 0 else 0.0

                log2_fc = np.log2(
                    (prop_sel + pseudo) / (prop_bg + pseudo)
                )

                enrichment_rows.append({
                    "cell_type":        ct,
                    "count_selected":   int(count_sel),
                    "count_background": int(count_bg),
                    "prop_selected":    round(prop_sel,  6),
                    "prop_background":  round(prop_bg,   6),
                    "log2_enrichment":  round(log2_fc,   6),
                    "sample_id": sample_id,
                    "cluster_id": cluster_id,
                    "filter_diff": round(filter_diff, 4)
                })

            enrich_df = (
                pd.DataFrame(enrichment_rows)
                .sort_values("log2_enrichment", ascending=False)
            )
            enrich_df.to_csv(
                os.path.join(sample_dir, "enrichment.csv"), index=False
            )

            all_enrichment.append(enrich_df)

            print(f"  [Cluster {cluster_id} | Sample {sample_id}] "
                  f"{n_selected}/{n_total} cells selected — saved to {sample_dir}")

        # ----------------------------------------------------------------
        # 3. Save a cluster-level summary: mean enrichment across samples
        # ----------------------------------------------------------------
        if all_enrichment:
            combined = pd.concat(all_enrichment, ignore_index=True)
            summary  = (
                combined
                .groupby("cell_type")["log2_enrichment"]
                .agg(
                    mean_log2_enrichment="mean",
                    std_log2_enrichment="std",
                    n_samples="count",
                )
                .reset_index()
                .sort_values("mean_log2_enrichment", ascending=False)
            )
            summary["cluster_id"] = cluster_id
            summary["filter_diff"] = filter_diff
            summary.to_csv(
                os.path.join(cluster_dir, "enrichment_summary.csv"), index=False
            )
            print(f"[Cluster {cluster_id}] Summary saved → "
                  f"{os.path.join(cluster_dir, 'enrichment_summary.csv')}")
            
    return save_path
            

def enrichment_summary(save_path: str, output_file: str = None) -> str:
    """
    Scans *save_path* for all enrichment_summary.csv files produced by
    save_high_response_stats and saves a single interactive Plotly HTML
    figure with one bar chart per filter cluster.

    Parameters
    ----------
    save_path   : str — root directory passed to save_high_response_stats
    output_file : str — path for the HTML output; defaults to
                        <save_path>/enrichment_summary_report.html

    Returns
    -------
    str — absolute path to the written HTML file
    """
    if output_file is None:
        output_file = os.path.join(save_path, "enrichment_summary_report.html")

    # Find enrichment files
    csv_files = sorted(
        glob.glob(os.path.join(save_path, "cluster_*", "enrichment_summary.csv"))
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No enrichment_summary.csv files found under: {save_path}"
        )

    # Load data 
    clusters = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path).sort_values("mean_log2_enrichment", ascending=True)
        cluster_id = (
            df["cluster_id"].iloc[0]
            if "cluster_id" in df.columns
            else os.path.basename(os.path.dirname(csv_path)).replace("cluster_", "")
        )
        filter_diff = (
            df["filter_diff"].iloc[0]
            if "filter_diff" in df.columns
            else "Not provided"
        )
        filter_diff = round(float(filter_diff), 2)
        clusters.append((cluster_id, filter_diff, df))

    # Figure panel
    n = len(clusters)
    ncols = int(min(3, n))
    nrows = int(np.ceil(n / ncols))
    fig = make_subplots(
    rows=nrows, cols=ncols,
    subplot_titles=[f"Consensus filter {cid} (diff: {fil})" for cid, fil, _ in clusters],
    vertical_spacing= 0.3 / max(nrows, 1),
    horizontal_spacing= 0.2 / max(ncols, 1),
    ) 

    for i, (cluster_id, filter_val, df) in enumerate(clusters):
        row = i // ncols + 1
        col = i % ncols + 1

        has_std = "std_log2_enrichment" in df.columns and df["std_log2_enrichment"].notna().any()

        # Custom hover data per sample
        customdata=list(zip(
            df["cell_type"],
            df["mean_log2_enrichment"].round(3),
            df["std_log2_enrichment"].fillna(0).round(3) if has_std else [0]*len(df),
        ))

        bar_colors = [
            "#8b0000" if v >= 0 else "#2166ac"
            for v in df["mean_log2_enrichment"]
        ]

        fig.add_trace(
            go.Bar(
                x=df["mean_log2_enrichment"],
                y=df["cell_type"],
                orientation="h",
                marker_color=bar_colors,
                error_x=dict(
                    type="data",
                    array=df["std_log2_enrichment"].fillna(0).tolist(),
                    visible=True,
                    thickness=1.5,
                    width=4,
                ) if has_std else None,
                customdata=customdata,
                hovertemplate="<b>%{customdata[0]}</b><br>log2 enrichment: %{customdata[1]:.3f}"
                              "<br>± std: %{customdata[2]:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=row, col=col,
        )
        fig.update_xaxes(title_text="log2 enrichment", row=row, col=col, zeroline=True,
                         zerolinewidth=1, zerolinecolor="rgba(0,0,0,0.3)")
        fig.update_yaxes(automargin=True, row=row, col=col)

    fig.update_layout(
        height=max(300, 380 * nrows),
        title_text="Filter Cluster Enrichment Summary on test samples",
        title_font_size=16,
        paper_bgcolor="#f7f6f2",
        plot_bgcolor="#ffffff",
        font=dict(family="Arial, sans-serif", size=12),
    )
    fig.show()

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    fig.write_html(output_file, include_plotlyjs="cdn")
    print(f"Enrichment summary report written → {os.path.abspath(output_file)}")
    return os.path.abspath(output_file)