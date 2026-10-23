"""Plotting functions to for the S3-CIMA model outputs"""

# Imports

import os
import glob
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# CONSTANTS - unchanged as these are fixed in S3-CIMA file generation
_BAR_COLOR = "#8b0000"
_CSV_COLUMNS = {"cell_type", "sample_id", "cluster_id", "filter_diff", "condition_id"}
_SUMMARY_COLUMNS = ["cell_type", "mean_log2_enrichment", "std_log2_enrichment"]

# Helpers ------------------------------------------------------------------------------

def _condition_label(df: pd.DataFrame) -> str:
    """Majority condition if it covers >= 90% of selected cells, else 'mixed'."""
    if "condition_id" not in df.columns:
        return "unknown"
    top = df["condition_id"].value_counts()
    return str(top.index[0]) if top.iloc[0] / len(df) >= 0.9 else "mixed"


def _log2_enrichment_summary(
    sel_df: pd.DataFrame, bg_counts: pd.DataFrame, bg_totals: pd.Series
) -> pd.DataFrame:
    """
    Per-sample log2(selected frequency / background frequency) for every cell type,
    summarised as mean and std across samples. The background of a sample is all of
    its cells minus the selected ones. Computed as samples x cell-types matrices.
    """
    n_sel = sel_df.groupby("sample_id").size()
    sel = pd.crosstab(sel_df["sample_id"], sel_df["cell_type"]).reindex(
        index=n_sel.index, columns=bg_counts.columns, fill_value=0
    )
    n_bg = bg_totals.reindex(n_sel.index, fill_value=0) - n_sel

    keep = n_bg > 0
    if not keep.any():
        return pd.DataFrame(columns=_SUMMARY_COLUMNS)
    sel, n_sel, n_bg = sel[keep], n_sel[keep], n_bg[keep]
    bg = bg_counts.reindex(sel.index, fill_value=0)

    pseudo = 1.0 / (n_sel + n_bg)
    prop_sel = sel.div(n_sel, axis=0)
    prop_bg = (bg - sel).clip(lower=0).div(n_bg, axis=0)
    log2_fc = np.log2(prop_sel.add(pseudo, axis=0) / prop_bg.add(pseudo, axis=0))

    return pd.DataFrame({
        "cell_type": log2_fc.columns,
        "mean_log2_enrichment": log2_fc.mean().to_numpy(),
        "std_log2_enrichment": log2_fc.std().to_numpy(),  # ddof=1; NaN for one sample
    }).sort_values("mean_log2_enrichment")


def _plot_proportions(
    df: pd.DataFrame, out_dir: str, cluster_id, condition_label: str, filter_diff: float
) -> None:
    """Save a horizontal bar chart of cell type proportions among selected cells."""
    # Ascending order puts the largest proportion at the top of a barh plot.
    props = df["cell_type"].value_counts(normalize=True, ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(props) + 1.5)), dpi=150)
    bars = ax.barh(
        props.index, props.to_numpy(),
        color=_BAR_COLOR, edgecolor="white", linewidth=0.4, height=0.65,
    )
    for bar, val in zip(bars, props.to_numpy()):
        ax.text(
            bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", ha="left", fontsize=7, color="#333333",
        )
    ax.set_xlabel("Proportion of selected cells", fontsize=9)
    ax.set_title(
        f"Filter {cluster_id}  |  majority condition: {condition_label}"
        f"  |  filter_diff: {filter_diff:.4f}",
        fontsize=9, fontweight="bold", pad=10,
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(0, props.max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cell_type_proportions.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [Filter {cluster_id}] Proportion plot saved.")


def _enrichment_figure(filters: list, prefix: str) -> go.Figure:
    """One subplot per filter showing only positively enriched cell types."""
    n = len(filters)
    ncols = 2 if n == 4 else min(3, n)
    nrows = -(-n // ncols)  # ceiling division

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[
            f"Filter {cid}  |  {cond}  |  diff={fdiff:.3f}"
            for cid, fdiff, cond, _ in filters
        ],
        vertical_spacing=0.28 / nrows,
        horizontal_spacing=0.22 / ncols,
    )

    for i, (_, _, _, enriched) in enumerate(filters):
        row, col = divmod(i, ncols)
        row, col = row + 1, col + 1

        if enriched.empty:
            axis = "" if i == 0 else str(i + 1)  # make_subplots numbers axes row-major
            fig.add_annotation(
                text="No enriched cell types", showarrow=False,
                x=0.5, y=0.5, xref=f"x{axis} domain", yref=f"y{axis} domain",
                font=dict(color="#666666"),
            )
            continue

        std = enriched["std_log2_enrichment"].fillna(0)
        fig.add_trace(
            go.Bar(
                x=enriched["mean_log2_enrichment"],
                y=enriched["cell_type"],
                orientation="h",
                marker_color=_BAR_COLOR,
                error_x=dict(
                    type="data", array=std.tolist(), visible=True, thickness=1.5, width=4,
                ) if enriched["std_log2_enrichment"].notna().any() else None,
                customdata=std.round(3).to_numpy()[:, None],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "log2 enrichment: %{x:.3f}<br>"
                    "± std: %{customdata[0]:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row, col=col,
        )
        fig.update_xaxes(
            title_text="log2 enrichment",
            zeroline=True, zerolinewidth=1, zerolinecolor="rgba(0,0,0,0.3)",
            row=row, col=col,
        )
        fig.update_yaxes(
            automargin=True, categoryorder="array",
            categoryarray=enriched["cell_type"].tolist(), dtick=1,
            row=row, col=col,
        )

    fig.update_layout(
        height=max(350, 400 * nrows),
        title_text=f"Consensus Filter Enrichment Summary ({prefix})",
        title_font_size=15,
        paper_bgcolor="#f7f6f2",
        plot_bgcolor="#ffffff",
        font=dict(family="Arial, sans-serif", size=11),
    )
    return fig


# Functions ------------------------------------------------------------------

def plot_filter_weights(
    meta: list,
    save_path: str = ".",
    filename: str = "consensus_filter_weights",
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
    z = df.values                          # (n_clusters, n_genes)
    x_labels = df.columns.tolist()                # gene names
    y_labels = df.index.tolist()                  # cluster labels

    # Symmetric color scale centred at 0
    abs_max = np.abs(z).max()

    fig = go.Figure(go.Heatmap(
        z = z,
        x = x_labels,
        y = y_labels,
        colorscale  = "RdBu_r",
        zmid = 0,
        zmin = -abs_max,
        zmax =  abs_max,
        colorbar = dict(title="Weight"),
        hoverongaps = False,
        hovertemplate = "Gene: %{x}<br>Cluster: %{y}<br>Weight: %{z:.4f}<extra></extra>",
    ))

    n_genes = len(x_labels)
    n_clusters = len(y_labels)

    fig.update_layout(
        title = dict(
            text = (
                f"Filter Cluster Gene Weights "
                f"({n_clusters} clusters, {n_genes} genes)"
                f"<br><span style='font-size:14px;font-weight:normal;'>"
                f"Red = positive weight (activating) · Blue = negative (suppressing)"
                f"</span>"
            ),
            x = 0.5,
            xanchor = "center",
        ),
        xaxis = dict(
            title_text = "Gene / Marker",
            tickangle = -45,
            tickfont = dict(size=max(8, min(12, int(400 / max(n_genes, 1))))),
        ),
        yaxis = dict(
            title_text = "Filter Cluster",
            autorange = "reversed",   # cluster 0 at top
            tickfont = dict(size=11),
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

    return figure_path


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
        

def save_stats(
    high_response: dict,
    x: np.ndarray,
    y: np.ndarray,
    cell_id: np.ndarray,
    cell_type: np.ndarray,
    sample_id: np.ndarray,
    condition_id: np.ndarray,
    label_map: dict,
    save_path: str = ".",
    test: bool = True,
):
    """
    For each consensus filter cluster:
      - Per sample : spatial .png named by sample ID
      - Per filter : all_selected_cells.csv — x, y, cell_id, cell_type,
                     sample_id, label, cluster_id, filter_diff
                     for every high-response cell across all samples

    Parameters
    ----------
    high_response : dict — output of get_high_response_cells, keyed by cluster_id.
                    Must also contain a "samples" key listing valid sample IDs.
    x, y          : np.ndarray — spatial coordinates
    cell_id       : np.ndarray — unique cell identifiers
    cell_type     : np.ndarray — cell type labels
    sample_id     : np.ndarray — sample/image identifiers
    condition_id  : np.ndarray - condition id
    label_map     : dict       - map back from int to string label
    save_path     : str        — root directory for outputs
    test          : bool       — save under /test or /train subdirectory
    """

    save_path = os.path.join(save_path, "test" if test else "train")
    os.makedirs(save_path, exist_ok=True)

    valid_samples = list(high_response["samples"])
    sample_mask   = np.isin(sample_id, valid_samples)

    x = x[sample_mask]
    y = y[sample_mask]
    cell_id = cell_id[sample_mask]
    cell_type = cell_type[sample_mask]
    sample_id = sample_id[sample_mask]

    condition_id = [label_map[l] for l in condition_id]
    condition_id = np.asarray(condition_id)
    condition_id = condition_id[sample_mask]

    if cell_id.size == 0:
        raise ValueError("No cells left after filtering by sample_ids.")

    hr = {k: v for k, v in high_response.items() if k != "samples"}

    # For each filter 
    for cluster_id, res in hr.items():
        high_ids = set(res["cell_ids"])
        filter_diff = res["filter_diff"]

        cluster_dir = os.path.join(save_path, f"filter_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        all_selected_rows = []

        for sid in np.unique(sample_id):
            smask = sample_id == sid

            sx = x[smask]
            sy  = y[smask]
            s_cell_id  = cell_id[smask]
            s_cell_type = cell_type[smask]
            s_condition_id  = condition_id[smask]

            sel_mask = np.isin(s_cell_id, list(high_ids))
            n_selected = int(sel_mask.sum())
            n_total = int(smask.sum())

            if n_selected == 0:
                print(f"  [Filter {cluster_id} | Sample {sid}] "
                      f"0 selected cells — skipping.")
                continue

            # Make the spatial.png
            fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
            ax.scatter(
                sx[~sel_mask], sy[~sel_mask],
                c="#b0b0b0", s=0.25, rasterized=True
            )
            ax.scatter(
                sx[sel_mask], sy[sel_mask],
                c="#8b0000", s=0.25, rasterized=True,
                label=f"Selected (n={n_selected})"
            )
            ax.set_title(
                f"Filter {cluster_id} | Sample {sid} | "
                f"{n_selected}/{n_total} cells | "
                f"filter_diff={filter_diff:.3f}",
                fontsize=8,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(markerscale=6, fontsize=7, framealpha=0.6)
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(
                os.path.join(cluster_dir, f"spatial_plot_sample_{sid}.png")
            )
            plt.close(fig)

            # Get selected
            all_selected_rows.append(pd.DataFrame({
                "x": sx[sel_mask],
                "y": sy[sel_mask],
                "cell_id": s_cell_id[sel_mask],
                "cell_type": s_cell_type[sel_mask],
                "condition_id": s_condition_id[sel_mask],
                "sample_id": sid,
            }))

            print(f"  [Filter {cluster_id} | Sample {sid}] "
                  f"{n_selected}/{n_total} cells selected.")

        # Per-filtert csv
        if all_selected_rows:
            all_selected_df = pd.concat(all_selected_rows, ignore_index=True)
            all_selected_df["cluster_id"]  = cluster_id
            all_selected_df["filter_diff"] = round(filter_diff, 4)

            out_path = os.path.join(cluster_dir, "all_selected_cells.csv")
            all_selected_df.to_csv(out_path, index=False)

            print(f"[Filter {cluster_id}] {len(all_selected_df)} cells across "
                  f"{all_selected_df['sample_id'].nunique()} samples → {out_path}")
        else:
            print(f"[Filter {cluster_id}] No selected cells in any sample.")

    return save_path


def enrichment_summary(
    save_path: str,
    cell_type: np.ndarray,
    sample_id: np.ndarray,
    test: bool = True,
    output_file: Optional[str] = None,
) -> str:
    """
    For every filter_ folder found under save_path:
      1. Reads all_selected_cells.csv
      2. Saves a matplotlib bar chart of cell type proportions in selected cells,
         titled with the majority condition label
      3. Computes per-sample log2 enrichment (selected vs background frequency)
         using the full cell_type / sample_id arrays as background
      4. Saves a single interactive Plotly HTML with one enrichment bar chart
         per filter (mean log2 enrichment ± std across samples). Only cell types
         with a positive mean enrichment are shown; depleted ones are omitted.

    Parameters
    ----------
    save_path   : str         — root directory passed to save_high_response_stats
    cell_type   : np.ndarray  — cell type labels for ALL cells in the dataset
    sample_id   : np.ndarray  — sample IDs for ALL cells in the dataset
    test        : bool        — labels the report as test or train data
    output_file : str         — path for the HTML; defaults to
                                <save_path>/enrichment_summary_report_<prefix>.html

    Returns
    -------
    str — absolute path to the written HTML file
    """
    prefix = "test" if test else "train"
    if output_file is None:
        output_file = os.path.join(save_path, f"enrichment_summary_report_{prefix}.html")

    csv_files = sorted(
        glob.glob(os.path.join(save_path, "filter_*", "all_selected_cells.csv"))
    )
    if not csv_files:
        raise FileNotFoundError(f"No all_selected_cells.csv files found under: {save_path}")

    # Background: per-sample cell type counts over the full dataset, in one pass.
    # Labels are compared as strings because the CSVs are read back as strings.
    sample_id = pd.Series(np.asarray(sample_id).astype(str), name="sample_id")
    cell_type = pd.Series(np.asarray(cell_type).astype(str), name="cell_type")
    bg_counts = pd.crosstab(sample_id, cell_type)  # samples x cell types
    bg_totals = sample_id.value_counts()

    filters = []
    for csv_path in csv_files:
        filter_dir = os.path.dirname(csv_path)
        df = pd.read_csv(
            csv_path,
            usecols=lambda c: c in _CSV_COLUMNS,
            dtype={"sample_id": str, "cell_type": str},
        )

        cluster_id = (
            int(df["cluster_id"].iloc[0])
            if "cluster_id" in df.columns
            else os.path.basename(filter_dir).replace("filter_", "")
        )
        filter_diff = (
            round(float(df["filter_diff"].iloc[0]), 4)
            if "filter_diff" in df.columns
            else float("nan")
        )
        condition_label = _condition_label(df)

        _plot_proportions(df, filter_dir, cluster_id, condition_label, filter_diff)

        summary = _log2_enrichment_summary(df, bg_counts, bg_totals)
        enriched = summary[summary["mean_log2_enrichment"] > 0]
        filters.append((cluster_id, filter_diff, condition_label, enriched))

    fig = _enrichment_figure(filters, prefix)
    fig.show()

    output_file = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fig.write_html(output_file, include_plotlyjs="cdn")
    print(f"Enrichment summary report written → {output_file}")
    return output_file