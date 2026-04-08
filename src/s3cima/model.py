"""Re-implementation of S3-CIMA in Pytorch.

To use the original implementation, see branch 'legacy' in repo.
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Main model ------------------------------------------------

class CellCNN(nn.Module):
    """
    PyTorch reimplementation of CellCNN (single time-point).

    Parameters
    ----------
    nmark          : int   — number of input markers (features per cell)
    nfilter        : int   — number of convolutional filters
    k              : int   — number of top cells to pool per filter
    n_classes      : int   — output classes (ignored if regression=True)
    dropout        : bool  — whether to apply dropout after pooling
    dropout_p      : float — dropout probability
    regression     : bool  — regression (True) or classification (False)
    selection_type : str   — 'mean' or 'max' for top-k pooling
    """

    def __init__(
        self,
        nmark: int,
        nfilter: int,
        k: int,
        n_classes: int,
        dropout: bool = False,
        dropout_p: float = 0.5,
        regression: bool = False,
        selection_type: str = "mean",
    ):
        super().__init__()
        self.k = k
        self.regression = regression
        self.selection_type = selection_type

        self.conv = nn.Conv1d(
            in_channels = nmark,
            out_channels = nfilter,
            kernel_size = 1,
            bias = True,
        )
        # Different than original CellCNN - kaiming more standard
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', 
                                nonlinearity='relu')

        self.dropout = nn.Dropout(p=dropout_p) if dropout else None

        out_units = 1 if regression else n_classes
        self.fc = nn.Linear(nfilter, out_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, ncell, nmark)

        Returns
        -------
        (B, n_classes) logits or (B, 1) for regression
        """
        x = x.permute(0, 2, 1)                  
        x = F.relu(self.conv(x))                 
        x = select_top_pool(x, self.k, self.selection_type)  
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc(x)                       
    

def select_top_pool(x: torch.Tensor, k: int, selection_type: str = "mean") -> torch.Tensor:
    """
    Pools the top-k activated cells per filter.

    Parameters
    ----------
    x              : (B, nfilter, ncell) — output of Conv1d after ReLU
    k              : number of top cells to select per filter
    selection_type : 'mean' or 'max'

    Returns
    -------
    (B, nfilter) pooled representation
    """
    # topk over the cell dimension (dim=2)
    topk_vals, _ = torch.topk(x, k=k, dim=2) 
    if selection_type == "mean":
        return topk_vals.mean(dim=2)
    elif selection_type == "max":
        return topk_vals[:, :, 0]
    else:
        raise ValueError(f"selection_type must be 'mean' or 'max', got '{selection_type}'")