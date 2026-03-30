"""Re-implementation of S3-CIMA in Pytorch.

To use the original implementation, see branch '' in repo.
"""

# Imports





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
            in_channels  = nmark,
            out_channels = nfilter,
            kernel_size  = 1,
            bias = True,
        )
        nn.init.uniform_(self.conv.weight)
        nn.init.uniform_(self.conv.bias)

        self.dropout = nn.Dropout(p=dropout_p) if dropout else None

        out_units = 1 if regression else n_classes
        self.fc = nn.Linear(nfilter, out_units)
        nn.init.uniform_(self.fc.weight)
        nn.init.uniform_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, ncell, nmark)

        Returns
        -------
        (B, n_classes) logits or (B, 1) for regression
        """
        x = x.permute(0, 2, 1)                  # (B, nmark, ncell)
        x = F.relu(self.conv(x))                 # (B, nfilter, ncell)
        x = select_top_pool(x, self.k, self.selection_type)  # (B, nfilter)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc(x)                        # (B, n_classes) or (B, 1)