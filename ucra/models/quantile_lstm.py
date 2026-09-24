"""Quantile LSTM - Stage 1 (Uncertainty Estimation).

One network, direct quantile heads: input window -> {horizon x n_quantiles}.
Trained with the pinball (quantile) loss so the heads estimate the full
predictive distribution: q05 ... q99. ucra/core/uncertainty.py then turns
these into u_t (point), U_t (band width) and rho_t (risk indicator).

IMPORTANT: train in z-scored space (scale targets too). Pinball gradients
are bounded (~tau), so original-scale targets in the tens of thousands make
training stall. Invert the scaler after prediction.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class QuantileLSTM(nn.Module):
    def __init__(self, seq_len: int, horizon: int, quantiles: list[float],
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.quantiles = quantiles
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, horizon * len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))          # (B, L, H)
        last = out[:, -1, :]                         # (B, H)
        q = self.head(last)                          # (B, horizon*Q)
        return q.view(-1, self.horizon, len(self.quantiles))


def make_loss(quantiles: list[float]):
    """Build the pinball loss bound to the model's quantile levels."""
    taus = torch.tensor(quantiles, dtype=torch.float32).view(1, 1, -1)

    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred (B,H,Q), y_true (B,H) -> broadcast
        e = y_true.unsqueeze(-1) - y_pred
        return torch.mean(torch.maximum(taus * e, (taus - 1.0) * e))

    return loss_fn


def train_model(model: QuantileLSTM, Xtr, Ytr, Xva, Yva, cfg: dict,
                verbose: bool = True) -> dict:
    """Standard early-stopped training. Returns history dict."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    loss_fn = make_loss(cfg["quantiles"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
                    batch_size=cfg["batch_size"], shuffle=True)
    va = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(Yva)),
                    batch_size=cfg["batch_size"])

    best_val, best_state, patience, hist = float("inf"), None, 0, {"train": [], "val": []}
    for ep in range(1, cfg["max_epochs"] + 1):
        model.train()
        tl = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            l = loss_fn(model(xb), yb)
            l.backward()
            opt.step()
            tl += float(l.detach()) * len(xb)
        tl /= max(1, len(Xtr))

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in va:
                vl += float(loss_fn(model(xb.to(dev)), yb.to(dev))) * len(xb)
        vl /= max(1, len(Xva))
        hist["train"].append(tl)
        hist["val"].append(vl)

        if verbose:
            print(f"  epoch {ep:02d}  train {tl:.4f}  val {vl:.4f}")
        if vl < best_val - 1e-5:
            best_val, patience = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["patience"]:
                if verbose:
                    print(f"  early stop @ epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


@torch.no_grad()
def predict_quantiles(model: QuantileLSTM, X: np.ndarray,
                      batch_size: int = 256) -> np.ndarray:
    """X (N, L) -> quantile predictions (N, horizon, Q) in the model's space."""
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=dev)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)
