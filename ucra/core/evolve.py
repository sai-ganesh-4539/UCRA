"""Stage 5 - Self-Evolving Learning.

DriftMonitor watches the feedback (violation rate, demand z-score). When a
trigger fires, the EvolutionEngine fine-tunes theta_t -> theta_{t+1} on a
reservoir-sampled replay buffer of recent windows (continual learning
without catastrophic forgetting).
"""
from __future__ import annotations

import random

import numpy as np
import torch


class ReplayBuffer:
    """Reservoir sampling over (window, target) pairs."""

    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.X: list[np.ndarray] = []
        self.Y: list[np.ndarray] = []
        self._n_seen = 0

    def push(self, x: np.ndarray, y: np.ndarray):
        self._n_seen += 1
        if len(self.X) < self.capacity:
            self.X.append(x)
            self.Y.append(y)
        else:
            j = random.randint(0, self._n_seen - 1)
            if j < self.capacity:
                self.X[j] = x
                self.Y[j] = y

    def arrays(self):
        if not self.X:
            return None, None
        return np.stack(self.X), np.stack(self.Y)


class DriftMonitor:
    def __init__(self, cfg: dict):
        self.violation_trigger = cfg["violation_trigger"]
        self.drift_zscore = cfg["drift_zscore"]
        self.window = cfg.get("drift_window", 96)
        self.violations: list[int] = []
        self.mu: float | None = None
        self.sd: float = 1.0

    def set_reference(self, train_demand: np.ndarray):
        self.mu = float(np.mean(train_demand))
        self.sd = float(np.std(train_demand) + 1e-8)

    def observe(self, violated: bool, demand: float) -> dict:
        self.violations.append(int(violated))
        if len(self.violations) > self.window:
            self.violations.pop(0)
        viol_rate = float(np.mean(self.violations))
        z = abs(demand - self.mu) / self.sd if self.mu is not None else 0.0
        return {"viol_rate": viol_rate, "z": z}

    def triggered(self, stats: dict) -> bool:
        return (stats["viol_rate"] > self.violation_trigger
                or stats["z"] > self.drift_zscore)


class EvolutionEngine:
    def __init__(self, model, cfg: dict, train_cfg: dict):
        self.model = model
        self.cfg = cfg                 # evolution section
        self.train_cfg = train_cfg     # model section (lr etc.)
        self.buffer = ReplayBuffer(cfg["replay_size"])
        self.n_updates = 0

    def remember(self, x: np.ndarray, y: np.ndarray):
        self.buffer.push(x, y)

    def finetune(self, loss_fn) -> dict:
        """theta_t -> theta_{t+1} on the replay buffer (few epochs)."""
        X, Y = self.buffer.arrays()
        if X is None:
            return {"updated": False}
        dev = next(self.model.parameters()).device
        opt = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg["lr"] * 0.3)
        xt = torch.tensor(X, dtype=torch.float32, device=dev)
        yt = torch.tensor(Y, dtype=torch.float32, device=dev)
        self.model.train()
        for _ in range(self.cfg["finetune_epochs"]):
            opt.zero_grad()
            loss = loss_fn(self.model(xt), yt)
            loss.backward()
            opt.step()
        self.n_updates += 1
        self.model.eval()
        return {"updated": True, "replay": len(X),
                "loss": float(loss), "n_updates": self.n_updates}
