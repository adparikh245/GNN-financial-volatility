import itertools
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from pre_GNN_H_ETE import compute_transfer_entropy  # your TE function

# --------------------------------------------------
# 1) Utility functions
# --------------------------------------------------

def detect_regime(rv: pd.Series, s: int) -> pd.Series:
    w = max(2, 2 * s)
    stds = rv.rolling(w).std().fillna(0)
    return ((rv - rv.mean()).abs() > s * stds).astype(int)

def build_ete_edges(window_lr: pd.DataFrame, m: int, z_thresh: float = 1.96):
    """
    Compute dynamic ETE+Z for this window of log‐returns.
    Returns edge_index (2×E) and edge_weight (E).
    """
    T = window_lr.shape[1]
    E = np.zeros((T, T), float)

    # 1) compute pairwise TE
    for i in range(T):
        for j in range(T):
            if i != j:
                E[i, j] = compute_transfer_entropy(
                    window_lr.iloc[:, i].values,
                    window_lr.iloc[:, j].values,
                    m
                )
        if i % 10 == 0:
            print(f"    · TE row {i+1}/{T}")

    # 2) z‐score threshold
    nonzero = E[E > 0]
    if len(nonzero) > 1:
        μ, σ = nonzero.mean(), nonzero.std(ddof=1)
        Z = (E - μ) / σ
    else:
        Z = np.zeros_like(E)

    mask = (E > 0) & (Z > z_thresh)
    src, dst = np.where(mask)
    weights = E[mask]

    # 3) fallback to full clique if nothing passed
    if len(src) == 0:
        src = np.repeat(np.arange(T), T-1)
        dst = np.concatenate([np.delete(np.arange(T), i) for i in range(T)])
        weights = np.ones_like(src, float)

    edge_index  = torch.from_numpy(np.vstack((src, dst))).long()
    edge_weight = torch.from_numpy(weights).float()
    return edge_index, edge_weight

def make_data(logret: pd.DataFrame,
              rv:     pd.DataFrame,
              N:      int,
              s:      int,
              m:      int):
    """
    Build a list of Data(x, y, edge_index, edge_weight), one per time t,
    with dynamic ETE graphs on the past N days.
    """
    dates = logret.index
    data_list = []
    total = len(dates) - N

    print(f"→ Building {total} graphs (N={N}, s={s}, m={m})…")
    for t in range(N-1, len(dates)-1):
        if (t - (N-1)) % 10 == 0:
            print(f"  Window {t-(N-1)+1}/{total}")

        # slice windows
        win_lr = logret.iloc[t-N+1:t+1]
        win_rv = rv.iloc[t-N+1:t+1].values

        # dynamic graph
        edge_index, edge_weight = build_ete_edges(win_lr, m)

        # features at time t, target at t+1
        x_lr, x_rv = logret.iloc[t].values, rv.iloc[t].values
        y_next     = rv.iloc[t+1].values
        if np.isnan(x_lr).any() or np.isnan(x_rv).any() or np.isnan(y_next).any():
            continue

        regime_flags = []
        for col in range(win_rv.shape[1]):
            regime_flags.append(detect_regime(pd.Series(win_rv[:, col]), s).iloc[-1])

        x = np.stack([x_lr, x_rv, regime_flags], axis=1)
        data_list.append(Data(
            x=torch.from_numpy(x).float(),
            y=torch.from_numpy(y_next).float(),
            edge_index=edge_index,
            edge_weight=edge_weight
        ))

    print(f"← Created {len(data_list)} samples\n")
    return data_list

# --------------------------------------------------
# 2) GCN & training loop
# --------------------------------------------------

class HETE_GCN(torch.nn.Module):
    def __init__(self, in_feats: int, h1: int, h2: int):
        super().__init__()
        self.conv1 = GCNConv(in_feats, h1)
        self.conv2 = GCNConv(h1,      h2)
        self.lin   = torch.nn.Linear(h2, 1)

    def forward(self, x, edge_index, edge_weight):
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        return self.lin(h).squeeze(-1)

def train_model(train_loader, val_loader, model, optimizer, loss_fn, epochs=50):
    train_hist, val_hist = [], []
    for epoch in range(epochs):
        # train
        model.train()
        tot, n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_weight)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            tot += loss.item(); n += 1
        train_hist.append(tot/n)

        # val
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index, batch.edge_weight)
                tot += loss_fn(out, batch.y).item()
                n += 1
        val_hist.append(tot/n)

        if epoch % 10 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch:>3}: train={train_hist[-1]:.4f}  val={val_hist[-1]:.4f}")

    return train_hist, val_hist

# --------------------------------------------------
# 3) Main: sweep then final retrain
# --------------------------------------------------

if __name__ == "__main__":
    # load data once
    df = pd.read_excel("nifty_ohlc_with_regimes.xlsx", parse_dates=["Date"])
    lr = df.pivot(index="Date", columns="Ticker", values="LogRet")
    rv = df.pivot(index="Date", columns="Ticker", values="RV")

    # hyper‐parameter grid
    param_grid = {
        "lr":      [1e-3, 5e-4],
        "batch":   [8, 16],
        "hidden1": [20, 50],
        "hidden2": [20, 50],
        "N":       [125, 250, 500],
        "s":       [4, 6, 8],
        "m":       [5, 10]          # m=history for TE
    }
    combos = [dict(zip(param_grid.keys(), v))
              for v in itertools.product(*param_grid.values())]

    # Phase 1: hyper‐parameter sweep (80/20)
    results = []
    print("=== Phase 1: hyper‐parameter sweep ===")
    for p in combos:
        print(f"\n-- Testing {p}")
        data80 = make_data(lr, rv, p["N"], p["s"], p["m"])
        split = int(0.8 * len(data80))
        train_ds, val_ds = data80[:split], data80[split:]
        trn_loader = DataLoader(train_ds, batch_size=p["batch"], shuffle=True)
        val_loader = DataLoader(val_ds,   batch_size=p["batch"])
        model = HETE_GCN(in_feats=3, h1=p["hidden1"], h2=p["hidden2"])
        opt   = torch.optim.Adam(model.parameters(), lr=p["lr"])
        loss_fn = torch.nn.MSELoss()
        _, val_hist = train_model(trn_loader, val_loader, model, opt, loss_fn, epochs=50)
        results.append({"params":p, "val_loss":val_hist[-1]})

    # pick best
    best = min(results, key=lambda x: x["val_loss"])["params"]
    print("\n>>> Best hyper‐parameters:", best)

    # Phase 2: final retrain (90/10)
    print("\n=== Phase 2: final retrain with best params ===")
    data90 = make_data(lr, rv, best["N"], best["s"], best["m"])
    split = int(0.9 * len(data90))
    tr_ds, te_ds = data90[:split], data90[split:]
    trn_loader = DataLoader(tr_ds, batch_size=best["batch"], shuffle=True)
    tst_loader = DataLoader(te_ds, batch_size=best["batch"])
    model_fin = HETE_GCN(in_feats=3, h1=best["hidden1"], h2=best["hidden2"])
    opt_fin   = torch.optim.Adam(model_fin.parameters(), lr=best["lr"])
    loss_fn   = torch.nn.MSELoss()
    hist_tr, hist_te = train_model(trn_loader, tst_loader, model_fin, opt_fin, loss_fn, epochs=100)

    print(f"\nFinal test loss: {hist_te[-1]:.4f}")

    # save
    SAVE_PATH = "hete_gcn_best.pt"
    torch.save(model_fin.state_dict(), SAVE_PATH)
    print(f"✅ Saved final model to {SAVE_PATH}")

    # optional: plot final train vs test loss
    plt.plot(hist_tr, label="Train")
    plt.plot(hist_te, label="Test")
    plt.title("Final retrain loss curves")
    plt.legend()
    plt.show()