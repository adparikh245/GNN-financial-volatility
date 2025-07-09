import itertools
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# 1) Load & pivot your three panels
df = pd.read_excel("nifty_ohlc_with_regimes.xlsx", parse_dates=["Date"])
logret_panel = df.pivot(index="Date", columns="Ticker", values="LogRet")
rv_panel     = df.pivot(index="Date", columns="Ticker", values="RV")

# Load static adjacency matrices
ete_ret = pd.read_excel("ETE_LogRet.xlsx", index_col=0)
z_ret   = pd.read_excel("Zscore_LogRet.xlsx", index_col=0)

# 2) Align only on dates (LogRet & RV)
common_dates = logret_panel.index.intersection(rv_panel.index)
logret_panel = logret_panel.loc[common_dates]
rv_panel     = rv_panel.loc[common_dates]

# 3) Regime detector: exactly as in the paper
def detect_regime(rv: np.ndarray, s: int) -> np.ndarray:
    w = max(2, 2 * s)
    stds = pd.Series(rv).rolling(w).std().fillna(0).values
    return (np.abs(rv - rv.mean()) > s * np.asarray(stds)).astype(int)

# 4) Paper's RTE/Z-score edge builder → returns edge_index & edge_weight
def build_ete_edges():
    E = ete_ret.values
    Z = z_ret.values
    mask = (E > 0) & (Z > 1.96)

    src, dst = np.where(mask)
    weights = E[mask]  # raw ETE values as edge weights

    # fallback to full graph if no edges
    if len(src) == 0:
        T = E.shape[0]
        src = np.array([i for i in range(T) for j in range(T) if i != j])
        dst = np.array([j for i in range(T) for j in range(T) if i != j])
        weights = np.ones_like(src, dtype=float)

    edge_index = np.vstack((src, dst))
    return (torch.from_numpy(edge_index).long(),
            torch.from_numpy(weights).float())

# 5) Build your Data list with sliding window, now including regime flag
def make_data(logret, rv, N, s, m):
    dates = logret.index
    data_list = []

    # build static graph once per call
    edge_index, edge_weight = build_ete_edges()

    for t in range(N - 1, len(dates) - 1):
        window_rv = rv.iloc[t - N + 1 : t + 1].values  # N×T
        x_lr      = logret.iloc[t].values              # (T,)
        x_rv      = rv.iloc[t].values
        y_next    = rv.iloc[t + 1].values

        if np.isnan(x_lr).any() or np.isnan(x_rv).any() or np.isnan(y_next).any():
            continue

        # regime per ticker, then take last flag
        regime = np.stack(
            [detect_regime(window_rv[:, i], s) for i in range(window_rv.shape[1])],
            axis=1
        )
        regime_flag = regime[-1]  # shape: (T,)

        # stack features: [LogRet, RV, RegimeFlag]
        x = np.stack([x_lr, x_rv, regime_flag], axis=1)  # [T,3]

        data_list.append(Data(
            x=torch.from_numpy(x).float(),
            y=torch.from_numpy(y_next).float(),
            edge_index=edge_index,
            edge_weight=edge_weight
        ))

    print(f"make_data: created {len(data_list)} samples")
    return data_list

# 6) Two-layer GCN from paper, now taking edge_weight
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

# 7) Training helper
def train_model(train_loader, val_loader, model, optimizer, loss_fn, epochs=50):
    train_hist, val_hist = [], []
    for epoch in range(epochs):
        model.train()
        total_train, n_train = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_weight)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_train += loss.item()
            n_train += 1
        avg_train = total_train / n_train
        train_hist.append(avg_train)

        model.eval()
        total_val, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index, batch.edge_weight)
                loss = loss_fn(out, batch.y)
                total_val += loss.item()
                n_val += 1
        avg_val = total_val / n_val
        val_hist.append(avg_val)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:>2}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")

    return train_hist, val_hist
# 8) Quick hyper-parameter sweep exactly as in paper
param_grid = {
    "lr":      [1e-3, 5e-4],
    "batch":   [8, 16],
    "hidden1": [20, 50],
    "hidden2": [20, 50],
    "N":       [125, 250, 500],
    "s":       [4, 6, 8],
    "m":       [10],  # unused currently
}

all_combos = [dict(zip(param_grid.keys(), v))
              for v in itertools.product(*param_grid.values())]

results = []
# --- Start of the loop that iterates through each hyperparameter combination ---
for p in all_combos:
    print(f"\n--- Training with params: {p} ---") # Added for clarity

    # 1. Create the dataset using the current params
    train_list = make_data(logret_panel, rv_panel, p["N"], p["s"], p["m"])
    split = int(0.8 * len(train_list))

    train_ds = train_list[:split]
    val_ds   = train_list[split:]

    # --- THE FOLLOWING LOGIC WAS MISSING ---

    # 2. Create DataLoaders for batching
    train_loader = DataLoader(train_ds, batch_size=p["batch"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=p["batch"])

    # 3. Instantiate the model with current params
    # Your features are [LogRet, RV, RegimeFlag], so in_feats=3
    model = HETE_GCN(in_feats=3, h1=p["hidden1"], h2=p["hidden2"])

    # 4. Define the loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
    loss_fn   = torch.nn.MSELoss() # Mean Squared Error is a good choice for regression

    # 5. Call the training function
    train_hist, val_hist = train_model(
        train_loader, val_loader, model, optimizer, loss_fn, epochs=50
    )

    # 6. (Optional) Store results for later analysis
    results.append({
        "params": p,
        "final_val_loss": val_hist[-1]
    })

print("\n--- Training sweep finished ---")
print("Results:", results)

plt.plot(train_hist, label='Train')
plt.plot(val_hist, label='Val')
plt.title(f"LR={p['lr']}, Hidden={p['hidden1']}")
plt.legend()
plt.show()
