import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from pre_GNN_H_ETE import compute_transfer_entropy, compute_rte_zscores

# ─── 1) HURST EXPONENT (R/S ANALYSIS) ───────────────────────────────────────────────

def compute_hurst(ts: np.ndarray) -> float:
    # R/S analysis: split into segments at multiple scales, fit slope of log(R/S) vs log(w)
    scales = np.array([10, 20, 50, 100, 250])  # example scales
    rs = []
    for w in scales:
        if len(ts) < w: break
        chunks = len(ts) // w
        segs = ts[:chunks*w].reshape(chunks, w)
        RSi = []
        for seg in segs:
            x = seg - seg.mean()
            Y = np.cumsum(x)
            R = Y.max() - Y.min()
            S = seg.std()
            if S > 0:
                RSi.append(R/S)
        rs.append(np.mean(RSi))
    # linear fit in log–log
    logs = np.log(rs)
    logw = np.log(scales[:len(rs)])
    H, _ = np.polyfit(logw, logs, 1)
    return H

def detect_hurst_regimes(world_returns: pd.Series,
                         window: int, sensitivity: int) -> pd.Series:
    # rolling Hurst exponent
    Hs = world_returns.rolling(window).apply(compute_hurst, raw=True)
    # classify: count last `window` days above 0.5
    flags = (Hs > 0.5).rolling(window).sum() >= sensitivity
    return flags.astype(int)

# ─── 2) DYNAMIC ETE GRAPH PER WINDOW ────────────────────────────────────────────

def build_ete_edges_dynamic(window_lr: pd.DataFrame, m: int, z_thresh=1.96):
    # recompute TE & Z per window
    T = window_lr.shape[1]
    te_mat = np.zeros((T, T), dtype=np.float32)
    for i in range(T):
        for j in range(T):
            if i != j:
                te_mat[i, j] = compute_transfer_entropy(
                    window_lr.iloc[:, i].values,
                    window_lr.iloc[:, j].values,
                    m
                )
    z_mat = compute_rte_zscores(te_mat)              # same shape
    mask = (te_mat > 0) & (z_mat > z_thresh)
    src, dst = np.where(mask)
    weights = te_mat[mask].astype(np.float32)
    if len(src)==0:
        # fallback star graph
        T = te_mat.shape[0]
        src = np.repeat(np.arange(1,T), 1)
        dst = np.zeros_like(src)
        weights = np.ones_like(src, dtype=np.float32)
    edge_index  = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight

# ─── 3) MULTI-SCALE 1D-CONV FEATURE EXTRACTOR ─────────────────────────────────

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels=1, channels=12, kernels=(3,5,7)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, channels, k, padding=k//2)
            for k in kernels
        ])
    def forward(self, x):
        # x: [num_nodes, window_len]
        x = x.unsqueeze(1)  # → [num_nodes, 1, window_len]
        feats = []
        for conv in self.convs:
            h = F.relu(conv(x))         # [num_nodes, channels, window_len]
            h = h.mean(dim=2)           # global pooling → [num_nodes, channels]
            feats.append(h)
        return torch.cat(feats, dim=1)  # [num_nodes, channels * len(kernels)]

# ─── 4) H-ETE-GCN MODEL ─────────────────────────────────────────────────────────

class HETE_GNN(nn.Module):
    def __init__(self, in_feats, h1, h2):
        super().__init__()
        self.conv1 = GCNConv(in_feats, h1)
        self.conv2 = GCNConv(h1,      h2)
        self.lin   = nn.Linear(h2, 1)
    def forward(self, x, edge_index, edge_weight):
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        return self.lin(h).squeeze(-1)

# ─── 5) DATASET CONSTRUCTION ─────────────────────────────────────────────────────

def make_dynamic_dataset(logret: pd.DataFrame,
                         rv:     pd.DataFrame,
                         world_r: pd.Series,
                         params: dict):
    N, m, s = params["N"], params["m"], params["s"]
    # Precompute world-regimes
    regimes = detect_hurst_regimes(world_r, window=N, sensitivity=s)
    data_list = []
    msconv = MultiScaleConv()

    for t in range(N-1, len(logret)-1):
        # 1) get window and regime flag
        window_lr = logret.iloc[t-N+1:t+1]
        if regimes.iloc[t] != regimes.iloc[t-1]:
            # regime changed → recompute graph
            edge_index, edge_weight = build_ete_edges_dynamic(window_lr, m)
        # 2) node features via multi-scale conv
        #    apply to each column’s window of returns
        arr = window_lr.values.T  # shape [num_nodes, N]
        x_feats = msconv(torch.from_numpy(arr).float())
        # 3) target
        y_next = torch.from_numpy(rv.iloc[t+1].values).float()
        data_list.append(Data(x=x_feats,
                              y=y_next,
                              edge_index=edge_index,
                              edge_weight=edge_weight))
    print(f"Created {len(data_list)} samples")
    return data_list

# ─── 6) TRAIN / EVAL LOOP ─────────────────────────────────────────────────────────

def train_and_eval(data, params):
    split = int(0.8 * len(data))
    train_ds, test_ds = data[:split], data[split:]
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=params["batch"], shuffle=shuffle)
    train_ld, test_ld = loader(train_ds, True), loader(test_ds, False)

    model = HETE_GNN(in_feats=12*3, h1=params["hidden1"], h2=params["hidden2"])
    opt   = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()

    # Training
    for epoch in range(1, 51):
        model.train()
        tl = np.mean([loss_fn(model(b.x, b.edge_index, b.edge_weight), b.y).item()
                      for b in train_ld])
        model.eval()
        vl = np.mean([loss_fn(model(b.x, b.edge_index, b.edge_weight), b.y).item()
                      for b in test_ld])
        if epoch%10==0:
            print(f"Epoch {epoch:>2} → train {tl:.4f}, val {vl:.4f}")
    return model

# ─── 7) MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # load your data
    df = pd.read_excel("nifty_ohlc_with_regimes.xlsx", parse_dates=["Date"])
    lr = df.pivot(index="Date", columns="Ticker", values="LogRet")
    rv = df.pivot(index="Date", columns="Ticker", values="RV")
    world_r = pd.read_excel("WORLD_LogRet.xlsx", index_col=0).iloc[:,0]  # your MSCI World series

    params = {
      "lr":      1e-3,
      "batch":   8,
      "hidden1": 20,
      "hidden2": 15,
      "N":       125,
      "s":       6,
      "m":       10
    }

    data = make_dynamic_dataset(lr, rv, world_r, params)
    model = train_and_eval(data, params)
    torch.save(model.state_dict(), "hete_gnn_full_paper.pt")
    print("✅ All done — full H-ETE-GNN replication complete!")
