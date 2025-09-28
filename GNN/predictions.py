import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# === 0) Point this at the real location of your saved model ===
MODEL_PATH = "../models/hete_gcn_best.pt"

# === 1) Best hyperparams (only for batching / data prep) ===
best_params = {
    "lr":      1e-3,
    "batch":   8,
    "hidden1": 20,
    "hidden2": 15,
    "N":       125,
    "s":       6,
    "m":       10
}

# === 2) Regime detector & edge builder ===
def detect_regime(rv: pd.Series, s: int) -> np.ndarray:
    w = max(2, 2 * s)
    stds = rv.rolling(w).std().fillna(0)
    return ((rv - rv.mean()).abs() > s * stds).astype(int).values

def build_ete_edges():
    E = pd.read_excel("../data/ETE_LogRet.xlsx", index_col=0).values
    Z = pd.read_excel("../data/Zscore_LogRet.xlsx", index_col=0).values
    mask = (E > 0) & (Z > 1.96)

    src, dst = np.where(mask)
    weights  = E[mask].astype(float)

    # fallback to full graph if no edges
    if len(src) == 0:
        T = E.shape[0]
        src, dst = np.broadcast_arrays(
            np.repeat(np.arange(T), T-1),
            np.concatenate([np.delete(np.arange(T), i) for i in range(T)])
        )
        weights = np.ones_like(src, dtype=float)

    edge_index  = torch.from_numpy(np.vstack((src, dst))).long()
    edge_weight = torch.from_numpy(weights).float()
    return edge_index, edge_weight

# === 3) make_data exactly as in training ===
def make_data(logret: pd.DataFrame,
              rv:     pd.DataFrame,
              N:      int,
              s:      int,
              m:      int):
    dates = logret.index
    edge_index, edge_weight = build_ete_edges()
    data_list = []
    for t in range(N-1, len(dates)-1):
        window_rv = rv.iloc[t-N+1:t+1]
        x_lr  = logret.iloc[t].values
        x_rv  = rv.iloc[t].values
        y_next= rv.iloc[t+1].values
        if np.isnan(x_lr).any() or np.isnan(x_rv).any() or np.isnan(y_next).any():
            continue
        regime_flags = np.array([ detect_regime(window_rv[col], s)[-1]
                                  for col in window_rv.columns ])
        x = torch.from_numpy(
            np.stack([x_lr, x_rv, regime_flags], axis=1)
        ).float()
        y = torch.from_numpy(y_next).float()
        data_list.append(Data(x=x,
                              y=y,
                              edge_index=edge_index,
                              edge_weight=edge_weight))
    print(f"make_data: created {len(data_list)} samples")
    return data_list

# === 4) GCN model ===
class HETE_GCN(torch.nn.Module):
    def __init__(self, in_feats, h1, h2):
        super().__init__()
        self.conv1 = GCNConv(in_feats, h1)
        self.conv2 = GCNConv(h1,      h2)
        self.lin   = torch.nn.Linear(h2, 1)
    def forward(self, x, edge_index, edge_weight):
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        return self.lin(h).squeeze(-1)

# === 5) Inference only ===
def main():
    # load + pivot
    df = pd.read_excel("../data/nifty_ohlc_with_regimes.xlsx", parse_dates=["Date"])
    lr = df.pivot(index="Date", columns="Ticker", values="LogRet")
    rv = df.pivot(index="Date", columns="Ticker", values="RV")

    # rebuild dataset & take last 10% as test
    all_data  = make_data(lr, rv,
                          best_params["N"],
                          best_params["s"],
                          best_params["m"])
    split     = int(0.9 * len(all_data))
    test_list = all_data[split:]

    test_loader = DataLoader(test_list, batch_size=1)

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = HETE_GCN(in_feats=3,
                      h1=best_params["hidden1"],
                      h2=best_params["hidden2"]).to(device)
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Could not find model at {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # collect preds & trues
    preds, trues = [], []
    with torch.no_grad():
        for b in test_loader:
            b = b.to(device)
            out = model(b.x, b.edge_index, b.edge_weight)
            print("out shape:", out.shape)
            preds.append(out.cpu().numpy())
            trues.append(b.y.cpu().numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)

    tickers     = rv.columns.tolist()
    dates       = rv.index
    y_dates     = dates[best_params["N"]:]      # one y per Data sample
    test_dates  = y_dates[split:]               # align with your test_list

    # sanity check
    print("preds.shape:", preds.shape, "trues.shape:", trues.shape)
    print("len(test_dates):", len(test_dates))

    # build DataFrames
    preds_df = pd.DataFrame(preds, index=test_dates, columns=pd.Index(tickers))
    trues_df = pd.DataFrame(trues, index=test_dates, columns=pd.Index(tickers))

    # --- new: save to CSV ---
    preds_df.to_csv(
        "../outputs/predictions/out_of_sample_predictions.csv",
        float_format="%.4f",
        date_format="%Y-%m-%d",
        index_label="Date"
    )
    trues_df.to_csv(
        "../outputs/predictions/out_of_sample_actuals.csv",
        float_format="%.4f",
        date_format="%Y-%m-%d",
        index_label="Date"
    )
    print("✅ Saved CSVs: outputs/predictions/out_of_sample_predictions.csv, outputs/predictions/out_of_sample_actuals.csv")
    # --- end new ---

    # Create results directory if it doesn't exist
    results_dir = "../graphs"
    os.makedirs(results_dir, exist_ok=True)
    
    # loop over tickers and both show & save each figure
    for ticker in tickers:
        fig, ax = plt.subplots(figsize=(10,4))
        trues_df[ticker].plot(ax=ax, label="True RV")
        preds_df[ticker].plot(ax=ax, label="Pred RV", linestyle="--")
        ax.set_title(f"Out-of-Sample RV for {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("RV")
        ax.legend()
        fig.tight_layout()

        # show interactive window (if your backend supports it)
        plt.show()
        
        # and also save to PNG in the results directory
        outfn = os.path.join(results_dir, f"oos_{ticker}.png")
        fig.savefig(outfn, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {outfn}")
        plt.close(fig)

if __name__ == "__main__":
    main()