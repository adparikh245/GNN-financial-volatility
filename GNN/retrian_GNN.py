import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
import os

SAVE_PATH = "/Users/ananyaparikh/Documents/Coding/EquirusResearch/GNN/hete_gcn_best.pt"

# === 1) Your best hyperparameters ===
best_params = {
    "lr":      1e-3,
    "batch":   8,
    "hidden1": 50,
    "hidden2": 50,
    "N":       500,
    "s":       4,
    "m":       10
}


# === 2) Regime detector as before ===
def detect_regime(rv: pd.Series, s: int) -> torch.Tensor:
    w = max(2, 2 * s)
    stds = rv.rolling(w).std().fillna(0)
    return ((rv - rv.mean()).abs() > s * stds).astype(int)

# === 3) Build static ETE+Z edges ===
def build_ete_edges():
    ete = pd.read_excel("ETE_LogRet.xlsx", index_col=0).values
    z   = pd.read_excel("Zscore_LogRet.xlsx", index_col=0).values
    mask = (ete > 0) & (z > 1.96)
    src, dst = mask.nonzero()
    weights  = ete[mask].astype(float)
    if len(src)==0:
        T = ete.shape[0]
        src = [i for i in range(T) for j in range(T) if i!=j]
        dst = [j for i in range(T) for j in range(T) if i!=j]
        weights = [1.0]*len(src)
    edge_index  = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight

# === 4) make_data ===
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
        x_lr  = logret.iloc[t]
        x_rv  = rv.iloc[t]
        y_next= rv.iloc[t+1]
        if x_lr.isna().any() or x_rv.isna().any() or y_next.isna().any():
            continue
        regime_flags = []
        for col in window_rv:
            regime_flags.append(detect_regime(window_rv[col], s).iloc[-1])
        x = torch.stack([
            torch.from_numpy(x_lr.values).float(),
            torch.from_numpy(x_rv.values).float(),
            torch.tensor(regime_flags, dtype=torch.float)
        ], dim=1)
        y = torch.from_numpy(y_next.values).float()
        data_list.append(Data(x=x,
                              y=y,
                              edge_index=edge_index,
                              edge_weight=edge_weight))
    print(f"make_data: created {len(data_list)} samples")
    return data_list

# === 5) GCN model ===
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

# === 6) Training loop ===
def train_model(train_loader, val_loader, model, optimizer, loss_fn, epochs=100):
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for b in train_loader:
            optimizer.zero_grad()
            out = model(b.x, b.edge_index, b.edge_weight)
            loss= loss_fn(out, b.y)
            loss.backward()
            optimizer.step()
            total += loss.item(); n+=1
        train_loss = total/n

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for b in val_loader:
                out = model(b.x, b.edge_index, b.edge_weight)
                total += loss_fn(out, b.y).item()
                n += 1
        val_loss = total/n

        if epoch%10==0 or epoch==epochs-1:
            print(f"Epoch {epoch:>3}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

# === 7) Main ===
def main():
    # load + pivot
    df = pd.read_excel("nifty_ohlc_with_regimes.xlsx", parse_dates=["Date"])
    lr = df.pivot(index="Date", columns="Ticker", values="LogRet")
    rv = df.pivot(index="Date", columns="Ticker", values="RV")

    data = make_data(lr, rv,
                     best_params["N"],
                     best_params["s"],
                     best_params["m"])

    split = int(0.9 * len(data))
    train_ds, test_ds = data[:split], data[split:]
    train_loader = DataLoader(train_ds,
                              batch_size=best_params["batch"],
                              shuffle=True)
    test_loader  = DataLoader(test_ds,
                              batch_size=best_params["batch"])

    model     = HETE_GCN(in_feats=3,
                         h1=best_params["hidden1"],
                         h2=best_params["hidden2"])
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=best_params["lr"])
    loss_fn   = torch.nn.MSELoss()

    print(f"\n*** Training with {best_params} ***\n")
    train_model(train_loader, test_loader,
                model, optimizer, loss_fn,
                epochs=100)

    # final test
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in test_loader:
            p = model(b.x, b.edge_index, b.edge_weight)
            tot += loss_fn(p, b.y).item(); n+=1
    print(f"\nFinal test loss: {tot/n:.4f}\n")

    # plot first 10
    batch = next(iter(test_loader))
    with torch.no_grad():
        preds = model(batch.x, batch.edge_index, batch.edge_weight)
    plt.figure(figsize=(8,4))
    plt.plot(preds.numpy()[:10], label="Pred")
    plt.plot(batch.y.numpy()[:10],   label="True")
    plt.legend(); plt.show()

    print("Saving model...")
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
