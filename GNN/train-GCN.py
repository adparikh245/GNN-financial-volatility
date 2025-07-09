import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# --- Model definition ---
class HETE_GNN(torch.nn.Module):
    def __init__(self, in_feats, hidden1, hidden2):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.lin   = torch.nn.Linear(hidden2, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.lin(h).squeeze(-1)

# --- Hyperparameters ---
lr         = 1e-2
batch_size = 8
hidden1    = 10
hidden2    = 10
epochs     = 50

# --- 1) Build or load your data_list as before ---
# data_list = [...]  # list of torch_geometric.data.Data

# --- 2) Train/test split ---
train_data, test_data = train_test_split(data_list, test_size=0.2, shuffle=False)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=batch_size)

# --- 3) Instantiate model, optimizer, loss ---
in_feats = data_list[0].x.size(1)
model     = HETE_GNN(in_feats, hidden1, hidden2)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn   = torch.nn.MSELoss()

# --- 4) Training loop with history ---
train_losses = []
test_losses  = []

for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch.num_graphs
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # evaluate on test set
    model.eval()
    running_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch.x, batch.edge_index)
            running_loss += loss_fn(pred, batch.y).item() * batch.num_graphs
            y_true.append(batch.y.cpu().numpy().ravel())
            y_pred.append(pred.cpu().numpy().ravel())
    test_loss = running_loss / len(test_loader.dataset)
    test_losses.append(test_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d} — train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}")

# --- 5) Final metrics ---
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
print(f"\nFinal: RMSE = {rmse:.4f}, MAE = {mae:.4f}")

# --- 6) Plot loss curves ---
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses,  label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Training vs Test Loss")
plt.show()

# --- 7) Save model ---
torch.save(model.state_dict(), "hete_gcn_best.pth")
print("Model saved to hete_gcn_best.pth")
