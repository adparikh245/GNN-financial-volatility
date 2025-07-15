#!/usr/bin/env python3
import time
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from joblib import Parallel, delayed
from tqdm import tqdm
from pre_GNN_H_ETE import compute_transfer_entropy  # your TE function
from numba import jit, prange
import multiprocessing as mp

# 4 hours
# --------------------------------------------------
# 1) Optimized utility functions
# --------------------------------------------------

@jit(nopython=True)
def detect_regime_fast(rv_values, s):
    """Numba-optimized regime detection"""
    n = len(rv_values)
    w = max(2, 2 * s)
    result = np.zeros(n, dtype=np.int32)
    
    if n < w:
        return result
    
    rv_mean = np.mean(rv_values)
    
    for i in range(w-1, n):
        # Rolling std calculation
        window = rv_values[i-w+1:i+1]
        window_std = np.std(window)
        
        if window_std > 0:
            threshold = s * window_std
            if abs(rv_values[i] - rv_mean) > threshold:
                result[i] = 1
    
    return result

def detect_regime(rv: pd.Series, s: int) -> pd.Series:
    """Optimized regime detection using numba"""
    values = rv.values
    result = detect_regime_fast(values, s)
    return pd.Series(result, index=rv.index)

def build_ete_edges_vectorized(window_lr: pd.DataFrame,
                              m: int,
                              z_thresh: float = 1.96,
                              verbose: bool = False,
                              window_idx: int = None):
    """
    Vectorized version of ETE computation with better memory management
    """
    start_time = time.time()
    T = window_lr.shape[1]
    data_matrix = window_lr.values.T  # Shape: (T, time_steps)
    
    if verbose:
        print(f"      Window {window_idx}: Processing {T} assets, {window_lr.shape[0]} time steps")
    
    # Pre-allocate edge storage
    edge_list = []
    weight_list = []
    
    # Vectorized TE computation using chunking to avoid memory issues
    chunk_size = min(50, T)  # Process in chunks
    
    E = np.zeros((T, T), dtype=np.float32)  # Use float32 for memory efficiency
    
    te_start = time.time()
    total_pairs = T * (T - 1)
    pairs_processed = 0
    
    for i_start in range(0, T, chunk_size):
        i_end = min(i_start + chunk_size, T)
        
        # Process chunk of source nodes
        for i in range(i_start, i_end):
            for j in range(T):
                if i != j:
                    E[i, j] = compute_transfer_entropy(
                        data_matrix[i], data_matrix[j], m
                    )
                    pairs_processed += 1
            
            if verbose and i % 10 == 0:
                elapsed = time.time() - te_start
                pairs_per_sec = pairs_processed / elapsed if elapsed > 0 else 0
                print(f"        · TE row {i+1}/{T}, {pairs_processed}/{total_pairs} pairs, {pairs_per_sec:.1f} pairs/sec")
    
    te_elapsed = time.time() - te_start
    if verbose:
        print(f"      Window {window_idx}: TE computation took {te_elapsed:.2f}s")
    
    # Vectorized z-score computation
    zscore_start = time.time()
    nonzero_mask = E > 0
    if np.any(nonzero_mask):
        nonzero = E[nonzero_mask]
        if len(nonzero) > 1:
            mu, sigma = nonzero.mean(), nonzero.std(ddof=1)
            Z = np.divide(E - mu, sigma, out=np.zeros_like(E), where=sigma != 0)
        else:
            Z = np.zeros_like(E)
    else:
        Z = np.zeros_like(E)
    
    # Find significant edges
    mask = nonzero_mask & (Z > z_thresh)
    src, dst = np.where(mask)
    weights = E[mask]
    
    # Fallback full clique if no significant edges
    if len(src) == 0:
        src = np.repeat(np.arange(T), T-1)
        dst = np.concatenate([np.delete(np.arange(T), i) for i in range(T)])
        weights = np.ones_like(src, dtype=np.float32)
    
    edge_index = torch.from_numpy(np.vstack((src, dst))).long()
    edge_weight = torch.from_numpy(weights).float()
    
    total_elapsed = time.time() - start_time
    if verbose:
        print(f"      Window {window_idx}: Total time {total_elapsed:.2f}s, {len(src)} edges created")
    
    return edge_index, edge_weight

# --------------------------------------------------
# 2) Optimized parallel precomputation
# --------------------------------------------------

def precompute_edge_graphs_optimized(lr: pd.DataFrame,
                                   Ns: list,
                                   ms: list,
                                   z_thresh: float = 1.96,
                                   n_jobs: int = -1,
                                   verbose: bool = True):
    """
    Optimized precomputation with better memory management and chunking
    """
    dates = lr.index
    ete_cache = {}
    
    print(f"Data shape: {lr.shape} (dates: {len(dates)}, assets: {lr.shape[1]})")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Convert to numpy once for all computations
    lr_values = lr.values
    print(f"Memory usage of lr_values: {lr_values.nbytes / 1024 / 1024:.1f} MB")
    
    for N in Ns:
        total_windows = len(dates) - N
        print(f"\n=== Processing N={N} (window size), {total_windows} windows ===")
        
        for m in ms:
            print(f"\nProcessing m={m} (TE lag parameter)")
            print(f"Will compute {total_windows} windows, each with {lr.shape[1]} assets")
            print(f"Each window: {lr.shape[1]} × {lr.shape[1]} = {lr.shape[1]**2} TE computations")
            print(f"Total TE computations: {total_windows * lr.shape[1]**2:,}")
            
            start_time = time.time()
            
            # Create windows more efficiently
            print("Creating window data...")
            window_creation_start = time.time()
            windows = []
            for t in range(N-1, len(dates)-1):
                window_data = lr.iloc[t-N+1:t+1]
                windows.append((window_data, m, z_thresh, t-(N-1)))
            
            window_creation_time = time.time() - window_creation_start
            print(f"Window creation took {window_creation_time:.2f}s")
            
            # Determine actual number of jobs
            if n_jobs == -1:
                actual_jobs = mp.cpu_count()
            else:
                actual_jobs = min(n_jobs, mp.cpu_count())
            
            print(f"Starting parallel processing with {actual_jobs} jobs...")
            
            # Process first few windows with verbose output to see progress
            if verbose and len(windows) > 0:
                print("Processing first window with verbose output...")
                sample_result = build_ete_edges_vectorized(
                    windows[0][0], windows[0][1], windows[0][2], True, 0
                )
                print(f"Sample result: {sample_result[0].shape[1]} edges")
                
                # Estimate time for all windows
                window_time = time.time() - start_time
                estimated_total = window_time * len(windows) / actual_jobs
                print(f"Estimated time for all windows: {estimated_total/60:.1f} minutes")
            
            # Process in parallel with progress tracking
            print(f"Processing {len(windows)} windows in parallel...")
            parallel_start = time.time()
            
            # Use a progress callback approach
            def process_window_with_progress(args):
                window_data, m, z_thresh, idx = args
                if idx % 100 == 0:
                    print(f"  Processing window {idx+1}/{len(windows)}")
                return build_ete_edges_vectorized(window_data, m, z_thresh, False, idx)
            
            edges = Parallel(n_jobs=actual_jobs, backend='threading')(
                delayed(process_window_with_progress)(window_args)
                for window_args in windows
            )
            
            parallel_time = time.time() - parallel_start
            total_time = time.time() - start_time
            
            print(f"Parallel processing took {parallel_time/60:.1f} min")
            print(f"Total time for N={N}, m={m}: {total_time/60:.1f} min")
            print(f"Average time per window: {total_time/len(windows):.3f}s")
            
            # Memory usage check
            edges_memory = sum(edge[0].numel() + edge[1].numel() for edge in edges) * 4 / 1024 / 1024
            print(f"Memory usage for edges: {edges_memory:.1f} MB")
            
            ete_cache[(N, m)] = edges
    
    total_memory = sum(
        sum(edge[0].numel() + edge[1].numel() for edge in edges) * 4 
        for edges in ete_cache.values()
    ) / 1024 / 1024
    print(f"\nTotal cache memory usage: {total_memory:.1f} MB")
    
    return ete_cache

# --------------------------------------------------
# 3) Optimized dataset builder with caching
# --------------------------------------------------

def make_data_cached_optimized(lr: pd.DataFrame,
                              rv: pd.DataFrame,
                              N: int,
                              s: int,
                              m: int,
                              edges: list):
    """
    Optimized dataset creation with vectorized operations
    """
    dates = lr.index
    data_list = []
    
    # Pre-convert to numpy for faster access
    lr_values = lr.values
    rv_values = rv.values
    
    # Pre-compute all regime flags for all windows
    regime_cache = {}
    
    for idx, t in enumerate(range(N-1, len(dates)-1)):
        edge_index, edge_weight = edges[idx]
        
        # Use numpy indexing for speed
        x_lr = lr_values[t]
        x_rv = rv_values[t]
        y_next = rv_values[t+1]
        
        # Skip any NaNs
        if np.isnan(x_lr).any() or np.isnan(x_rv).any() or np.isnan(y_next).any():
            continue
        
        # Optimized regime detection
        win_rv = rv_values[t-N+1:t+1]
        regime_flags = np.zeros(win_rv.shape[1], dtype=np.float32)
        
        for col in range(win_rv.shape[1]):
            col_series = pd.Series(win_rv[:, col])
            regime_result = detect_regime(col_series, s)
            regime_flags[col] = regime_result.iloc[-1]
        
        # Stack features more efficiently
        x = torch.from_numpy(
            np.column_stack([x_lr, x_rv, regime_flags])
        ).float()
        y = torch.from_numpy(y_next).float()
        
        data_list.append(Data(x=x,
                              y=y,
                              edge_index=edge_index,
                              edge_weight=edge_weight))
    
    print(f"Created {len(data_list)} samples [N={N}, m={m}, s={s}]")
    return data_list

# --------------------------------------------------
# 4) Optimized model with dropout and better architecture
# --------------------------------------------------

class HETE_GCN_Optimized(torch.nn.Module):
    def __init__(self, in_feats: int, h1: int, h2: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_feats, h1)
        self.conv2 = GCNConv(h1, h2)
        self.dropout = torch.nn.Dropout(dropout)
        self.lin = torch.nn.Linear(h2, 1)
        self.bn1 = torch.nn.BatchNorm1d(h1)
        self.bn2 = torch.nn.BatchNorm1d(h2)

    def forward(self, x, edge_index, edge_weight):
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        return self.lin(h).squeeze(-1)

# --------------------------------------------------
# 5) Optimized training with early stopping
# --------------------------------------------------

def train_model_optimized(train_loader, val_loader, model, optimizer, loss_fn, 
                         epochs=50, early_stopping_patience=10, device='cpu'):
    """
    Optimized training with early stopping and device support
    """
    model = model.to(device)
    history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, epochs+1):
        # Training
        model.train()
        loss_train = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_weight)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        
        avg_train_loss = loss_train / len(train_loader)
        history['train'].append(avg_train_loss)
        
        # Validation
        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_weight)
                loss_val += loss_fn(pred, batch.y).item()
        
        avg_val_loss = loss_val / len(val_loader)
        history['val'].append(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}")
    
    return history

# --------------------------------------------------
# 6) Main: optimized execution
# --------------------------------------------------

if __name__ == "__main__":
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1) Load & pivot data with optimized dtypes
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    load_start = time.time()
    df = pd.read_excel("nifty_ohlc_with_regimes.xlsx", parse_dates=["Date"])
    load_time = time.time() - load_start
    print(f"Excel loading took {load_time:.2f}s")
    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique tickers: {df['Ticker'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    pivot_start = time.time()
    lr = df.pivot(index="Date", columns="Ticker", values="LogRet").astype(np.float32)
    rv = df.pivot(index="Date", columns="Ticker", values="RV").astype(np.float32)
    pivot_time = time.time() - pivot_start
    print(f"Pivot operations took {pivot_time:.2f}s")
    print(f"LR shape: {lr.shape}, RV shape: {rv.shape}")
    print(f"Memory usage: LR={lr.memory_usage(deep=True).sum()/1024/1024:.1f}MB, RV={rv.memory_usage(deep=True).sum()/1024/1024:.1f}MB")
    
    # Check for NaN values
    lr_nans = lr.isna().sum().sum()
    rv_nans = rv.isna().sum().sum()
    print(f"NaN values: LR={lr_nans}, RV={rv_nans}")
    
    if lr_nans > 0 or rv_nans > 0:
        print("WARNING: NaN values detected - this may cause issues!")
    
    print(f"Total data loading time: {time.time() - load_start:.2f}s")
    
    # 2) Reduced hyper-parameter grid for faster testing
    print("\n" + "="*60)
    print("HYPERPARAMETER CONFIGURATION")
    print("="*60)
    
    param_grid = {
        'lr':      [1e-3, 5e-4],
        'batch':   [16, 32],  # Larger batches for efficiency
        'hidden1': [32, 64],  # Slightly larger for better capacity
        'hidden2': [32, 64],
        'N':       [125, 250],  # Reduced options
        's':       [4, 6],      # Reduced options
        'm':       [5, 10]
    }
    
    print("Parameter grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    
    combos = [dict(zip(param_grid, v)) for v in itertools.product(*param_grid.values())]
    print(f"\nTotal combinations: {len(combos)}")
    
    # Estimate computational load
    total_ete_computations = 0
    for N in param_grid['N']:
        for m in param_grid['m']:
            windows = len(lr) - N
            assets = lr.shape[1]
            computations = windows * assets * assets
            total_ete_computations += computations
            print(f"  N={N}, m={m}: {windows} windows × {assets}² assets = {computations:,} TE computations")
    
    print(f"\nTotal Transfer Entropy computations: {total_ete_computations:,}")
    print(f"Estimated time (rough): {total_ete_computations / 10000:.1f} minutes")
    
    # 3) Precompute all dynamic ETE graphs in parallel
    print("\n" + "="*60)
    print("PRECOMPUTING ETE GRAPHS")
    print("="*60)
    
    overall_start = time.time()
    ete_cache = precompute_edge_graphs_optimized(
        lr,
        Ns=param_grid['N'],
        ms=param_grid['m'],
        z_thresh=1.96,
        n_jobs=-1,
        verbose=True
    )
    overall_time = time.time() - overall_start
    print(f"\nOVERALL PRECOMPUTATION TIME: {overall_time/60:.1f} minutes")
    print(f"Cache contains {len(ete_cache)} parameter combinations")
    
    # 4) Phase 1: hyper-parameter sweep with early stopping
    results = []
    print("=== Phase 1: hyper-parameter sweep ===")
    
    for i, p in enumerate(combos):
        print(f"\n-- Testing {i+1}/{len(combos)}: {p}")
        edges = ete_cache[(p['N'], p['m'])]
        data = make_data_cached_optimized(lr, rv, p['N'], p['s'], p['m'], edges)
        
        if len(data) < 50:  # Skip if not enough data
            continue
            
        split = int(0.8 * len(data))
        train_ds, val_ds = data[:split], data[split:]
        tr_loader = DataLoader(train_ds, batch_size=p['batch'], shuffle=True)
        vl_loader = DataLoader(val_ds, batch_size=p['batch'])
        
        model = HETE_GCN_Optimized(in_feats=3, h1=p['hidden1'], h2=p['hidden2'])
        optimizer = torch.optim.Adam(model.parameters(), lr=p['lr'], weight_decay=1e-5)
        loss_fn = torch.nn.MSELoss()
        
        hist = train_model_optimized(tr_loader, vl_loader, model, optimizer, loss_fn, 
                                   epochs=50, early_stopping_patience=10, device=device)
        results.append({'params': p, 'val_loss': hist['val'][-1]})
    
    # 5) Select best and final retrain
    if results:
        best = min(results, key=lambda x: x['val_loss'])['params']
        print(f"\n>>> Best hyper-parameters: {best}")
        
        edges = ete_cache[(best['N'], best['m'])]
        data = make_data_cached_optimized(lr, rv, best['N'], best['s'], best['m'], edges)
        split = int(0.9 * len(data))
        tr_ds, te_ds = data[:split], data[split:]
        tr_loader = DataLoader(tr_ds, batch_size=best['batch'], shuffle=True)
        te_loader = DataLoader(te_ds, batch_size=best['batch'])
        
        model_fin = HETE_GCN_Optimized(in_feats=3, h1=best['hidden1'], h2=best['hidden2'])
        optimizer = torch.optim.Adam(model_fin.parameters(), lr=best['lr'], weight_decay=1e-5)
        loss_fn = torch.nn.MSELoss()
        
        hist = train_model_optimized(tr_loader, te_loader, model_fin, optimizer, loss_fn, 
                                   epochs=100, early_stopping_patience=15, device=device)
        print(f"\nFinal test loss: {hist['val'][-1]:.4f}")
        
        # 6) Save & plot
        torch.save(model_fin.state_dict(), "hete_gcn_best_optimized.pt")
        print("✅ Saved final model to hete_gcn_best_optimized.pt")
        
        plt.figure(figsize=(10, 6))
        plt.plot(hist['train'], label='Train')
        plt.plot(hist['val'], label='Test')
        plt.title("Loss curves (final)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No valid results found!")