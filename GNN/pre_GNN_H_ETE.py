import numpy as np
import pandas as pd
from collections import Counter

def discretise(series, bins=3):
    """Quantile-based binning for entropy calculations"""
    q = np.linspace(0, 1, bins+1)
    return pd.qcut(series, q, labels=False, duplicates="drop")

def transfer_entropy(x, y, lag=1):
    """
    Compute raw transfer entropy from Y → X using triple and pair frequencies
    
    Args:
        x: discretized time series (target)
        y: discretized time series (source) 
        lag: time lag for the transfer entropy calculation
    
    Returns:
        float: transfer entropy value
    """
    # x[t+1] ~ future of X, x[t] ~ past X, y[t] ~ past Y
    # Both x and y already discretised
    trip = list(zip(x[lag:], x[:-lag], y[:-lag]))
    pair = list(zip(x[lag:], x[:-lag]))
    p_trip = Counter(trip)
    p_pair = Counter(pair)
    p_x = Counter(x[lag:])
    N = len(trip)
    te = 0.0
    for (xf, xp, yp), c in p_trip.items():
        p_xyz = c/N
        p_xf_xp = p_pair[(xf, xp)]/N
        p_xf = p_x[xf]/N
        te += p_xyz * np.log2(p_xyz*p_xf / (p_xf_xp*p_xf))
    return te

def effective_transfer_entropy(x_raw, y_raw, m=50, bins=3, lag=1):
    """
    Compute Effective Transfer Entropy (ETE) from y → x.
    Subtracts shuffled baseline and computes Z-score.
    
    Args:
        x_raw: raw time series (target)
        y_raw: raw time series (source)
        m: number of shuffles for baseline
        bins: number of bins for discretization
        lag: time lag
    
    Returns:
        dict: containing TE, ETE, Z-score, and statistics
    """
    # Discretize
    x = discretise(x_raw, bins)
    y = discretise(y_raw, bins)
    
    # Drop NaNs due to discretisation or lag
    x, y = np.array(x), np.array(y)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    te_actual = transfer_entropy(x, y, lag)

    te_shuffled = []
    for _ in range(m):
        y_shuffled = np.random.permutation(y)
        te_shuffled.append(transfer_entropy(x, y_shuffled, lag))

    mu = np.mean(te_shuffled)
    sigma = np.std(te_shuffled, ddof=1)
    ete = te_actual - mu
    z = (te_actual - mu) / sigma if sigma > 0 else np.nan

    return {"TE": te_actual, "ETE": ete, "Z": z, "mu": mu, "sigma": sigma}

def compute_rte_zscores(te_mat, eps=1e-8):
    """
    Compute Z-scores for a TE matrix.
    Args:
        te_mat: 2D numpy array (TE matrix)
        eps: small value to avoid division by zero
    Returns:
        z_mat: 2D numpy array of Z-scores, same shape as te_mat
    """
    nonzero = te_mat[te_mat > 0]
    if len(nonzero) > 1:
        mu = nonzero.mean()
        sigma = nonzero.std(ddof=1)
        if sigma < eps:
            sigma = eps
        z_mat = (te_mat - mu) / sigma
    else:
        z_mat = np.zeros_like(te_mat)
    return z_mat

def compute_transfer_entropy(x_raw, y_raw, m=10, bins=3, lag=1):
    """
    Compute transfer entropy from y → x.
    This is a simplified wrapper that returns just the ETE value.
    
    Args:
        x_raw: raw time series (target)
        y_raw: raw time series (source)
        m: number of shuffles for baseline
        bins: number of bins for discretization
        lag: time lag
    
    Returns:
        float: effective transfer entropy value
    """
    result = effective_transfer_entropy(x_raw, y_raw, m, bins, lag)
    return result["ETE"] 

