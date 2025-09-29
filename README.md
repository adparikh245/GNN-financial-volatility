# GNN for Financial Time Series

**Inspired by:** "Graph-Based Stock Volatility Forecasting with Effective Transfer Entropy and Hurst-Based Regime Adaptation" by Sangheon Lee and Poongjin Cho

This project implements a Graph Neural Network (GNN) approach for stock volatility prediction using:
- **Effective Transfer Entropy (ETE)** for capturing information flow between stocks
- **Hurst Exponent** for regime detection and adaptation
- **Heterogeneous Graph Convolutional Networks (HETE-GCN)** for volatility forecasting

## Project Structure

```
Structure/
├── data/                          # Raw data files
│   ├── nifty_ohlc_formatted.xlsx
│   ├── NIFTY50_index_prices.xlsx
│   └── nifty_ohlc_with_regimes.xlsx
├── outputs/                       # Generated outputs
│   ├── transfer_entropy/          # TE matrices
│   │   ├── ETE_LogRet.xlsx
│   │   └── Zscore_LogRet.xlsx
│   ├── predictions/               # Model predictions
│   │   ├── out_of_sample_predictions.csv
│   │   └── out_of_sample_actuals.csv
│   └── results/                   # Analysis results
│       ├── per_ticker_metrics.csv
│       └── overall_metrics.csv
├── models/                        # Saved models
│   └── hete_gcn_best.pt
├── graphs/                        # Result graphs
│   └── oos_*.png files
├── GNN/                          # Code files
│   ├── pre_GNN_H_ETE.py
│   ├── trian_GNN.py
│   ├── predictions.py
│   └── results_analysis.py
└── notebooks/                     # Jupyter notebooks
    └── pre-GNN_H_ETE.ipynb
```

## Methodology

### 1. **Transfer Entropy Computation**
- Computes Effective Transfer Entropy (ETE) between all stock pairs
- Uses Z-scores to identify significant information flows
- Creates directed graph edges based on statistical significance

### 2. **Regime Detection**
- Implements Hurst exponent-based regime detection
- Adapts model behavior based on market regimes (trending vs mean-reverting)
- Uses rolling window analysis for regime classification

### 3. **Graph Neural Network**
- **HETE-GCN**: Heterogeneous Graph Convolutional Network
- **Node Features**: Log returns, realized volatility, regime flags
- **Edge Weights**: Transfer entropy values
- **Output**: Volatility predictions for all stocks

## Quick Start

### 1. Data Preparation
```bash
python GNN/pre_GNN_H_ETE.py
```
**Outputs:** `outputs/transfer_entropy/ETE_LogRet.xlsx`, `outputs/transfer_entropy/Zscore_LogRet.xlsx`

### 2. Model Training
```bash
python GNN/trian_GNN.py
```
**Outputs:** `models/hete_gcn_best.pt`

### 3. Model Prediction
```bash
python GNN/predictions.py
```
**Outputs:** 
- `outputs/predictions/out_of_sample_predictions.csv`
- `outputs/predictions/out_of_sample_actuals.csv`
- `graphs/oos_*.png` (individual ticker plots)

### 4. Results Analysis
```bash
python GNN/results_analysis.py
```
**Outputs:** Performance metrics and analysis

## Prerequisites

Make sure you have these data files in the `data/` directory:
- `nifty_ohlc_formatted.xlsx`
- `NIFTY50_index_prices.xlsx`

## Dependencies

Install required packages:
```bash
pip install -r GNN/requirements.txt
```

## File Organization

- **Raw Data**: All input Excel files go in `data/`
- **Generated Outputs**: All processed data goes in `outputs/` with subfolders
- **Models**: Saved model weights go in `models/`
- **Graphs**: All result plots go in `graphs/`
- **Code**: All Python scripts stay in `GNN/`
- **Notebooks**: Jupyter notebooks go in `notebooks/`

This structure keeps your project clean and organized!

## Limitations & Challenges

### **1. Index Composition Changes**
- **NIFTY50 constituents change over time** - Stocks enter/exit the index during the 5-year period
- **Regime detection may be inaccurate** - Hurst exponent calculated on changing stock universe
- **Transfer entropy relationships shift** - Graph structure becomes inconsistent over time

### **2. Data Quality Issues**
- **Survivorship bias** - Only includes stocks that remained in index
- **Missing data handling** - Different stocks may have different data availability
- **Corporate actions** - Stock splits, mergers affect price continuity

### **3. Model Limitations**
- **Static graph structure** - Transfer entropy edges don't adapt to changing relationships
- **Fixed time windows** - May not capture regime transitions effectively
- **Limited feature engineering** - Only basic price-based features

## Potential Improvements

### **Data Preprocessing**
- **Rolling index composition** - Use only stocks that were in index at each time point
- **Dynamic graph updates** - Recalculate transfer entropy periodically
- **Regime-consistent training** - Train separate models for different market regimes

### **Model Enhancements**
- **Temporal graph networks** - Allow graph structure to evolve over time
- **Attention mechanisms** - Focus on most relevant relationships
- **Ensemble methods** - Combine multiple regime-specific models
