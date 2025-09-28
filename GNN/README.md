# GNN-based Volatility Prediction with Transfer Entropy

This folder contains the implementation of a Heterogeneous Graph Neural Network (GNN) for predicting realized volatility (RV) in NIFTY50 stocks using transfer entropy-based relationships.

## 📁 File Structure

### Core Scripts
- **`pre_GNN_H_ETE.py`** - Transfer Entropy calculations and ETE matrix computation
- **`retrian_GNN.py`** - Model training script with optimized hyperparameters
- **`predictions.py`** - Inference script for out-of-sample predictions
- **`results_analysis.py`** - Performance metrics calculation and visualization

### Model & Data
- **`hete_gcn_best.pt`** - Trained model weights (best performing configuration)

## 🔄 Workflow

### 1. Data Preparation
The workflow starts with the Jupyter notebook `pre-GNN_H_ETE.ipynb` in the root directory, which:
- Loads NIFTY50 stock data
- Computes log returns and realized volatility
- Calculates Hurst exponents for regime detection
- Computes transfer entropy matrices (ETE and Z-scores)

### 2. Transfer Entropy Computation (`pre_GNN_H_ETE.py`)
```python
# Key functions:
- discretise()           # Quantile-based binning
- transfer_entropy()     # Raw transfer entropy calculation
- effective_transfer_entropy()  # ETE with statistical significance
- compute_rte_zscores()  # Z-score normalization
```

### 3. Model Training (`retrian_GNN.py`)
- **Architecture**: 2-layer GCN with ReLU activation
- **Input Features**: [Log Returns, Realized Volatility, Regime Flags]
- **Best Hyperparameters**:
  - Learning Rate: 1e-3
  - Batch Size: 8
  - Hidden Layers: [20, 15]
  - Window Size: 125
  - Regime Threshold: 6
  - Shuffle Iterations: 10

### 4. Inference (`predictions.py`)
- Loads trained model
- Generates out-of-sample predictions
- Saves predictions and actuals to CSV
- Creates visualization plots for each ticker

### 5. Results Analysis (`results_analysis.py`)
- Computes performance metrics (RMSE, MAE, MAPE, Correlation, Hit Ratio)
- Generates summary statistics
- Saves results to CSV files

## 🚀 Usage

### Training
```bash
cd GNN
python retrian_GNN.py
```

### Inference
```bash
cd GNN
python predictions.py
```

### Analysis
```bash
cd GNN
python results_analysis.py
```

## 📊 Output Files

- `out_of_sample_predictions.csv` - Model predictions
- `out_of_sample_actuals.csv` - True values
- `per_ticker_metrics.csv` - Individual stock performance
- `metrics_summary.csv` - Overall performance summary
- `oos_*.png` - Visualization plots for each ticker

## 🔧 Dependencies

- torch
- torch-geometric
- pandas
- numpy
- matplotlib
- scipy

## 📈 Model Performance

The model uses transfer entropy to capture non-linear dependencies between stocks and incorporates regime information through Hurst exponent analysis. The GNN architecture allows for dynamic relationship modeling based on statistical significance of information transfer.

## 🎯 Key Features

1. **Transfer Entropy Integration**: Uses ETE matrices to build graph edges
2. **Regime Detection**: Incorporates market regime information via Hurst exponents
3. **Statistical Significance**: Filters edges using Z-score thresholds (>1.96)
4. **Heterogeneous Graph**: Captures complex inter-stock relationships
5. **Out-of-Sample Validation**: Proper train/test split for robust evaluation

