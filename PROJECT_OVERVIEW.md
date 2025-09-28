# NIFTY50 Volatility Prediction using Graph Neural Networks

## 🎯 Project Overview

This repository implements a novel approach to predicting realized volatility in NIFTY50 stocks using Graph Neural Networks (GNNs) with transfer entropy-based relationships and regime detection.

## 🔬 Methodology

### 1. Transfer Entropy Analysis
- **Effective Transfer Entropy (ETE)**: Measures information flow between stocks
- **Statistical Significance**: Z-score filtering (>1.96) for robust relationships
- **Non-linear Dependencies**: Captures complex inter-stock relationships

### 2. Regime Detection
- **Hurst Exponent**: Quantifies market persistence/mean reversion
- **Dynamic Regimes**: Above/Below 0.5 threshold for momentum/mean-reverting periods
- **Feature Engineering**: Regime flags incorporated into GNN inputs

### 3. Graph Neural Network Architecture
- **Heterogeneous GCN**: 2-layer Graph Convolutional Network
- **Input Features**: [Log Returns, Realized Volatility, Regime Flags]
- **Edge Weights**: Transfer entropy values for relationship strength
- **Output**: Predicted realized volatility for next period

## 📁 Repository Structure

```
EquirusResearch/
├── GNN/                          # Core GNN implementation
│   ├── pre_GNN_H_ETE.py         # Transfer entropy calculations
│   ├── retrian_GNN.py           # Model training
│   ├── predictions.py           # Inference & visualization
│   ├── results_analysis.py     # Performance metrics
│   ├── run_pipeline.py         # Complete workflow script
│   ├── requirements.txt        # Dependencies
│   ├── README.md               # GNN documentation
│   └── hete_gcn_best.pt        # Trained model weights
├── pre-GNN_H_ETE.ipynb         # Data preparation notebook
├── Result_Graphs/               # Visualization outputs
├── *.xlsx                       # Data files
├── *.csv                        # Results files
└── *.png                        # Plot outputs
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r GNN/requirements.txt
```

### Complete Pipeline
```bash
cd GNN
python run_pipeline.py --all
```

### Individual Steps
```bash
# 1. Data preparation (run notebook first)
jupyter notebook pre-GNN_H_ETE.ipynb

# 2. Train model
python GNN/retrian_GNN.py

# 3. Generate predictions
python GNN/predictions.py

# 4. Analyze results
python GNN/results_analysis.py
```

## 📊 Key Results

- **Model Architecture**: 2-layer GCN with [20, 15] hidden units
- **Training Window**: 125-day rolling window
- **Regime Threshold**: 6-day Hurst exponent analysis
- **Edge Filtering**: Z-score > 1.96 for statistical significance
- **Performance**: Out-of-sample validation with comprehensive metrics

## 🔧 Technical Details

### Data Requirements
- NIFTY50 stock prices (OHLC format)
- Date range with sufficient history for rolling calculations
- Clean, non-missing data for all tickers

### Model Hyperparameters
- Learning Rate: 1e-3
- Batch Size: 8
- Hidden Layers: [20, 15]
- Window Size: 125
- Regime Threshold: 6
- Shuffle Iterations: 10

### Output Files
- `out_of_sample_predictions.csv` - Model predictions
- `out_of_sample_actuals.csv` - True values
- `per_ticker_metrics.csv` - Individual performance
- `metrics_summary.csv` - Overall statistics
- `oos_*.png` - Visualization plots

## 📈 Performance Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Correlation**: Pearson correlation coefficient
- **Hit Ratio**: Directional accuracy

## 🎓 Research Applications

This implementation demonstrates:
1. **Information Theory in Finance**: Transfer entropy for relationship modeling
2. **Graph Neural Networks**: Dynamic relationship learning
3. **Regime Detection**: Hurst exponent for market state identification
4. **Volatility Prediction**: Multi-asset realized volatility forecasting

## 📚 References

- Transfer Entropy for financial time series
- Graph Neural Networks for financial modeling
- Hurst exponent for regime detection
- Realized volatility prediction methods

## 🤝 Contributing

This is a research implementation. For questions or improvements, please refer to the documentation in each module.

## 📄 License

Research implementation - please cite appropriately if used in academic work.

