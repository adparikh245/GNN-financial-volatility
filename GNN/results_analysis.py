import numpy as np
import pandas as pd

# Assuming preds_df and trues_df are already defined in your session
# If not, load them from the CSVs saved earlier:
preds_df = pd.read_csv("out_of_sample_predictions.csv", index_col=0, parse_dates=True)
trues_df = pd.read_csv("out_of_sample_actuals.csv", index_col=0, parse_dates=True)

metrics_list = []
for ticker in preds_df.columns:
    y_true = trues_df[ticker].to_numpy()
    y_pred = preds_df[ticker].to_numpy()
    
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mae  = np.mean(np.abs(y_pred - y_true))
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    corr = np.corrcoef(y_pred, y_true)[0,1]
    # Hit ratio: fraction of times directional move is predicted correctly
    hits = np.mean(np.sign(np.diff(y_pred)) == np.sign(np.diff(y_true)))
    
    metrics_list.append({
        "Ticker": ticker,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "Correlation": corr,
        "Hit Ratio": hits
    })

metrics_df = pd.DataFrame(metrics_list).set_index("Ticker")

# Display per-ticker metrics
print("Per-Ticker Metrics:")
print(metrics_df)
print("\n")

# Compute mean ± std for each metric
summary = metrics_df.agg(['mean', 'std']).T
summary['mean ± std'] = summary.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)

# Display summary metrics
metrics_summary = summary[['mean ± std']].rename_axis("Metric").reset_index()
print("Overall Metrics (Mean ± Std):")
print(metrics_summary)

# Save the summary to a CSV file
metrics_summary.to_csv("metrics_summary.csv", index=False)
print("✅ Saved metrics_summary.csv")