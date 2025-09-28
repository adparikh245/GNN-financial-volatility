#!/usr/bin/env python3
"""
Complete pipeline for GNN-based volatility prediction.
Run this script to execute the entire workflow from training to analysis.
"""

import os
import sys
import subprocess
import argparse

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              check=True, 
                              capture_output=True, 
                              text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}:")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Run GNN volatility prediction pipeline')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--predict', action='store_true', help='Run predictions')
    parser.add_argument('--analyze', action='store_true', help='Run analysis')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    
    args = parser.parse_args()
    
    if not any([args.train, args.predict, args.analyze, args.all]):
        print("Please specify what to run. Use --help for options.")
        return
    
    # Check if we're in the right directory
    if not os.path.exists('trian_GNN.py'):
        print("❌ Please run this script from the GNN directory")
        return
    
    success = True
    
    if args.all or args.train:
        success &= run_script('trian_GNN.py', 'Training GNN Model')
    
    if args.all or args.predict:
        success &= run_script('predictions.py', 'Generating Predictions')
    
    if args.all or args.analyze:
        success &= run_script('results_analysis.py', 'Analyzing Results')
    
    if success:
        print(f"\n{'='*50}")
        print("🎉 Pipeline completed successfully!")
        print("Check the output files:")
        print("- outputs/predictions/out_of_sample_predictions.csv")
        print("- outputs/predictions/out_of_sample_actuals.csv") 
        print("- outputs/results/per_ticker_metrics.csv")
        print("- outputs/results/metrics_summary.csv")
        print("- graphs/oos_*.png (visualization plots)")
        print(f"{'='*50}")
    else:
        print("\n❌ Pipeline failed. Check the error messages above.")

if __name__ == "__main__":
    main()

