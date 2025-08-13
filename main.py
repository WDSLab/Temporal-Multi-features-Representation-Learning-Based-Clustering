#!/usr/bin/env python3
"""
Time Series Clustering with VMD and LSTM Autoencoder

This script performs time series clustering using Variational Mode Decomposition (VMD)
and LSTM Autoencoders for feature extraction.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_UCR_data
from src.model import VMD_LSTM_Clustering
from src.utils import save_results, create_output_dir

def parse_args():
    parser = argparse.ArgumentParser(description='VMD-LSTM Time Series Clustering')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='GunPoint',
                       help='Dataset name from UCR/UEA archive')
    parser.add_argument('--data_dir', type=str, default='datasets/UCR',
                       help='Path to dataset directory')
    
    # Model hyperparameters
    parser.add_argument('--K_list', type=int, nargs='+', default=[2, 3, 4, 5, 6],
                       help='List of VMD decomposition modes to try')
    parser.add_argument('--hidden_size_list', type=int, nargs='+', 
                       default=[10, 50, 100, 400, 800, 1200, 1600, 2000],
                       help='List of LSTM hidden sizes to try')
    parser.add_argument('--num_epochs', type=int, default=1001,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='Number of LSTM layers')
    
    # VMD parameters
    parser.add_argument('--vmd_alpha', type=int, default=2000,
                       help='VMD bandwidth constraint parameter')
    parser.add_argument('--vmd_tau', type=float, default=0.0,
                       help='VMD noise-tolerance parameter')
    parser.add_argument('--vmd_tol', type=float, default=1e-7,
                       help='VMD convergence tolerance')
    
    # Experiment settings
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')
    
    # Grid search options
    parser.add_argument('--single_run', action='store_true',
                       help='Run single experiment with first hyperparameters')
    parser.add_argument('--dataset_list', type=str, 
                       help='CSV file containing list of datasets to run')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def run_single_experiment(args, dataset_name, K, hidden_size):
    """Run a single clustering experiment"""
    print(f"Running: {dataset_name}, K={K}, hidden_size={hidden_size}")
    
    # Load data
    X_train, y_train, X_test, y_test = load_UCR_data(
        dataset_name, args.data_dir, device=args.device
    )
    
    # Initialize model
    model = VMD_LSTM_Clustering(
        K=K,
        hidden_size=hidden_size,
        num_layers=args.num_layers,
        num_classes=len(np.unique(y_train)),
        vmd_params={
            'alpha': args.vmd_alpha,
            'tau': args.vmd_tau, 
            'tol': args.vmd_tol
        },
        device=args.device
    )
    
    # Train and evaluate
    ri_scores, nmi_scores = model.fit_predict(
        X_train, X_test, y_test,
        num_epochs=args.num_epochs,
        lr=args.lr,
        verbose=args.verbose
    )
    
    return ri_scores, nmi_scores

def run_grid_search(args, dataset_name):
    """Run grid search over hyperparameters"""
    
    # Create parameter combinations
    param_combinations = list(product(args.K_list, args.hidden_size_list))
    
    ri_results = []
    nmi_results = []
    
    print(f"Running {len(param_combinations)} experiments for {dataset_name}")
    
    for K, hidden_size in tqdm(param_combinations, desc="Grid Search"):
        ri_scores, nmi_scores = run_single_experiment(
            args, dataset_name, K, hidden_size
        )
        
        # Append max score to the end of the list
        ri_scores.append(max(ri_scores))
        nmi_scores.append(max(nmi_scores))
        
        ri_results.append(ri_scores)
        nmi_results.append(nmi_scores)
    
    return ri_results, nmi_results, param_combinations

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    create_output_dir(args.output_dir)
    
    # Save experiment configuration
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 60)
    print("VMD-LSTM Time Series Clustering")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Determine datasets to run
    if args.dataset_list:
        dataset_df = pd.read_csv(args.dataset_list, header=None)
        datasets = dataset_df[0].tolist()
    else:
        datasets = [args.dataset]
    
    # Run experiments
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        
        if args.single_run:
            # Single run with first hyperparameters
            K = args.K_list[0]
            hidden_size = args.hidden_size_list[0]
            ri_scores, nmi_scores = run_single_experiment(
                args, dataset_name, K, hidden_size
            )
            
            print(f"Max RI: {max(ri_scores):.4f}")
            print(f"Max NMI: {max(nmi_scores):.4f}")
            
        else:
            # Grid search
            ri_results, nmi_results, param_combinations = run_grid_search(
                args, dataset_name
            )
            
            # Save results
            save_results(
                ri_results, nmi_results, param_combinations,
                dataset_name, args.output_dir
            )
            
            # Print best results
            best_ri_idx = np.argmax([max(ri) for ri in ri_results])
            best_nmi_idx = np.argmax([max(nmi) for nmi in nmi_results])
            
            print(f"Best RI: {max(ri_results[best_ri_idx]):.4f} "
                  f"(K={param_combinations[best_ri_idx][0]}, "
                  f"hidden_size={param_combinations[best_ri_idx][1]})")
            
            print(f"Best NMI: {max(nmi_results[best_nmi_idx]):.4f} "
                  f"(K={param_combinations[best_nmi_idx][0]}, "
                  f"hidden_size={param_combinations[best_nmi_idx][1]})")
        
        print("-" * 40)

if __name__ == "__main__":
    main()