"""
Utility functions for experiments and result handling
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def save_results(ri_results, nmi_results, param_combinations, 
                dataset_name, output_dir):
    """
    Save experimental results to CSV files
    
    Args:
        ri_results (list): List of RI score lists
        nmi_results (list): List of NMI score lists  
        param_combinations (list): List of (K, hidden_size) tuples
        dataset_name (str): Name of the dataset
        output_dir (str): Output directory path
    """
    
    # Convert to DataFrames
    ri_df = pd.DataFrame(ri_results).T
    nmi_df = pd.DataFrame(nmi_results).T
    
    # Set column names as parameter combinations
    column_names = [f"K{k}_H{h}" for k, h in param_combinations]
    ri_df.columns = column_names
    nmi_df.columns = column_names
    
    # Save results
    ri_path = os.path.join(output_dir, f"ri_{dataset_name}.csv")
    nmi_path = os.path.join(output_dir, f"nmi_{dataset_name}.csv")
    
    ri_df.to_csv(ri_path, index=False)
    nmi_df.to_csv(nmi_path, index=False)
    
    print(f"Results saved:")
    print(f"  RI scores: {ri_path}")
    print(f"  NMI scores: {nmi_path}")
    
    # Save summary
    summary_data = []
    for i, (k, h) in enumerate(param_combinations):
        summary_data.append({
            'K': k,
            'hidden_size': h,
            'max_ri': max(ri_results[i][:-1]),  # Exclude the duplicate max value
            'max_nmi': max(nmi_results[i][:-1]),
            'final_ri': ri_results[i][-2] if len(ri_results[i]) > 1 else ri_results[i][-1],
            'final_nmi': nmi_results[i][-2] if len(nmi_results[i]) > 1 else nmi_results[i][-1]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f"summary_{dataset_name}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary: {summary_path}")

def load_results(dataset_name, output_dir):
    """
    Load experimental results from CSV files
    
    Args:
        dataset_name (str): Name of the dataset
        output_dir (str): Output directory path
        
    Returns:
        tuple: (ri_df, nmi_df, summary_df)
    """
    ri_path = os.path.join(output_dir, f"ri_{dataset_name}.csv")
    nmi_path = os.path.join(output_dir, f"nmi_{dataset_name}.csv")
    summary_path = os.path.join(output_dir, f"summary_{dataset_name}.csv")
    
    ri_df = pd.read_csv(ri_path) if os.path.exists(ri_path) else None
    nmi_df = pd.read_csv(nmi_path) if os.path.exists(nmi_path) else None
    summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    
    return ri_df, nmi_df, summary_df

def plot_convergence(ri_scores, nmi_scores, save_path=None):
    """
    Plot convergence curves for RI and NMI scores
    
    Args:
        ri_scores (list): RI scores over epochs
        nmi_scores (list): NMI scores over epochs
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # RI plot
    ax1.plot(ri_scores[:-1], 'b-', linewidth=2)  # Exclude duplicate max value
    ax1.set_title('Rand Index Convergence')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RI Score')
    ax1.grid(True, alpha=0.3)
    
    # NMI plot
    ax2.plot(nmi_scores[:-1], 'r-', linewidth=2)  # Exclude duplicate max value
    ax2.set_title('NMI Convergence')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('NMI Score')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved: {save_path}")
    else:
        plt.show()

def plot_tsne_clustering(features, true_labels, pred_labels, 
                        dataset_name, K, hidden_size, save_path=None):
    """
    Plot t-SNE visualization of clustering results
    
    Args:
        features (np.array): Feature vectors
        true_labels (np.array): True cluster labels
        pred_labels (np.array): Predicted cluster labels
        dataset_name (str): Dataset name
        K (int): Number of VMD modes
        hidden_size (int): LSTM hidden size
        save_path (str): Path to save the plot
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(features)//4), random_state=42)
    features_2d = tsne.fit_transform(features.reshape(len(features), -1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # True labels plot
    scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=true_labels, cmap='tab10', s=20, alpha=0.7)
    ax1.set_title(f'True Labels\n{dataset_name} (K={K}, H={hidden_size})')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    
    # Predicted labels plot
    scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=pred_labels, cmap='tab10', s=20, alpha=0.7)
    ax2.set_title(f'Predicted Labels\n{dataset_name} (K={K}, H={hidden_size})')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved: {save_path}")
    else:
        plt.show()

def create_summary_report(output_dir, dataset_list=None):
    """
    Create a summary report of all experiments
    
    Args:
        output_dir (str): Output directory path
        dataset_list (list): List of datasets to include
    """
    if dataset_list is None:
        # Find all summary files
        dataset_list = []
        for file in os.listdir(output_dir):
            if file.startswith('summary_') and file.endswith('.csv'):
                dataset_name = file.replace('summary_', '').replace('.csv', '')
                dataset_list.append(dataset_name)
    
    all_results = []
    
    for dataset_name in dataset_list:
        _, _, summary_df = load_results(dataset_name, output_dir)
        if summary_df is not None:
            summary_df['dataset'] = dataset_name
            all_results.append(summary_df)
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Find best results for each dataset
        best_results = []
        for dataset in combined_df['dataset'].unique():
            dataset_data = combined_df[combined_df['dataset'] == dataset]
            best_ri_idx = dataset_data['max_ri'].idxmax()
            best_nmi_idx = dataset_data['max_nmi'].idxmax()
            
            best_ri_row = dataset_data.loc[best_ri_idx].copy()
            best_ri_row['metric'] = 'RI'
            best_ri_row['score'] = best_ri_row['max_ri']
            
            best_nmi_row = dataset_data.loc[best_nmi_idx].copy()
            best_nmi_row['metric'] = 'NMI'
            best_nmi_row['score'] = best_nmi_row['max_nmi']
            
            best_results.extend([best_ri_row, best_nmi_row])
        
        best_df = pd.DataFrame(best_results)
        
        # Save reports
        combined_path = os.path.join(output_dir, 'all_results.csv')
        best_path = os.path.join(output_dir, 'best_results.csv')
        
        combined_df.to_csv(combined_path, index=False)
        best_df.to_csv(best_path, index=False)
        
        print(f"Summary reports saved:")
        print(f"  All results: {combined_path}")
        print(f"  Best results: {best_path}")
        
        return combined_df, best_df
    
    return None, None

def pkl_load(fname):
    """Load pickle file (compatibility function)"""
    import pickle
    with open(fname, 'rb') as f:
        return pickle.load(f)

def pad_nan_to_target(array, target_length, axis=0):
    """Pad array with NaN to target length (compatibility function)"""
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, pad_size)
    
    return np.pad(array, pad_width, mode='constant', constant_values=np.nan)