"""
Data loading utilities for UCR/UEA time series datasets
"""

import os
import numpy as np
import pandas as pd
import torch
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def load_UCR_data(dataset_name, data_dir='datasets/UCR', device='cuda'):
    """
    Load UCR dataset with the custom data loader format
    
    Args:
        dataset_name (str): Name of the dataset
        data_dir (str): Path to the dataset directory
        device (str): Device to load tensors on
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test) as torch tensors
    """
    
    train_file = os.path.join(data_dir, dataset_name, dataset_name + "_TRAIN.tsv")
    test_file = os.path.join(data_dir, dataset_name, dataset_name + "_TEST.tsv")
    
    # Check if files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    # Load data
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    # Extract features and labels
    train_data = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test_data = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Check if dataset needs normalization
    non_normalized_datasets = {
        'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'BME',
        'Chinatown', 'Crop', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'Fungi',
        'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3', 'GesturePebbleZ1',
        'GesturePebbleZ2', 'GunPointAgeSpan', 'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung', 'HouseTwenty', 'InsectEPGRegularTrain',
        'InsectEPGSmallTrain', 'MelbournePedestrian', 'PickupGestureWiimoteZ',
        'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'PLAID', 'PowerCons',
        'Rock', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ', 'SmoothSubspace', 'UMD'
    }
    
    # Apply normalization if needed
    if dataset_name not in non_normalized_datasets:
        mean = np.nanmean(train_data)
        std = np.nanstd(train_data)
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std
    
    # Add channel dimension
    train_data = train_data[..., np.newaxis]
    test_data = test_data[..., np.newaxis]
    
    # Apply tslearn normalization (mean-variance scaling)
    scaler = TimeSeriesScalerMeanVariance()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(train_data, dtype=torch.float).to(device)
    X_test = torch.tensor(test_data, dtype=torch.float).to(device)
    
    # Reshape for LSTM input: (batch, time, features) -> (batch, features, time)
    X_train = X_train.squeeze().unsqueeze(dim=1)
    X_test = X_test.squeeze().unsqueeze(dim=1)
    
    print(f"Loaded {dataset_name}:")
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"  Number of classes: {len(np.unique(train_labels))}")
    print(f"  Train samples: {len(train_labels)}, Test samples: {len(test_labels)}")
    
    return X_train, train_labels, X_test, test_labels

def get_dataset_info(dataset_name, data_dir='datasets/UCR'):
    """
    Get basic information about a dataset without loading it
    
    Args:
        dataset_name (str): Name of the dataset
        data_dir (str): Path to the dataset directory
    
    Returns:
        dict: Dataset information
    """
    train_file = os.path.join(data_dir, dataset_name, dataset_name + "_TRAIN.tsv")
    test_file = os.path.join(data_dir, dataset_name, dataset_name + "_TEST.tsv")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        return None
    
    # Read just the first few lines to get info
    train_df = pd.read_csv(train_file, sep='\t', header=None, nrows=5)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    
    info = {
        'name': dataset_name,
        'train_size': len(pd.read_csv(train_file, sep='\t', header=None)),
        'test_size': len(test_df),
        'length': len(train_df.columns) - 1,  # -1 for label column
        'num_classes': len(pd.read_csv(train_file, sep='\t', header=None)[0].unique())
    }
    
    return info

def list_available_datasets(data_dir='datasets/UCR'):
    """
    List all available datasets in the data directory
    
    Args:
        data_dir (str): Path to the dataset directory
    
    Returns:
        list: List of available dataset names
    """
    if not os.path.exists(data_dir):
        return []
    
    datasets = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            train_file = os.path.join(item_path, item + "_TRAIN.tsv")
            test_file = os.path.join(item_path, item + "_TEST.tsv")
            if os.path.exists(train_file) and os.path.exists(test_file):
                datasets.append(item)
    
    return sorted(datasets)