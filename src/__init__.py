"""
VMD-LSTM Time Series Clustering Package

A deep learning approach for time series clustering that combines 
Variational Mode Decomposition (VMD) with LSTM Autoencoders.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your-email@example.com"

from .model import VMD_LSTM_Clustering, LSTMAutoEncoder
from .data_loader import load_UCR_data, get_dataset_info, list_available_datasets
from .utils import save_results, load_results, plot_convergence, plot_tsne_clustering

__all__ = [
    'VMD_LSTM_Clustering',
    'LSTMAutoEncoder', 
    'load_UCR_data',
    'get_dataset_info',
    'list_available_datasets',
    'save_results',
    'load_results',
    'plot_convergence',
    'plot_tsne_clustering'
]