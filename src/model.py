"""
VMD-LSTM Autoencoder model for time series clustering
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import rand_score, normalized_mutual_info_score
from vmdpy import VMD
import itertools

class LSTMAutoEncoder(nn.Module):
    """LSTM Autoencoder for feature extraction"""
    
    def __init__(self, input_size, num_layers, hidden_size, dropout=0):
        super(LSTMAutoEncoder, self).__init__()
        
        self.encoder = Encoder(input_size, num_layers, hidden_size, dropout)
        self.decoder = Decoder(hidden_size, num_layers, input_size, dropout)

    def forward(self, x):
        hidden_cell = self.encoder(x)
        output = self.decoder(hidden_cell)
        return hidden_cell, output

class Encoder(nn.Module):
    """LSTM Encoder"""
    
    def __init__(self, input_size, num_layers, hidden_size, dropout=0):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=False, 
            dropout=dropout, 
            bias=True
        )
        
    def forward(self, x):
        hidden_cell = self.lstm(x)
        hidden_cell = hidden_cell[0]
        return hidden_cell

class Decoder(nn.Module):
    """LSTM Decoder"""
    
    def __init__(self, input_size, num_layers, hidden_size, dropout=0):
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=False, 
            dropout=dropout, 
            bias=True
        )
        
    def forward(self, x):
        output = self.lstm(x)
        output = output[0]
        return output

class VMDDecomposer:
    """Variational Mode Decomposition wrapper"""
    
    def __init__(self, K, alpha=2000, tau=0.0, DC=0, init=1, tol=1e-7):
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol
    
    def decompose(self, data, device='cuda'):
        """
        Decompose time series data using VMD
        
        Args:
            data (torch.Tensor): Input time series data
            device (str): Device for tensor operations
            
        Returns:
            list: List of decomposed modes as torch tensors
        """
        print(f"Starting VMD decomposition with K={self.K}")
        data = data.squeeze().unsqueeze(dim=2).detach().cpu().numpy()
        print(f"Data shape for VMD: {data.shape}")
        
        results_list = []
        
        # Initialize result containers
        mode_results = {}
        for j in range(self.K):
            mode_results[f'u{j}_results'] = np.empty(shape=(0, data.shape[1]))
        
        try:
            # Decompose each time series
            for i in range(data.shape[0]):
                if i % 10 == 0:
                    print(f"Processing time series {i+1}/{data.shape[0]}")
                
                X = data[i].flatten()  # Ensure 1D input for VMD
                
                u, u_hat, omega = VMD(
                    X, self.alpha, self.tau, self.K, 
                    self.DC, self.init, self.tol
                )
                
                # Store decomposed modes
                for q in range(self.K):
                    mode = np.reshape(u[q], (1, data.shape[1]))
                    mode_results[f'u{q}_results'] = np.concatenate(
                        (mode_results[f'u{q}_results'], mode), axis=0
                    )
        
        except Exception as e:
            print(f"VMD decomposition failed: {e}")
            print("Using fallback approach with padding...")
            
            # Fallback with padding (원래 코드 방식)
            for i in range(data.shape[0]):
                if i % 10 == 0:
                    print(f"Fallback processing {i+1}/{data.shape[0]}")
                
                X = data[i].flatten()
                
                try:
                    u, u_hat, omega = VMD(
                        X, self.alpha, self.tau, self.K, 
                        self.DC, self.init, self.tol
                    )
                    
                    # 원래 코드의 길이 조정 방식
                    u = np.append(u, np.empty(shape=(self.K, 1)), axis=1)
                    
                    for q in range(self.K):
                        mode = np.reshape(u[q], (1, data.shape[1]))
                        mode_results[f'u{q}_results'] = np.concatenate(
                            (mode_results[f'u{q}_results'], mode), axis=0
                        )
                        
                except Exception as inner_e:
                    print(f"Fallback also failed for sample {i}: {inner_e}")
                    # Create zero-filled dummy modes
                    for q in range(self.K):
                        mode = np.zeros((1, data.shape[1]))
                        mode_results[f'u{q}_results'] = np.concatenate(
                            (mode_results[f'u{q}_results'], mode), axis=0
                        )
        
        print("VMD decomposition completed, converting to tensors...")
        
        # Convert to torch tensors
        for l in range(self.K):
            mode_data = np.expand_dims(mode_results[f'u{l}_results'], axis=1)
            mode_tensor = torch.tensor(mode_data, dtype=torch.float).to(device)
            results_list.append(mode_tensor)
            print(f"Mode {l} tensor shape: {mode_tensor.shape}")
        
        print(f"VMD decomposition finished, created {len(results_list)} modes")
        return results_list

class VMD_LSTM_Clustering:
    """Main clustering model combining VMD and LSTM Autoencoder"""
    
    def __init__(self, K, hidden_size, num_layers, num_classes, 
                 vmd_params=None, device='cuda'):
        self.K = K
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device
        
        # VMD parameters
        if vmd_params is None:
            vmd_params = {'alpha': 2000, 'tau': 0.0, 'tol': 1e-7}
        self.vmd_decomposer = VMDDecomposer(K, **vmd_params)
        
        # Will be initialized during fit
        self.models = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
    
    def _initialize_models(self, input_size):
        """Initialize LSTM Autoencoder models for each VMD mode"""
        self.models = {}
        
        for i in range(self.K):
            self.models[f'model{i}'] = LSTMAutoEncoder(
                input_size=input_size,
                num_layers=self.num_layers,
                hidden_size=self.hidden_size
            ).to(self.device)
        
        # Setup optimizer
        opt_params = []
        for i in range(self.K):
            opt_params.append(self.models[f'model{i}'].parameters())
        
        self.optimizer = torch.optim.Adam(
            itertools.chain(*opt_params), lr=0.0002
        )
    
    def fit_predict(self, X_train, X_test, y_test, num_epochs=1001, 
                   lr=0.0002, verbose=False):
        """
        Fit the model and predict clusters
        
        Args:
            X_train (torch.Tensor): Training data
            X_test (torch.Tensor): Test data  
            y_test (np.array): Test labels
            num_epochs (int): Number of training epochs
            lr (float): Learning rate
            verbose (bool): Print training progress
            
        Returns:
            tuple: (ri_scores, nmi_scores) lists
        """
        
        # VMD decomposition
        train_modes = self.vmd_decomposer.decompose(X_train, self.device)
        test_modes = self.vmd_decomposer.decompose(X_test, self.device)
        
        # Initialize models
        input_size = X_test.shape[2]
        self._initialize_models(input_size)
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        ri_scores = []
        nmi_scores = []
        
        print(f"Starting training for {num_epochs} epochs...")
        
        # Training loop
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass for all modes
            outputs = []
            hidden_cells = []
            
            for i in range(self.K):
                # Train on training modes
                _, output = self.models[f'model{i}'].forward(train_modes[i])
                # Get features from test modes
                hidden_cell, _ = self.models[f'model{i}'].forward(test_modes[i])
                
                outputs.append(output)
                hidden_cells.append(hidden_cell)
            
            # Reconstruction loss
            reconstructed = sum(outputs)
            loss = self.criterion(X_train, reconstructed)
            
            # Extract features for clustering
            combined_features = torch.cat(hidden_cells, 1).detach().cpu().numpy()
            feature_data = combined_features.reshape(
                X_test.shape[0], self.hidden_size, self.K
            )
            
            # Clustering
            kmeans = TimeSeriesKMeans(
                n_clusters=self.num_classes, 
                metric='euclidean', 
                max_iter=1000
            )
            y_pred = kmeans.fit_predict(feature_data)
            
            # Evaluation metrics
            ri = rand_score(y_test, y_pred)
            nmi = normalized_mutual_info_score(y_test, y_pred)
            
            ri_scores.append(ri)
            nmi_scores.append(nmi)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Loss: {loss.item():.8f}, '
                      f'RI: {ri:.4f}, NMI: {nmi:.4f}')
        
        if verbose:
            print(f'Best RI: {max(ri_scores):.4f}')
            print(f'Best NMI: {max(nmi_scores):.4f}')
        
        return ri_scores, nmi_scores