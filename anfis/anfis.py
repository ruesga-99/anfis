import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ANFIS(nn.Module):
    def __init__(self, n_input, n_mfs=3):
        """
        - n_input: Amount of input variables.
        - n_mfs: Amount of membership functions per input.
        """
        super(ANFIS, self).__init__()
        self.n_input = n_input
        self.n_mfs = n_mfs
        
        # Gaussian Membership Functions (antecedents)
        self.mu = nn.Parameter(torch.randn(n_input, n_mfs))                # Means
        self.sigma = nn.Parameter(torch.abs(torch.randn(n_input, n_mfs)))  # Sigmas
        
        # Linear layer for the consequent part (Sugeno-type output)
        self.consequent = nn.Linear(n_input * n_mfs, 1, bias=True)
        
    def forward(self, x):
        """
        - x: input tensor of shape [batch_size, n_input]
        - returns: output tensor of shape [batch_size]
        """

        batch_size = x.size(0)

        # Layer 1: Fuzzification
        x = x.unsqueeze(-1).expand(-1, -1, self.n_mfs)  # [batch, n_input, n_mfs]
        mu = self.mu.unsqueeze(0)
        sigma = self.sigma.unsqueeze(0)
        mfs = torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) # Gaussian MFs
        
        # Layer 2: Rule definition
        rules = torch.prod(mfs, dim=1)

        # Layer 3: Normalization
        rules_norm = rules / (rules.sum(dim=1, keepdim=True) + 1e-12)
        
        # Layer 4: Consequent layer
        x_tiled = x.reshape(batch_size, -1)
        z = self.consequent(x_tiled)

        # Layer 5: Output 
        return torch.sum(rules_norm * z, dim=1)
    
    def fit(self, X_train, y_train, epochs=1000, lr=0.01, val_split=0.2):
        """
        - X_train, y_train: training data
        - epochs: number of training epochs
        - lr: learning rate
        - val_split: fraction of data to use for validation
        """

        history = {'train_loss': [], 'val_loss': []}

        # Convert data to Torch tensors if needed
        X_train = torch.tensor(X_train, dtype=torch.float32) if not isinstance(X_train, torch.Tensor) else X_train.float()
        y_train = torch.tensor(y_train, dtype=torch.float32) if not isinstance(y_train, torch.Tensor) else y_train.float()
        
        # Divide training and validation data
        indices = np.arange(len(X_train))
        train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)
        
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        X_train = X_train[train_idx]
        y_train = y_train[train_idx]
        
        # Set training hyperparameters
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Execute training
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            history['train_loss'].append(loss.item())

            with torch.no_grad():
                val_outputs = self(X_val)
                val_loss = criterion(val_outputs, y_val)
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1:4d} | Train Loss: {loss.item():.15f} | Val Loss: {val_loss.item():.15f}")

        return history
    
    def predict(self, X):
        """
        - X: input data
        - returns: predicted values as a NumPy array
        """
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X.float()
            return self(X).numpy()
        
    def plot_membership_functions(self, X):
        """
        - X: input dataset used to determine the plotting range
        """

        fig_height = max(3 * self.n_input, 5)
        plt.figure(figsize=(8, fig_height))

        for i in range(self.n_input):
            plt.subplot(self.n_input, 1, i + 1)
            x_range = np.linspace(X[:, i].min(), X[:, i].max(), 200)

            for j in range(self.n_mfs):
                mu = self.mu[i, j].item()
                sigma = self.sigma[i, j].item()
                y = np.exp(-((x_range - mu) ** 2) / (2 * sigma ** 2))
                plt.plot(x_range, y, label=f'MF {j + 1}', linewidth=1)

            plt.title(f'Variable {i + 1}', fontsize=13)
            plt.xlabel(f'Input {i + 1}', fontsize=11)
            plt.ylabel('Membership', fontsize=11)
            plt.ylim(0, 1.05)
            plt.legend(loc='upper right', fontsize=9)
            plt.grid(True)

        plt.tight_layout(pad=2.0)
        plt.show()