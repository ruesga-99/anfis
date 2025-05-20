import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    def __init__(self, csv_path, target_column):
        self.csv_path = csv_path
        self.target_column = target_column
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def load_data(self):
        df = pd.read_csv(self.csv_path)
        X = df.drop(self.target_column, axis=1).values.astype(np.float32)
        y = df[self.target_column].values.astype(np.float32).reshape(-1, 1)
        
        # Normalize
        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y).flatten()
        
        return X, y