import pandas as pd
import numpy as np
import torch
import joblib
import torch.nn as nn


# Load the scaler
scaler = joblib.load('scaler.pkl')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(9, 64)  # Adjust input size to 7
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the model
model = NeuralNetwork()
model.load_state_dict(torch.load('model_epoch_1000.pth'))
model.eval()

# Load new data for prediction
data = pd.read_csv('test_data.csv')
X_new = data.iloc[:, [3, 4, 5, 6, 7, 9, 10, 11, 12]].values  # Include header row for input

# Normalize new data
X_new = scaler.transform(X_new)
X_new = torch.tensor(X_new, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    predictions = model(X_new).numpy()

# Save predictions to the 9th column of the original data
data['Prediction'] = predictions
data.to_csv('test_data.csv', index=False)
