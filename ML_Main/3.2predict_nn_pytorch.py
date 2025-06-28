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
        self.fc1 = nn.Linear(44, 128)  # 输入维度调整为44
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = torch.nn.functional.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = self.fc4(x)
        return x

# Load the model
model = NeuralNetwork()
model.load_state_dict(torch.load('3.1model_loss_0.8230@397_epoch.pth'))  # 加载最佳验证损失模型
model.eval()

# Load new data for prediction
data = pd.read_csv('3.2test_data.csv')
X_new = data.iloc[:, [i for i in range(3, 48) if i != 8]].values  # 输入特征维度调整为与训练一致

# Normalize new data
X_new = scaler.transform(X_new)
X_new = torch.tensor(X_new, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    predictions = model(X_new).numpy()

# Save predictions to the 9th column of the original data
data['Prediction'] = predictions
data.to_csv('3.2test_data_with_predictions.csv', index=False)  # 保存到新的文件
