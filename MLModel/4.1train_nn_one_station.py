import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_csv('4.1one_station.csv', skiprows=1)  # Skip the first row (header)

# Prepare input and output data
X = data.iloc[:, [i for i in range(3, 48) if i != 8]].values  # Columns 4, 5, 6, 7, 8, 10, 11 as input
y = data.iloc[:, 8].values    # Column 9 as output

# Normalize input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

X = X.astype(np.float32)
y = y.astype(np.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 60  # 设置batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(44, 64)  # Adjust input size to 7
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(32, 16) 
        self.dropout3 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(16, 1) 
    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = torch.nn.functional.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = NeuralNetwork()

# Define loss function and optimizer
criterion = nn.HuberLoss()
# L2: weight_decay
optimizer = optim.Adam(model.parameters(), weight_decay=0.0001, lr=0.0001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.3)

# Train the model
num_epochs = 2000
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_losses.append(epoch_train_loss / len(train_loader))
    
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            val_outputs = model(X_batch)
            val_loss = criterion(val_outputs, y_batch)
            epoch_val_loss += val_loss.item()
    val_losses.append(epoch_val_loss / len(test_loader))
    
    # Step the scheduler
    scheduler.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    
    # Save model parameters every 200 epochs
    if (epoch+1) % 200 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

# Plot the loss
plt.plot(train_losses, label='train_loss')
plt.plot(val_losses, label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    test_loss = 0
    for X_batch, y_batch in test_loader:
        test_loss += criterion(model(X_batch), y_batch).item()
print(f'Test Loss: {test_loss / len(test_loader)}')