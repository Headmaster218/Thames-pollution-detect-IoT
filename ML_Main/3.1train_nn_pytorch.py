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
from sklearn.metrics import r2_score  # Import r2_score

# Load dataset
data = pd.read_csv('2.9sin_cos_with_new_features.csv', skiprows=1)  # Skip the first row (header)

# Prepare input and output data
X = data.iloc[:, [i for i in range(3, 48) if i != 8]].values  # Select columns 4-48 excluding column 8 as input
y = data.iloc[:, 8].values    # Select column 9 as output

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

batch_size = 32  # Set batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(44, 128)  # Adjust input size
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

model = NeuralNetwork()

# Define loss function and optimizer
criterion = nn.HuberLoss()
# L2 regularization: weight_decay
optimizer = optim.Adam(model.parameters(), weight_decay=0.0001, lr=0.0001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.93)

# Train the model
num_epochs = 300
train_losses = []
val_losses = []
train_r2_scores = []  # Save training R² scores
val_r2_scores = []    # Save validation R² scores

best_val_loss = float('inf')  # Initialize best validation loss

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    train_preds = []  # Save training predictions
    train_targets = []  # Save training targets
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        train_preds.extend(outputs.detach().numpy())
        train_targets.extend(y_batch.numpy())
    train_losses.append(epoch_train_loss / len(train_loader))
    train_r2_scores.append(r2_score(train_targets, train_preds))  # Calculate training R²
    
    model.eval()
    epoch_val_loss = 0
    val_preds = []  # Save validation predictions
    val_targets = []  # Save validation targets
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            val_outputs = model(X_batch)
            val_loss = criterion(val_outputs, y_batch)
            epoch_val_loss += val_loss.item()
            val_preds.extend(val_outputs.numpy())
            val_targets.extend(y_batch.numpy())
    val_losses.append(epoch_val_loss / len(test_loader))
    val_r2_scores.append(r2_score(val_targets, val_preds))  # Calculate validation R²
    
    # Check if current validation loss is the best
    if val_losses[-1] < best_val_loss and val_losses[-1] < 1:
        best_val_loss = val_losses[-1]
        torch.save(model.state_dict(), f'model_best_val_loss.pth')
        print(f'{best_val_loss:.4f}@{epoch}_epoch')
    
    # Step the scheduler
    scheduler.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train R²: {train_r2_scores[-1]:.4f}, Val R²: {val_r2_scores[-1]:.4f}')

# Plot the loss and R²
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='train_loss')
plt.plot(val_losses, label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_r2_scores, label='train_r2')
plt.plot(val_r2_scores, label='val_r2')
plt.xlabel('Epochs')
plt.ylabel('R² Score')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    test_loss = 0
    for X_batch, y_batch in test_loader:
        test_loss += criterion(model(X_batch), y_batch).item()
print(f'Test Loss: {test_loss / len(test_loader)}')