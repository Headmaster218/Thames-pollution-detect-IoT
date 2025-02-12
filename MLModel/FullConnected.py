import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# 示例Dataset类
class CustomDataset(Dataset):
    def __init__(self, data):
        # 假设 data 是一个 numpy 数组，形状为 (n_samples, 6)
        self.x = torch.tensor(data.iloc[:, :4].values, dtype=torch.float32)  # 前四个factor
        self.y = torch.tensor(data.iloc[:, 4:].values, dtype=torch.float32)  # 后两个factor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 全连接神经网络模型
class FactorRelationModel(nn.Module):
    def __init__(self):
        super(FactorRelationModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 输出是两个比例值
        )

    def forward(self, x):
        return self.fc(x)

# 数据加载
# 假设 data 是一个 numpy 数组，加载你的数据替换此处
# data = np.load("your_data.npy")
csv_file = "./filtered_Results_MADE.csv"  # 替换为实际文件路径
data = pd.read_csv(csv_file)

# 创建Dataset和DataLoader
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型、损失函数和优化器
model = FactorRelationModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# 测试模型
model.eval()
with torch.no_grad():
    for batch_x, batch_y in dataloader:
        predictions = model(batch_x)
        print("Predictions:", predictions)
        print("True Values:", batch_y)
        break  # 打印一个批次的结果

# 保存模型
torch.save(model.state_dict(), "factor_relation_model.pth")
