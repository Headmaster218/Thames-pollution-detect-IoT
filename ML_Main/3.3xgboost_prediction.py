import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('2.9sin_cos_with_new_features.csv')  # Skip the first row (header)
feature_names = data.columns[[i for i in range(3, 48) if i != 8]]  # Use columns 4 to 48 excluding column 9 as feature names

# Prepare input and output data
X = data.iloc[:, [i for i in range(3, 48) if i != 8]].values  # Columns 4 to 108 excluding column 9 as input
y = data.iloc[:, 8].values    # Column 9 as output

# 添加处理，如果输入值太小则设为0
X[np.abs(X) < 0.00001] = 0

# Normalize input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DMatrix for XGBoost
train_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
test_dmatrix = xgb.DMatrix(data=X_test, label=y_test)

# Define XGBoost model parameters
# 定义优化后的 XGBoost 参数
params = {
    'objective': 'reg:squarederror',  # 回归问题
    'colsample_bytree': 0.8,  # 80% 特征采样，减少过拟合
    'learning_rate': 0.05,  # 降低学习率，提高稳定性
    'max_depth': 8,  # 增加深度，允许更复杂的模式
    'alpha': 5,  # 降低 L1 正则，防止过度稀疏
    'lambda': 1,  # 加入 L2 正则，提升泛化能力
    'subsample': 0.8,  # 80% 训练数据采样，防止过拟合
}

# 训练 XGBoost 模型，加入早停机制
num_boost_round = 500  # 允许最多 500 轮训练
model = xgb.train(
    params=params, 
    dtrain=train_dmatrix, 
    num_boost_round=num_boost_round,  
    evals=[(train_dmatrix, 'train'), (test_dmatrix, 'eval')],  # 监测训练集 & 测试集
    early_stopping_rounds=20,  # 20 轮 RMSE 无改进则停止
    verbose_eval=10  # 每 10 轮打印一次进度
)


# Save the model
model.save_model('xgboost_model.json')

# Make predictions
preds = model.predict(test_dmatrix)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f'RMSE: {rmse}')

# Plot feature importance
ax = xgb.plot_importance(model, importance_type='weight', max_num_features=105)
ax.set_yticklabels(feature_names)
plt.xlabel('Feature importance')
plt.ylabel('Features')
plt.title('Feature importance')
plt.show()
