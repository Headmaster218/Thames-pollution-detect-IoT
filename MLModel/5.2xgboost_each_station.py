import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import os

# 获取所有站点的CSV文件
station_files = [f for f in os.listdir('./station') if f.endswith('.csv')]

# 初始化保存所有站点权重的DataFrame
all_feature_importances = pd.DataFrame(columns=['Feature'])

for file in station_files:
    data = pd.read_csv(f'./station/{file}')
    
    # 检查数据量是否大于3
    if len(data) <= 3:
        continue

    feature_names = data.columns[[i for i in range(3, 48) if i != 8]]
    X = data.iloc[:, [i for i in range(3, 48) if i != 8]].values
    y = data.iloc[:, 8].values

    # 添加处理，如果输入值太小则设为0
    X[np.abs(X) < 0.00001] = 0
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, f'scaler_{file}.pkl')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    train_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    test_dmatrix = xgb.DMatrix(data=X_test, label=y_test)

    # Define XGBoost model parameters
    params = {
        'objective': 'reg:squarederror',
        'colsample_bytree': 0.8,
        'learning_rate': 0.05,
        'max_depth': 8,
        'alpha': 5,
        'lambda': 1,
        'subsample': 0.8,
    }

    # 训练 XGBoost 模型，加入早停机制
    num_boost_round = 500
    model = xgb.train(
        params=params, 
        dtrain=train_dmatrix, 
        num_boost_round=num_boost_round,  
        evals=[(train_dmatrix, 'train'), (test_dmatrix, 'eval')],
        early_stopping_rounds=20,
        verbose_eval=10
    )

    # Save the model
    model.save_model(f'xgboost_model_{file}.json')
    preds = model.predict(test_dmatrix)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f'{file} RMSE: {rmse}')

    # 获取特征重要性并保存
    feature_importances = model.get_score(importance_type='weight')
    feature_importances_df = pd.DataFrame(list(feature_importances.items()), columns=['Feature', f'Importance_{file}'])
    
    # 确保所有特征都保存，且按顺序
    all_feature_importances = pd.merge(all_feature_importances, feature_importances_df, on='Feature', how='outer').fillna(0)
    all_feature_importances = all_feature_importances.sort_values(by='Feature').reset_index(drop=True)

# 保存所有站点的特征重要性到CSV
all_feature_importances.to_csv('all_feature_importances.csv', index=False)
