# baseline_runner.py
import pandas as pd
import numpy as np
from pathlib import Path

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# xgboost
from xgboost import XGBRegressor

# keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ReduceLROnPlateau

###############################################################################
# 0. 读取数据
###############################################################################
DATA_PATH = Path("./2.6useful_data_New.csv")   # 改成你的真实路径
df = pd.read_csv(DATA_PATH)

FEATURE_COLS = ['Oxygen Diss', 'Temp Water', 'pH', 'ln Turd', 'ln Cond']
TARGET_COL = 'ln Coliform'

X      = df[FEATURE_COLS].astype(float).copy()
y      = df[TARGET_COL].astype(float).copy()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

###############################################################################
# 1. 评估函数
###############################################################################
def evaluate(model_name, y_true, y_pred, results):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    results.append({
        "Model": model_name,
        "RMSE": round(rmse, 4),
        "MAE" : round(mae , 4),
        "R2"  : round(r2  , 4)
    })

###############################################################################
# 2. Turbidity-only 一元线性回归
###############################################################################
results = []
X_turb  = X[['ln Turd']].values.reshape(-1, 1)
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_turb, y, test_size=0.20, random_state=42)

lr_turb = LinearRegression().fit(X_train_t, y_train_t)
y_pred_turb = lr_turb.predict(X_test_t)
evaluate("Turbidity-only LR", y_test_t, y_pred_turb, results)

###############################################################################
# 3. Multi-feature baselines
###############################################################################
scaler = StandardScaler()
numeric_preproc = ColumnTransformer(
    [("scaler", scaler, FEATURE_COLS)], remainder="drop")

# 3.1 全特征线性回归
lr_pipe = Pipeline([
    ("prep", numeric_preproc),
    ("reg",  LinearRegression())
]).fit(X_train, y_train)
evaluate("Linear Regression", y_test, lr_pipe.predict(X_test), results)

# 3.2 Random Forest
rf_pipe = Pipeline([
    ("prep", numeric_preproc),
    ("rf",   RandomForestRegressor(
              n_estimators=200, max_depth=None,
              random_state=42, n_jobs=-1))
]).fit(X_train, y_train)
evaluate("Random Forest", y_test, rf_pipe.predict(X_test), results)

# 3.3 XGBoost
xgb = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror',
        random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
evaluate("XGBoost", y_test, xgb.predict(X_test), results)

# 3.4 MLP 两层
mlp_pipe = Pipeline([
    ("prep", numeric_preproc),
    ("mlp",  MLPRegressor(hidden_layer_sizes=(128, 64),
                          activation='relu',
                          solver='adam',
                          learning_rate_init=1e-3,
                          max_iter=800,
                          random_state=42))
]).fit(X_train, y_train)
evaluate("MLP-2L", y_test, mlp_pipe.predict(X_test), results)

###############################################################################
# 5. 保存结果
###############################################################################
results_df = pd.DataFrame(results)
results_df.sort_values("RMSE", inplace=True)
results_df.to_csv("A.a baseline_cross_results.csv", index=False)
print(results_df)
