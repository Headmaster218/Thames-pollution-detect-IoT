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
X = data.iloc[:, [i for i in range(3, 48) if i != 8]].values  # Columns 4 to 48 excluding column 9 as input
y = data.iloc[:, 8].values  # Column 9 as output

# Add processing to set values close to zero to 0
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
# Define optimized XGBoost parameters
params = {
    'objective': 'reg:squarederror',  # Regression problem
    'colsample_bytree': 0.8,  # 80% feature sampling to reduce overfitting
    'learning_rate': 0.05,  # Lower learning rate for stability
    'max_depth': 8,  # Increase depth for more complex patterns
    'alpha': 5,  # Reduce L1 regularization to prevent excessive sparsity
    'lambda': 1,  # Add L2 regularization to improve generalization
    'subsample': 0.8,  # 80% training data sampling to prevent overfitting
}

# Train the XGBoost model with early stopping
num_boost_round = 500  # Allow up to 500 rounds of training
model = xgb.train(
    params=params, 
    dtrain=train_dmatrix, 
    num_boost_round=num_boost_round,  
    evals=[(train_dmatrix, 'train'), (test_dmatrix, 'eval')],  # Monitor training and testing sets
    early_stopping_rounds=20,  # Stop if RMSE doesn't improve for 20 rounds
    verbose_eval=10  # Print progress every 10 rounds
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
