# Thames Pollution Detection IoT - ML Model

## Author
John Wu, Tommy Xin, Flex Fan

## Project Overview
This project aims to detect pollution levels in the Thames River using IoT devices and machine learning models with inorganic sensors. The data is collected from various stations along the river and processed to identify pollution indicators such as Coliform and E.coli.

## File Descriptions
### Data Processing
- **0download_data.py**: Downloads raw data from the government water quality website.
- **1find Coli.py**: Processes CSV files from 2000 to 2025 to filter and retain data related to Coliform and E.coli. From **23GB** raw data to **1。2GB**.
- **2.0extract_entry_with_Coli.py**: Extracts entries containing Coliform data from the **1.2GB** file.
- **2.1LLM_filter_columns.py**: Filters columns using a language model to identify relevant features.
- **2.2extract_specific_columns_datas.py**: Extracts rows with specific columns from the input CSV file. From **1.2GB** data to **284MB**.
- **2.3rearrange_columns.py**: Rearranges columns in the extracted data and writes to a new CSV file. From **284MB** data to **25MB**.
- **2.5filter_data.py**: Filters data where values are less than 4, leaving 5,460 entries. From **25MB** data to **610KB**.
- **2.7sincos_trans_time.py**: Applies sine and cosine transformations to time-related data for feature engineering.
- **2.8relatiion_analysis.py**: Analyzes relationships between features and generates a visualization (`relatiion_result.png`).
- **2.9create_feature_relations.py**: Creates new features based on relationships and saves the transformed data.

### Model Training and Prediction
- **3.1train_nn_pytorch.py**: Trains a neural network model using PyTorch. Outputs include `loss.png`, `min_loss.png`, and the trained model (`model_loss_0.8230@397_epoch.pth`).
- **3.2predict_nn_pytorch.py**: Uses the trained neural network to make predictions on test data (`test_data.csv`) and saves results (`test_data_with_predictions.csv`).
- **3.3xgboost_prediction.py**: Trains and evaluates an XGBoost model, generating feature importance visualization (`xgboost_importance_result.png`).

### Station-Specific Analysis
- **4.1train_nn_one_station.py**: Trains a neural network model for data from a single station (`one_station.csv`).
- **5.2xgboost_each_station.py**: Applies XGBoost to analyze feature importance for each station and saves results (`all_feature_importances.csv`).

### Utilities
- **5.1save_by_name.py**: Saves processed data by station name.
- **5.3same_feature_data.csv**: Consolidates data with the same features across stations.
- **5.4train_nn.py**: General neural network training script for consolidated data.

### Pre-trained Models and Scalers
- **scaler.pkl**: Contains the scaler used for data normalization.
- **xgboost_model.json**: Pre-trained XGBoost model.

## Usage
1. Download all raw data using `0download_data.py`.
2. Run the scripts in the following order for data processing:
   - `1find Coli.py`
   - `2.0extract_entry_with_Coli.py`
   - `2.1LLM_filter_columns.py`
   - `2.2extract_specific_columns_datas.py`
   - `2.3rearrange_columns.py`
   - `2.5filter_data.py`
   - `2.7sincos_trans_time.py`
   - `2.8relatiion_analysis.py`
   - `2.9create_feature_relations.py`
3. Train models using:
   - `3.1train_nn_pytorch.py`
   - `3.3xgboost_prediction.py`
4. Make predictions using:
   - `3.2predict_nn_pytorch.py`
5. For station-specific analysis, use:
   - `4.1train_nn_one_station.py`
   - `5.2xgboost_each_station.py`

## Requirements
- Python 3.x
- pandas
- PyTorch
- XGBoost
- matplotlib

Install the required packages using:
```bash
pip install pandas torch xgboost matplotlib
```

## License
This project is licensed under the MIT License.
