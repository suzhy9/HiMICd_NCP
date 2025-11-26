# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from sklearn import metrics
import os
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

# %% STEP ONE: DATA INPUT
# =============================================================================
# Data Input and Preprocessing Module
# =============================================================================
# Load and clean training data
training_data_path = r"E:\high-resolution atmospheric moisture\Data\2 Split Data\Trainsample.csv"
raw_training_data = pd.read_csv(training_data_path)

# Remove unnamed columns and filter invalid values
unnamed_columns = raw_training_data.filter(regex="Unname").columns
cleaned_training_data = raw_training_data.drop(columns=unnamed_columns)
cleaned_training_data = cleaned_training_data[~cleaned_training_data.isin([-9999]).any(axis=1)]

training_dataset = cleaned_training_data
print("Training dataset shape:", training_dataset.shape)

# =============================================================================
# Feature Selection and Target Variable Configuration
# =============================================================================
# Configuration parameters
target_year = 2003  # Range: 2003 to 2020
target_variable = "AVP"  # Available indices: ['AVP', 'DPT', 'MR', 'RH', 'SH', 'VPD']

# Filter data for target year
year_filtered_data = training_dataset.loc[training_dataset['year'] == target_year]

# Prepare feature matrix
feature_matrix = year_filtered_data.iloc[:, 1:8].copy()

# Map target variables to corresponding columns
variable_column_mapping = {
    "AVP": 8, "DPT": 9, "MR": 10, "RH": 11, "SH": 12, "VPD": 13
}

# Add corresponding feature column based on target variable
if target_variable in variable_column_mapping:
    feature_column_name = target_variable.lower()
    feature_matrix[feature_column_name] = year_filtered_data [feature_column_name]

# Prepare final training sets
feature_matrix[target_variable] = year_filtered_data[target_variable]
X_train = feature_matrix.iloc[:, 0:8]
y_train = feature_matrix[target_variable]

print(f"Feature matrix shape: {X_train.shape}")
print(f"Target variable shape: {y_train.shape}")


# %% STEP TWO: HYPERPARAMETER OPTIMIZATIOM
# =============================================================================
# Hyperparameter Optimization - Phase 1: Base Parameters
# =============================================================================
# Step 1: Optimize n_estimators
print("Starting n_estimators optimization...")
step1_start = time.time()

base_config = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate': 0.1,
    'num_leaves': 50,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

lgb_dataset = lgb.Dataset(X_train, y_train)

cv_output = lgb.cv(
    base_config,
    lgb_dataset,
    num_boost_round=30000,
    nfold=5,
    stratified=False,
    shuffle=True,
    metrics='rmse',
    seed=0,
    callbacks=[
        lgb.early_stopping(500),
        lgb.log_evaluation(period=100)
    ]
)

optimal_n_estimators = len(cv_output['valid rmse-mean'])
best_cv_rmse = cv_output['valid rmse-mean'][-1]

step1_end = time.time()
print(f'Optimal n_estimators: {optimal_n_estimators}')
print(f'Best CV RMSE: {best_cv_rmse:.6f}')
print(f'Step 1 duration: {step1_end - step1_start:.2f} seconds')

# =============================================================================
# Hyperparameter Optimization - Phase 2: Tree Structure
# =============================================================================
# Step 2: Optimize max_depth and num_leaves
print("Starting tree structure optimization...")
step2_start = time.time()

lgb_model_1 = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    num_leaves=50,
    learning_rate=0.1,
    n_estimators=optimal_n_estimators,
    metric='rmse',
    bagging_fraction=0.8,
    feature_fraction=0.8
)

depth_leaves_grid = {
    'max_depth': range(2, 11, 3),
    'num_leaves': range(10, 150, 30)
}

valid_combinations = [
    param for param in ParameterGrid(depth_leaves_grid)
    if param['num_leaves'] <= 2 ** param['max_depth']
]

filtered_grid = {
    'max_depth': sorted(set(p['max_depth'] for p in valid_combinations)),
    'num_leaves': sorted(set(p['num_leaves'] for p in valid_combinations))
}

grid_search_1 = GridSearchCV(
    estimator=lgb_model_1,
    param_grid=filtered_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search_1.fit(X_train, y_train)
optimal_max_depth = grid_search_1.best_params_["max_depth"]
optimal_num_leaves = grid_search_1.best_params_["num_leaves"]

step2_end = time.time()
print(f"Optimal tree parameters: max_depth={optimal_max_depth}, num_leaves={optimal_num_leaves}")
print(f"Step 2 duration: {step2_end - step2_start:.2f} seconds")

# Step 2a: Refine tree parameters
print("Refining tree parameters...")
step2a_start = time.time()

lgb_model_2 = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    learning_rate=0.1,
    n_estimators=optimal_n_estimators,
    metric='rmse',
    bagging_fraction=0.8,
    feature_fraction=0.8
)

refined_tree_grid = {
    'max_depth': [optimal_max_depth - 1, optimal_max_depth, optimal_max_depth + 1],
    'num_leaves': [optimal_num_leaves - 1, optimal_num_leaves, optimal_num_leaves + 1],
}

grid_search_2 = GridSearchCV(
    estimator=lgb_model_2,
    param_grid=refined_tree_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search_2.fit(X_train, y_train)
optimal_max_depth = grid_search_2.best_params_["max_depth"]
optimal_num_leaves = grid_search_2.best_params_["num_leaves"]

step2a_end = time.time()
print(f"Refined tree parameters: max_depth={optimal_max_depth}, num_leaves={optimal_num_leaves}")
print(f"Step 2a duration: {step2a_end - step2a_start:.2f} seconds")

# =============================================================================
# Hyperparameter Optimization - Phase 3: Leaf Parameters
# =============================================================================
# Step 3: Optimize leaf parameters
print("Optimizing leaf parameters...")
step3_start = time.time()

leaf_parameters_grid = {
    'min_child_samples': [5, 10, 15, 20],
    'min_child_weight': [0.001, 0.002, 0.01, 0.1]
}

lgb_model_3 = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    n_estimators=optimal_n_estimators,
    num_leaves=optimal_num_leaves,
    max_depth=optimal_max_depth,
    metric='rmse',
    bagging_fraction=0.8,
    feature_fraction=0.8,
    learning_rate=0.1
)

grid_search_3 = GridSearchCV(
    estimator=lgb_model_3,
    param_grid=leaf_parameters_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search_3.fit(X_train, y_train)
optimal_min_child_samples = grid_search_3.best_params_["min_child_samples"]
optimal_min_child_weight = grid_search_3.best_params_["min_child_weight"]

step3_end = time.time()
print(f"Optimal leaf parameters: min_child_samples={optimal_min_child_samples}, min_child_weight={optimal_min_child_weight}")
print(f"Step 3 duration: {step3_end - step3_start:.2f} seconds")

# Step 3a: Refine leaf samples
print("Refining leaf samples...")
step3a_start = time.time()

refined_leaf_grid = {
    'min_child_samples': [
        optimal_min_child_samples - 2, optimal_min_child_samples - 1,
        optimal_min_child_samples, optimal_min_child_samples + 1,
        optimal_min_child_samples + 2
    ],
    'min_child_weight': [optimal_min_child_weight]
}

lgb_model_3a = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    n_estimators=optimal_n_estimators,
    num_leaves=optimal_num_leaves,
    max_depth=optimal_max_depth,
    metric='rmse',
    learning_rate=0.1,
    bagging_fraction=0.8,
    feature_fraction=0.8
)

grid_search_3a = GridSearchCV(
    estimator=lgb_model_3a,
    param_grid=refined_leaf_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search_3a.fit(X_train, y_train)
optimal_min_child_samples = grid_search_3a.best_params_["min_child_samples"]

step3a_end = time.time()
print(f"Refined min_child_samples: {optimal_min_child_samples}")
print(f"Step 3a duration: {step3a_end - step3a_start:.2f} seconds")

# =============================================================================
# Hyperparameter Optimization - Phase 4: Fraction Parameters
# =============================================================================
# Step 4: Optimize fraction parameters
print("Optimizing fraction parameters...")
step4_start = time.time()

fraction_parameters_grid = {
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.8, 1.0]
}

lgb_model_4 = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    n_estimators=optimal_n_estimators,
    num_leaves=optimal_num_leaves,
    max_depth=optimal_max_depth,
    min_child_samples=optimal_min_child_samples,
    min_child_weight=optimal_min_child_weight,
    metric='rmse',
    learning_rate=0.1
)

grid_search_4 = GridSearchCV(
    estimator=lgb_model_4,
    param_grid=fraction_parameters_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search_4.fit(X_train, y_train)
optimal_feature_fraction = grid_search_4.best_params_["feature_fraction"]
optimal_bagging_fraction = grid_search_4.best_params_["bagging_fraction"]

step4_end = time.time()
print(f"Optimal fractions: feature_fraction={optimal_feature_fraction}, bagging_fraction={optimal_bagging_fraction}")
print(f"Step 4 duration: {step4_end - step4_start:.2f} seconds")

# Step 4a: Refine feature fraction
print("Refining feature fraction...")
step4a_start = time.time()

refined_feature_grid = {
    'feature_fraction': [
        optimal_feature_fraction - 0.1, optimal_feature_fraction - 0.05, optimal_feature_fraction - 0.02,
        optimal_feature_fraction, optimal_feature_fraction + 0.02, optimal_feature_fraction + 0.05,
        optimal_feature_fraction + 0.1
    ]
}

lgb_model_4a = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    n_estimators=optimal_n_estimators,
    num_leaves=optimal_num_leaves,
    max_depth=optimal_max_depth,
    min_child_samples=optimal_min_child_samples,
    min_child_weight=optimal_min_child_weight,
    bagging_fraction=optimal_bagging_fraction,
    metric='rmse',
    bagging_freq=5,
    learning_rate=0.1
)

grid_search_4a = GridSearchCV(
    estimator=lgb_model_4a,
    param_grid=refined_feature_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=4
)

grid_search_4a.fit(X_train, y_train)
optimal_feature_fraction = grid_search_4a.best_params_["feature_fraction"]

step4a_end = time.time()
print(f"Refined feature_fraction: {optimal_feature_fraction}")
print(f"Step 4a duration: {step4a_end - step4a_start:.2f} seconds")

# =============================================================================
# Hyperparameter Optimization - Phase 5: Learning Rate
# =============================================================================
# Step 5: Optimize learning rate
print("Optimizing learning rate...")
step5_start = time.time()

learning_rate_grid = {
    'learning_rate': [0.1, 0.001, 0.002, 0.005, 0.008]
}

lgb_model_5 = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    n_estimators=optimal_n_estimators,
    num_leaves=optimal_num_leaves,
    max_depth=optimal_max_depth,
    min_child_samples=optimal_min_child_samples,
    min_child_weight=optimal_min_child_weight,
    bagging_fraction=optimal_bagging_fraction,
    feature_fraction=optimal_feature_fraction,
    metric='rmse'
)

grid_search_5 = GridSearchCV(
    estimator=lgb_model_5,
    param_grid=learning_rate_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=4
)

grid_search_5.fit(X_train, y_train)
optimal_learning_rate = grid_search_5.best_params_["learning_rate"]

step5_end = time.time()
print(f"Optimal learning rate: {optimal_learning_rate}")
print(f"Step 5 duration: {step5_end - step5_start:.2f} seconds")

# =============================================================================
# Final Model Tuning
# =============================================================================
# Step 6: Final n_estimators tuning with optimized parameters
print("Performing final n_estimators tuning...")
step6_start = time.time()

final_config = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate': optimal_learning_rate,
    'num_leaves': optimal_num_leaves,
    'max_depth': optimal_max_depth,
    'min_child_samples': optimal_min_child_samples,
    'min_child_weight': optimal_min_child_weight,
    'bagging_fraction': optimal_bagging_fraction,
    'feature_fraction': optimal_feature_fraction,
}

final_dataset = lgb.Dataset(X_train, y_train, silent=True)

final_cv_output = lgb.cv(
    final_config,
    final_dataset,
    num_boost_round=50000,
    nfold=5,
    stratified=False,
    shuffle=True,
    metrics='rmse',
    callbacks=[
        lgb.early_stopping(500),
        lgb.log_evaluation(period=100)
    ],
    seed=0
)

final_n_estimators = len(final_cv_output['valid rmse-mean'])
final_rmse = final_cv_output['valid rmse-mean'][-1]

step6_end = time.time()
print(f'Final n_estimators: {final_n_estimators}')
print(f'Final CV RMSE: {final_rmse:.6f}')
print(f'Step 6 duration: {step6_end - step6_start:.2f} seconds')

# =============================================================================
# Results Summary
# =============================================================================
print("\n" + "="*50)
print("HYPERPARAMETER OPTIMIZATION RESULTS")
print("="*50)
print(f"Optimal n_estimators: {final_n_estimators}")
print(f"Optimal num_leaves: {optimal_num_leaves}")
print(f"Optimal max_depth: {optimal_max_depth}")
print(f"Optimal min_child_samples: {optimal_min_child_samples}")
print(f"Optimal min_child_weight: {optimal_min_child_weight}")
print(f"Optimal bagging_fraction: {optimal_bagging_fraction}")
print(f"Optimal feature_fraction: {optimal_feature_fraction}")
print(f"Optimal learning_rate: {optimal_learning_rate}")

total_optimization_time = step6_end - step1_start
print(f"\nTotal optimization time: {total_optimization_time:.2f} seconds")
print("="*50)


# %% STEP THREE AND FOUR: MODEL TRAINING AND PERFORMANCE EVALUATION MODULE
# =============================================================================
# Model Training and Evaluation Module
# =============================================================================
import lightgbm as lgb
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

# =============================================================================
# Configuration Parameters
# =============================================================================
TARGET_YEAR = 2003
TARGET_VARIABLE = "AVP"  # Options: ['AVP', 'DPT', 'MR', 'RH', 'SH', 'VPD']

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
# Load and clean training data
training_data_path = r"E:\high-resolution atmospheric moisture\Data\2 Split Data\Trainsample.csv"
raw_training_data = pd.read_csv(training_data_path)
training_cleaned = raw_training_data.drop(columns=raw_training_data.filter(regex="Unname").columns)
training_dataset = training_cleaned[~training_cleaned.isin([-9999]).any(axis=1)]

# Load and clean testing data
testing_data_path = r"E:\high-resolution atmospheric moisture\Data\2 Split Data\Testsample.csv"
raw_testing_data = pd.read_csv(testing_data_path)
testing_cleaned = raw_testing_data.drop(columns=raw_testing_data.filter(regex="Unname").columns)
testing_dataset = testing_cleaned[~testing_cleaned.isin([-9999]).any(axis=1)]

# =============================================================================
# Feature Engineering for Training Data
# =============================================================================
# Filter data for target year
training_year_filtered = training_dataset.loc[training_dataset['year'] == TARGET_YEAR]

# Prepare training feature matrix
training_features = training_year_filtered.iloc[:, 1:8].copy()

# Map target variables to feature columns
variable_column_map = {
    "AVP": 8, "DPT": 9, "MR": 10, "RH": 11, "SH": 12, "VPD": 13
}

# Add target-specific feature column
if TARGET_VARIABLE in variable_column_map:
    feature_col_name = TARGET_VARIABLE.lower()
    training_features[feature_col_name] = training_year_filtered[feature_col_name]

# Finalize training datasets
training_features[TARGET_VARIABLE] = training_year_filtered[TARGET_VARIABLE]
X_train = training_features.iloc[:, 0:8]
y_train = training_features[TARGET_VARIABLE]

# =============================================================================
# Feature Engineering for Testing Data
# =============================================================================
# Filter data for target year
testing_year_filtered = testing_dataset.loc[testing_dataset['year'] == TARGET_YEAR]

# Prepare testing feature matrix
testing_features = testing_year_filtered.iloc[:, 1:8].copy()

# Add target-specific feature column
if TARGET_VARIABLE in variable_column_map:
    col_idx = variable_column_map[TARGET_VARIABLE]
    feature_col_name = TARGET_VARIABLE.lower()
    testing_features[feature_col_name] = testing_year_filtered[feature_col_name]

# Finalize testing datasets
testing_features[TARGET_VARIABLE] = testing_year_filtered[TARGET_VARIABLE]
X_test = testing_features.iloc[:, 0:8]
y_test = testing_features[TARGET_VARIABLE]

# =============================================================================
# Model Configuration and Training
# =============================================================================
# Optimized hyperparameters
OPTIMAL_ESTIMATORS = 100000
OPTIMAL_LEARNING_RATE = 0.05
OPTIMAL_MAX_DEPTH = 9
OPTIMAL_NUM_LEAVES = 20
OPTIMAL_MIN_CHILD_SAMPLES = 90
OPTIMAL_MIN_CHILD_WEIGHT = 0.001
OPTIMAL_BAGGING_FRACTION = 0.8
OPTIMAL_FEATURE_FRACTION = 0.6

# Initialize LightGBM model
lightgbm_model = lgb.LGBMRegressor(
    n_estimators=OPTIMAL_ESTIMATORS,
    num_leaves=OPTIMAL_NUM_LEAVES,
    max_depth=OPTIMAL_MAX_DEPTH,
    min_child_samples=OPTIMAL_MIN_CHILD_SAMPLES,
    min_child_weight=OPTIMAL_MIN_CHILD_WEIGHT,
    bagging_fraction=OPTIMAL_BAGGING_FRACTION,
    feature_fraction=OPTIMAL_FEATURE_FRACTION,
    learning_rate=OPTIMAL_LEARNING_RATE,
    verbosity=100,
    boosting_type='gbdt',
    objective='regression',
    importance_type="gain",
    n_jobs=-1,
    random_state=42
)

# Train the model
print("Training LightGBM model...")
training_start_time = time.time()
lightgbm_model.fit(X_train.values, y_train.values)
training_duration = time.time() - training_start_time
print(f"Model training completed in {training_duration:.2f} seconds")

# =============================================================================
# Model Saving
# =============================================================================
model_booster = lightgbm_model.booster_
model_save_path = f'E:\\high-resolution atmospheric moisture\\Train\\model\\trained_LGBM_model_{TARGET_VARIABLE}_{TARGET_YEAR}.txt'
model_booster.save_model(model_save_path)
print(f"Model saved to: {model_save_path}")

# =============================================================================
# Model Prediction
# =============================================================================
print("Generating predictions...")
y_pred = lightgbm_model.predict(X_test.values)

# =============================================================================
# Model Evaluation
# =============================================================================
if 'y_test' in locals():
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print("\nModel Performance Metrics:")
    print(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
    print(f"R-squared (R²): {test_r2:.4f}")

# =============================================================================
# Results Compilation and Export
# =============================================================================
# Prepare results dataframe
testing_year_filtered['id'] = testing_year_filtered['station']
results_dataframe = pd.DataFrame()
results_dataframe["id"] = testing_year_filtered["id"]
results_dataframe['lon'] = np.nan
results_dataframe['lat'] = np.nan
results_dataframe["year"] = testing_year_filtered["year"]
results_dataframe["month"] = testing_year_filtered["month"]
results_dataframe["day"] = testing_year_filtered["day"]
results_dataframe["actual_values"] = y_test

# Load station coordinate information
station_coordinates_path = r"E:\high-resolution atmospheric moisture\Data\station_info.csv"
station_data = pd.read_csv(station_coordinates_path)
unnamed_columns = station_data.filter(regex="Unname")
station_data_cleaned = station_data.drop(unnamed_columns, axis=1)
station_data_sorted = station_data_cleaned.sort_values(by=['id'], ascending=[True])

# Merge coordinates with results
for index in range(results_dataframe.shape[0]):
    station_id = results_dataframe.iloc[index:index + 1]["id"].values[0]
    station_coords = station_data_sorted.loc[station_data_sorted['id'] == station_id]

    if not station_coords.empty:
        results_dataframe.iloc[index, 1] = station_coords["lon"].values[0]
        results_dataframe.iloc[index, 2] = station_coords["lat"].values[0]

# Add prediction results
results_dataframe["predicted_values"] = y_pred
results_dataframe["prediction_error"] = results_dataframe["predicted_values"] - results_dataframe["actual_values"]

# Export results
output_file_path = f"E:\\high-resolution atmospheric moisture\\Train\\{TARGET_YEAR}\\{TARGET_VARIABLE}_{TARGET_YEAR}_test_results.csv"
results_dataframe.to_csv(output_file_path, index=False)
print(f"Results exported to: {output_file_path}")

print("\nModel training and evaluation completed successfully!")


# %% STEP FIVE: SPATIAL PREDICTION MODULE - LIGHTGBM
# =============================================================================
# Spatial Prediction Module using LightGBM
# =============================================================================
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
import time
import lightgbm as lgb
from sklearn import metrics
from osgeo import gdal
import xgboost as xgb


# =============================================================================
# Utility Functions
# =============================================================================
def find_files_by_extension(directory_path, file_extension, file_list):
    import os
    directory_contents = os.listdir(directory_path)
    directory_contents.sort()

    for item in directory_contents:
        full_path = "/".join((directory_path, item))
        if os.path.isdir(full_path):
            find_files_by_extension(full_path, file_extension, file_list)
        elif os.path.isfile(full_path) and full_path.endswith(file_extension):
            file_list.append(full_path)


def is_leap_year(input_year):
    return (input_year % 400 == 0) or (input_year % 4 == 0 and input_year % 100 != 0)


# =============================================================================
# Configuration Parameters
# =============================================================================
TARGET_YEAR = 2003
TARGET_VARIABLE = "RH"  # Options: ['AVP', 'DPT', 'MR', 'RH', 'SH', 'VPD']

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
# Load training data
training_data_path = "/public/xml/High/Split Data/2003/Trainsample.csv"
raw_training_data = pd.read_csv(training_data_path)
training_cleaned = raw_training_data.drop(columns=raw_training_data.filter(regex="Unname").columns)
training_dataset = training_cleaned[~training_cleaned.isin([-9999]).any(axis=1)]

# Load testing data
testing_data_path = "/public/xml/High/Split Data/2003/humidity/Testsample.csv"
raw_testing_data = pd.read_csv(testing_data_path)
testing_cleaned = raw_testing_data.drop(columns=raw_testing_data.filter(regex="Unname").columns)
testing_dataset = testing_cleaned[~testing_cleaned.isin([-9999]).any(axis=1)].dropna()

# Filter data for target year
training_year_data = training_dataset.loc[training_dataset['year'] == TARGET_YEAR]
testing_year_data = testing_dataset.loc[testing_dataset['year'] == TARGET_YEAR]

# Load station coordinate data
station_coordinates_path = "/public/xml/High/Split Data/point.csv"
station_data = pd.read_csv(station_coordinates_path)
unnamed_columns = station_data.filter(regex="Unname")
station_coordinates = station_data.drop(unnamed_columns, axis=1)
station_coordinates = station_coordinates.sort_values(by=['id'], ascending=True)

# Clean up temporary variables
del unnamed_columns, station_coordinates_path

# =============================================================================
# Feature Engineering
# =============================================================================
# Prepare training features
training_features = training_year_data.iloc[:, 1:8].copy()

# Map target variables to feature columns
variable_column_mapping = {
    "AVP": 8, "DPT": 9, "MR": 10, "RH": 11, "SH": 12, "VPD": 13
}

# Add target-specific feature column
if TARGET_VARIABLE in variable_column_mapping:
    feature_column_name = TARGET_VARIABLE.lower()
    training_features[feature_column_name] = training_year_data[feature_column_name]

# Finalize training datasets
training_features[TARGET_VARIABLE] = training_year_data[TARGET_VARIABLE]
X_train = training_features.iloc[:, 0:8]
y_train = training_features[TARGET_VARIABLE]

# Prepare testing features
testing_features = testing_year_data.iloc[:, 1:8].copy()

# Add target-specific feature column
if TARGET_VARIABLE in variable_column_mapping:
    column_index = variable_column_mapping[TARGET_VARIABLE]
    feature_column_name = TARGET_VARIABLE.lower()
    testing_features[feature_column_name] = testing_year_data[feature_column_name]

# Finalize testing datasets
testing_features[TARGET_VARIABLE] = testing_year_data[TARGET_VARIABLE]
testing_features = testing_features.dropna()
X_test = testing_features.iloc[:, 0:8]
y_test = testing_features[TARGET_VARIABLE]

# =============================================================================
# Results Dataframe Preparation
# =============================================================================
results_dataframe = pd.DataFrame()
results_dataframe["id"] = testing_year_data["id"]
results_dataframe['lon'] = np.nan
results_dataframe['lat'] = np.nan
results_dataframe["year"] = testing_year_data["year"]
results_dataframe["month"] = testing_year_data["month"]
results_dataframe["day"] = testing_year_data["day"]
results_dataframe["actual_values"] = y_test

# Add coordinates to results
for index in range(results_dataframe.shape[0]):
    station_id = results_dataframe.iloc[index:index + 1]["id"].values[0]
    station_info = station_coordinates.loc[station_coordinates['id'] == station_id]

    if not station_info.empty:
        results_dataframe.iloc[index, 1] = station_info["lon"].values[0]
        results_dataframe.iloc[index, 2] = station_info["lat"].values[0]

# =============================================================================
# Model Loading
# =============================================================================
model_file_path = f'/public/xml/High/model/2003/trained_LGBM_model_{TARGET_VARIABLE}.txt'
trained_model = lgb.Booster(model_file=model_file_path)
print(f"Model type: {type(trained_model)}")
print(f"Model loaded successfully from: {model_file_path}")

# =============================================================================
# Spatial Prediction Setup
# =============================================================================
# Find all input raster files
input_files_list = []
input_directory_path = f"/public/xml/High/3 Predict Data/{TARGET_YEAR}/"
find_files_by_extension(input_directory_path, "tif", input_files_list)

# Prepare output directory structure
output_base_path = f"/public/xml/High/Predict/{TARGET_YEAR}/"
os.makedirs(output_base_path, exist_ok=True)

output_variable_path = output_base_path + f"{TARGET_VARIABLE}/"
os.makedirs(output_variable_path, exist_ok=True)

# =============================================================================
# Spatial Prediction Execution
# =============================================================================
model_suffix = "_LGBM"
print(f'Starting spatial predictions with {model_suffix}...')
prediction_start_time = time.time()

# Define band selection based on target variable
band_selection_map = {
    "AVP": list(range(1, 8)) + [9],
    "DPT": list(range(1, 8)) + [11],
    "MR": list(range(1, 8)) + [12],
    "RH": list(range(1, 8)) + [8],
    "SH": list(range(1, 8)) + [13],
    "VPD": list(range(1, 8)) + [10]
}

selected_band_indices = band_selection_map[TARGET_VARIABLE]

for file_index, input_file_path in enumerate(input_files_list):
    print(f"Processing file {file_index + 1}/{len(input_files_list)}: {os.path.basename(input_file_path)}")

    # Open and read raster data
    raster_dataset = gdal.Open(input_file_path)

    # Extract selected bands
    selected_bands_data = []
    for band_index in selected_band_indices:
        band = raster_dataset.GetRasterBand(band_index)
        band_array = band.ReadAsArray()
        selected_bands_data.append(band_array)

    # Stack bands and reshape for prediction
    raster_stack = np.stack(selected_bands_data)
    raster_reshaped = raster_stack.reshape((raster_stack.shape[0], -1)).T

    # Create data mask for valid pixels
    data_mask = pd.DataFrame(raster_reshaped) != -9999.0
    valid_pixels_mask = data_mask[0]
    for column in data_mask.columns:
        valid_pixels_mask = np.bitwise_and(valid_pixels_mask, data_mask[column])

    # Handle missing values
    raster_reshaped[np.isnan(raster_reshaped)] = -9999.0
    missing_value_imputer = SimpleImputer(missing_values=-9999.0, strategy='constant', fill_value=0)
    processed_data = missing_value_imputer.fit_transform(raster_reshaped)

    processed_data_df = pd.DataFrame(processed_data)
    processed_data_df.columns = X_test.columns

    # Generate predictions
    spatial_predictions = trained_model.predict(processed_data_df)

    # Reshape predictions to original spatial dimensions
    spatial_predictions = spatial_predictions.reshape(raster_stack[1, :, :].shape)
    valid_pixels_mask = valid_pixels_mask.values.reshape(raster_stack[1, :, :].shape)

    # Apply value constraints for non-DPT variables
    if TARGET_VARIABLE != "DPT":
        spatial_predictions = np.clip(spatial_predictions, 0, 100)

    # Apply mask to predictions
    spatial_predictions = spatial_predictions * valid_pixels_mask

    # Create masked array and fill no-data values
    masked_predictions = np.ma.masked_array(data=spatial_predictions, mask=~valid_pixels_mask)
    final_predictions = masked_predictions.filled(-9999.0)

    # Prepare output file path
    input_file_directory = input_file_path.split("/")[-2]
    output_directory = output_variable_path + input_file_directory
    os.makedirs(output_directory, exist_ok=True)

    input_filename = os.path.basename(input_file_path).split(".")[0]
    output_filename = f"{output_directory}/{input_filename}{model_suffix}.tif"

    # Write output raster
    rows, cols = final_predictions.shape
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_filename, cols, rows, 1, gdal.GDT_Float32)

    # Set spatial reference and geotransform
    geotransform = raster_dataset.GetGeoTransform()
    projection = raster_dataset.GetProjection()
    output_dataset.SetGeoTransform(geotransform)
    output_dataset.SetProjection(projection)

    # Write data and set no-data value
    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(final_predictions)
    output_band.SetNoDataValue(-9999.0)

    # Clean up
    output_dataset.FlushCache()
    output_dataset = None
    raster_dataset = None

    print(f'Completed: {input_filename}')

prediction_end_time = time.time()
total_processing_time = prediction_end_time - prediction_start_time
print(f"Spatial prediction completed in {total_processing_time:.2f} seconds")
print(f"Total files processed: {len(input_files_list)}")