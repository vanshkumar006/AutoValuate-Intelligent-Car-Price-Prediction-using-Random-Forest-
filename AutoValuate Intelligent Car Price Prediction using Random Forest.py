import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import statsmodels.api as sm
import joblib
import debugpy
debugpy.debug_this_thread()


"""
We are going to build a Random Forest model
on data obtained by omitting rows with any missing value
"""
# Reading CSV file

try:  # (error handling)
    cars_omit_data=pd.read_csv('omitted_data_cars_sampled.csv')
except FileNotFoundError:
    print("CSV file not found. Please check the filename.")
    exit()

# Dropping the first column
if 'Unnamed: 0' in cars_omit_data.columns:
    cars_omit_data=cars_omit_data.drop('Unnamed: 0',axis=1)

# Check for missing data
if cars_omit_data.isnull().sum().sum() > 0:
    print("There are still missing values in the dataset!")
    exit()

# Separating input and output features

x1 = cars_omit_data.drop(['price','model','brand'], axis='columns', inplace=False)
x1 = pd.get_dummies(x1,drop_first=True)

y1 = cars_omit_data.filter(['price'], axis=1).values.ravel()
y2 = np.log(y1)

# Splitting data into test and train
X_train, X_test, y_train_log, y_test_log = train_test_split(x1, y2, test_size=0.3, random_state = 3)
print(X_train.shape, X_test.shape, y_train_log.shape, y_test_log.shape)

def rmse_log(test_y, predicted_y):
    t1 = np.exp(test_y)
    t2 = np.exp(predicted_y)
    rmse_test = np.sqrt(mean_squared_error(t1, t2))
    #for base rmse
    base_pred = np.repeat(np.mean(t1), len(t1))
    rmse_base = np.sqrt(mean_squared_error(t1, base_pred))
    values = {'RMSE-test from model': rmse_test, 'Base RMSE': rmse_base}
    return values

rf = RandomForestRegressor(n_estimators = 220,max_depth=87, random_state=3)
#rf = RandomForestRegressor(max_depth=10)

# Model
model_rf1 = rf.fit(X_train, y_train_log)

# Predicting model on test and train set
cars_predictions_rf1_test = rf.predict(X_test)
cars_predictions_rf1_train = rf.predict(X_train)

# RMSE
rmse_test_result = rmse_log(y_test_log, cars_predictions_rf1_test)
print("Test RMSE and base:", rmse_test_result)
#  Train RMSE
rmse_train_result = rmse_log(y_train_log, cars_predictions_rf1_train)
print("Train RMSE and base:", rmse_train_result)

# Rsquared
train_r2 = model_rf1.score(X_train, y_train_log)
test_r2 = model_rf1.score(X_test, y_test_log)
print("Train R2:", train_r2)
print("Test R2:", test_r2)

# ---------------------- Feature Importance and Plots ----------------------

importances = model_rf1.feature_importances_
features = X_train.columns
indices = np.argsort(importances)[::-1][:10] # Top 10 features
plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices][::-1])
plt.yticks(range(len(indices)), features[indices][::-1])
plt.xlabel("Feature Importance")
plt.title("Top 10 Important Features (Random Forest)")
plt.show()

# ---------------------- Residuals and Actual vs Predicted ----------------------
residuals = np.exp(y_test_log) - np.exp(cars_predictions_rf1_test)
plt.figure(figsize=(8,6))
plt.scatter(np.exp(cars_predictions_rf1_test), residuals, alpha=0.5)
plt.xlabel("Predicted Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot (Random Forest)")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(np.exp(y_test_log), np.exp(cars_predictions_rf1_test), alpha=0.5)
plt.plot([min(np.exp(y_test_log)), max(np.exp(y_test_log))], [min(np.exp(y_test_log)), max(np.exp(y_test_log))], color='red', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (Random Forest)")
plt.show()

"""
Hyperparameter Tuning
"""

## Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 600,num = 15)]
print(n_estimators)

## Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 10)]

## Minimum number of samples required to split a node
min_samples_split = np.arange(100,1100,100)

## Create the random grid
random_grid1 = {'n_estimators': n_estimators}
random_grid2 = {'max_depth': max_depth}
random_grid3 = {'min_samples_split': min_samples_split}

random_grid = {'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split}

print(random_grid)

## Use the random grid to search for best hyperparameters

## First create the base model to tune
rf_for_tuning = RandomForestRegressor(random_state=3)

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf_for_tuning,
                            param_distributions = random_grid,
                            n_iter = 100, cv = 3, verbose=2, random_state=1)

## Fit the random search model
rf_random.fit(X_train, y_train_log)
print(rf_random.best_params_)

## finding the best model
rf_model_best = rf_random.best_estimator_
print(rf_model_best)

# predicting with the test data on best model
predictions_best = rf_model_best.predict(X_test)
predictions_best_train = rf_model_best.predict(X_train)

# Evaluate tuned model
print("Tuned Model Test RMSE and base:", rmse_log(y_test_log, predictions_best))
print("Tuned Model Train RMSE and base:", rmse_log(y_train_log, predictions_best_train))
print("Tuned Model Train R2:", rf_model_best.score(X_train, y_train_log))
print("Tuned Model Test R2:", rf_model_best.score(X_test, y_test_log))

# Save model and columns
joblib.dump(rf_model_best, "best_random_forest_model.pkl")
joblib.dump(X_train.columns, "model_feature_columns.pkl")
print("Model and features saved for deployment.")
