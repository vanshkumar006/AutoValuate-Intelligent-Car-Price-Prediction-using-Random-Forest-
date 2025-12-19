# AutoValuate-Intelligent-Car-Price-Prediction-using-Random-Forest-
A robust Random Forest Regressor for car price prediction. Features log-transformed targets, rigorous hyperparameter tuning via RandomizedSearchCV, and comprehensive error analysis with feature importance plots.


Project Description:
This project implements a Machine Learning pipeline to predict used car prices based on features like model, age, and specifications. The core utilizes a Random Forest Regressor within a Scikit-Learn workflow.
To handle the high variance in car prices, the target variable was log-transformed before training. A key focus of this project was solving overfitting. The initial model showed a massive gap between training and testing performance. By employing RandomizedSearchCV, I optimized the hyperparameters to create a generalized model that performs consistently on unseen data. The final model and feature columns are serialized using joblib for deployment.


Libraries & Modules Used
Data Processing: pandas, numpy (Log/Exp transformations)
Modeling: sklearn.ensemble.RandomForestRegressor
Optimization: sklearn.model_selection.RandomizedSearchCV, train_test_split
Evaluation: sklearn.metrics (MSE, RMSE), statsmodels
Visualization: matplotlib.pyplot, seaborn
Deployment: joblib


Project Results
The model underwent two phases of evaluation. The hyperparameter tuning significantly improved the model's stability.

1. Base Model (Overfitting):
Train R2R2: 96.47% (Model memorized the data)
Test R2R2: 81.04%
Observation: The high discrepancy (approx 15%) indicated severe overfitting.

2. Hyperparameter Tuning (RandomizedSearchCV):
We tested 100 combinations across 3 folds.
Best Parameters Found:
n_estimators: 52
min_samples_split: 100
max_depth: 110

3. Optimized Model Performance (Final Result):
Train R2R2: 84.96%
Test  R2R2 : 81.78%
Test RMSE: 4269.22

Observation: The gap between Train and Test scores closed significantly (less than 3.2% difference). The model is now robust and generalizes well to new data.

Conclusion:
While the base Random Forest model provided low error on training data, it failed to generalize effectively. By enforcing a higher min_samples_split (100) and optimizing the tree depth via RandomizedSearchCV, we successfully eliminated overfitting. The final model achieves a reliable 81.78% accuracy (
R2R2) on the test set, making it a viable tool for real-world price estimation. The pipeline is fully automated and saves the optimized model for immediate deployment.



