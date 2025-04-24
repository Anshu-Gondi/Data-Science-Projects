import os
import pandas as pd
import joblib
from scripts.data_preprocessing import preprocess_data, load_data
from scripts.outlier_removal import remove_outliers_zscore, remove_outliers_iqr
from scripts.correlation_analysis import plot_correlation_matrix
from scripts.train_test_split import split_data  
from scripts.model_training import train_decision_tree
from scripts.model_evaluation import evaluate_model

# Load and Preprocess Data
df = load_data("data/cars.csv")
df = preprocess_data(df)
df.to_csv("data/cleaned_data.csv", index=False)

# Remove Outliers using Z-score for specific columns
df = remove_outliers_zscore(df, ['year', 'mileage(kilometers)', 'volume(cm3)'])

# Remove Outliers using IQR for 'priceUSD'
df = remove_outliers_iqr(df, 'priceUSD')

# Save cleaned data
df.to_csv("data/cleaned_data.csv", index=False)

# Correlation Analysis
plot_correlation_matrix(df)

# Train-Test Split
X_train, X_test, y_train, y_test = split_data(df, "priceUSD")

# Train Model
model = train_decision_tree(X_train, y_train)
joblib.dump(model, "models/decision_tree_model.pkl")

# Evaluate Model
evaluate_model(model, X_test, y_test)

print("🚀 Data Science Pipeline Completed!")
