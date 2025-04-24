import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set,
    prints key metrics, saves feature importance,
    and plots a feature importance bar chart.
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Print the evaluation metrics
    print(f"✅ R2 Score: {r2:.4f}")
    print(f"✅ Mean Squared Error: {mse:.4f}")
    print(f"✅ Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"✅ Root Mean Squared Error: {rmse:.4f}")

    # Print accuracy as a percentage using R2 Score
    accuracy_percent = r2 * 100
    print(f"Model Accuracy: {accuracy_percent:.2f}%")

    # Feature importance
    feat_importance = np.array(model.feature_importances_)
    sorted_indices = np.argsort(feat_importance)[::-1]  # Descending order

    feat_df = pd.DataFrame({
        'Feature': X_test.columns[sorted_indices],
        'Importance': feat_importance[sorted_indices]
    })

    # Save feature importance
    feat_df.to_csv('data/feature_importance.csv', index=False)

    # Plot feature importance
    sns.set_style('darkgrid')
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feat_df)
    plt.title('Feature Importance')
    plt.show()
