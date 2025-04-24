import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_zscore(df, cols, threshold=3):
    """Remove outliers using Z-score method for specified columns."""
    z = np.abs(stats.zscore(df[cols]))
    df_clean = df[(z < threshold).all(axis=1)]
    return df_clean

def remove_outliers_iqr(df, column):
    """Remove outliers using the IQR method for a specified column."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]
    return df_clean

if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv("data/preprocessed_data.csv")

    # Remove outliers using Z-score for specific columns
    df = remove_outliers_zscore(df, ['year', 'mileage(kilometers)', 'volume(cm3)'])

    # Remove outliers using IQR for 'priceUSD'
    df = remove_outliers_iqr(df, 'priceUSD')

    # Save cleaned data (after outlier removal)
    df.to_csv("data/cleaned_data.csv", index=False)

    print("✅ Outliers removed successfully! Saved as 'data/cleaned_data.csv'.")
    print(df.head())
