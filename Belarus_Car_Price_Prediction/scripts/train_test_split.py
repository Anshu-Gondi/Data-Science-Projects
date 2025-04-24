import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df, target_column, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Save the split data
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print("✅ Train-test split completed and saved.")
    return X_train, X_test, y_train, y_test
