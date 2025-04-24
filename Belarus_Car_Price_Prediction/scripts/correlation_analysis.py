import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(df):
    """Plot correlation heatmap"""
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix Heatmap")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv")
    plot_correlation_matrix(df)
