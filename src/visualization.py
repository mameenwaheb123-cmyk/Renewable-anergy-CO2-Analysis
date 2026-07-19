import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_average_rec(df):
    avg_rec = df.groupby("CN")["REC"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    avg_rec.plot(kind="bar", color="forestgreen", edgecolor="black", alpha=0.7)
    plt.title("Average REC by Country (2015–2024)", fontsize=14, fontweight="bold")
    plt.xlabel("Country", fontsize=12)
    plt.ylabel("Renewable Energy Consumption (%)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_average_co2(df):
    avg_co2 = df.groupby("CN")["CO2"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    avg_co2.plot(kind="bar", color="firebrick", edgecolor="black", alpha=0.7)
    plt.title("Average CO₂ Emissions by Country (2015–2024)", fontsize=14, fontweight="bold")
    plt.xlabel("Country", fontsize=12)
    plt.ylabel("CO₂ Emissions (Metric Tons per Capita)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_average_gdp(df):
    avg_gdp = df.groupby("CN")["GDP"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    avg_gdp.plot(kind="bar", color="steelblue", edgecolor="black", alpha=0.7)
    plt.title("Average GDP Growth by Country (2015–2024)", fontsize=14, fontweight="bold")
    plt.xlabel("Country", fontsize=12)
    plt.ylabel("GDP Growth (%)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rec_vs_co2(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df["REC"], df["CO2"], color="darkgreen", alpha=0.6, s=100, edgecolors="black", linewidth=0.5)

    z = np.polyfit(df["REC"], df["CO2"], 1)
    p = np.poly1d(z)
    plt.plot(df["REC"], p(df["REC"]), "r--", linewidth=2, label="Trend Line")

    plt.title("REC vs CO₂ Emissions", fontsize=14, fontweight="bold")
    plt.xlabel("Renewable Energy Consumption (%)", fontsize=12)
    plt.ylabel("CO₂ Emissions (Metric Tons per Capita)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, linewidths=0.5)
    plt.title("Correlation Matrix: REC, GDP, and CO₂ Emissions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="blue", alpha=0.7, s=90, edgecolors="black", linewidth=0.5)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Fit")

    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "g-", linewidth=2, label="Model Fit")

    plt.xlabel("Actual CO₂ Emissions", fontsize=12)
    plt.ylabel("Predicted CO₂ Emissions", fontsize=12)
    plt.title("Actual vs Predicted CO₂ Emissions", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(comparison):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(comparison["Residual"], bins=10, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(comparison["Residual"].mean(), color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Residual Value", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("Distribution of Residuals", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].scatter(comparison["Predicted"], comparison["Residual"], color="green", alpha=0.6, s=100, edgecolors="black", linewidth=0.5)
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Predicted CO₂ Emissions", fontsize=11)
    axes[1].set_ylabel("Residuals", fontsize=11)
    axes[1].set_title("Residuals vs Predicted Values", fontsize=12, fontweight="bold")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()