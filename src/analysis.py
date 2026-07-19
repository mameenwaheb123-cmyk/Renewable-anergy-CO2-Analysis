import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

try:
    from .preprocessing import add_group_column
except ImportError:  # pragma: no cover
    from preprocessing import add_group_column


def get_data_quality_summary(df):
    """Return a summary of data quality checks matching the notebook."""
    return {
        "missing_values": df.isnull().sum(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes,
        "year_range": {
            "min": int(df["YR"].min()),
            "max": int(df["YR"].max()),
        },
        "negative_gdp_rows": int((df["GDP"] < 0).sum()),
    }


def get_dataset_summary(df):
    """Return the notebook-style dataset summary."""
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "countries": sorted(df["CN"].unique().tolist()),
        "years": [int(df["YR"].min()), int(df["YR"].max())],
    }


def perform_statistical_analysis(df):
    """Perform descriptive statistics and correlation analysis."""
    numeric_cols = ["REC", "GDP", "CO2"]

    descriptive_stats = df[numeric_cols].describe().round(2)
    correlation_matrix = df[numeric_cols].corr().round(2)
    quantiles = df[numeric_cols].quantile([0.25, 0.50, 0.75]).round(2)

    return {
        "descriptive_stats": descriptive_stats,
        "correlation_matrix": correlation_matrix,
        "quantiles": quantiles,
        "mean_rec": round(float(df["REC"].mean()), 2),
        "mean_gdp": round(float(df["GDP"].mean()), 2),
        "mean_co2": round(float(df["CO2"].mean()), 2),
        "std_rec": round(float(df["REC"].std()), 2),
        "std_gdp": round(float(df["GDP"].std()), 2),
        "std_co2": round(float(df["CO2"].std()), 2),
    }


def generate_country_summary(df):
    """Generate average REC, GDP, and CO2 values by country."""
    return (
        df.groupby("CN")[["REC", "GDP", "CO2"]]
        .mean()
        .round(2)
        .reset_index()
    )


def analyze_trends(df):
    """Analyze yearly trends in REC, GDP, and CO2."""
    return (
        df.groupby("YR")[["REC", "GDP", "CO2"]]
        .mean()
        .round(2)
        .reset_index()
    )


def compare_developed_developing(df):
    """Compare developed and developing countries using the notebook logic."""
    grouped_df = add_group_column(df.copy())
    developed = grouped_df[grouped_df["Group"] == "Developed"].copy()
    developing = grouped_df[grouped_df["Group"] == "Developing"].copy()

    return {
        "developed_summary": (
            developed.groupby("CN")[["REC", "GDP", "CO2"]]
            .mean()
            .round(2)
            .reset_index()
        ),
        "developing_summary": (
            developing.groupby("CN")[["REC", "GDP", "CO2"]]
            .mean()
            .round(2)
            .reset_index()
        ),
        "developed_stats": {
            "count": len(developed),
            "mean_rec": round(float(developed["REC"].mean()), 2),
            "mean_co2": round(float(developed["CO2"].mean()), 2),
        },
        "developing_stats": {
            "count": len(developing),
            "mean_rec": round(float(developing["REC"].mean()), 2),
            "mean_co2": round(float(developing["CO2"].mean()), 2),
        },
    }


def train_regression_model(df):
    """Train a linear regression model using REC and GDP to predict CO2."""
    X = df[["REC", "GDP"]]
    y = df["CO2"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": {
            "mae": round(float(mae), 3),
            "mse": round(float(mse), 3),
            "rmse": round(float(rmse), 3),
            "r2": round(float(r2), 3),
        },
        "coefficients": {
            "REC": round(float(model.coef_[0]), 3),
            "GDP": round(float(model.coef_[1]), 3),
            "intercept": round(float(model.intercept_), 3),
        },
    }


def fit_ols_regression(df):
    """Return OLS regression results for the notebook-style comparison."""
    X = sm.add_constant(df[["REC", "GDP"]])
    y = df["CO2"]
    model = sm.OLS(y, X).fit()

    return {
        "model": model,
        "summary_text": model.summary().as_text(),
        "r2": round(float(model.rsquared), 3),
        "f_pvalue": round(float(model.f_pvalue), 3),
        "coefficients": {
            "const": round(float(model.params[0]), 3),
            "REC": round(float(model.params[1]), 3),
            "GDP": round(float(model.params[2]), 3),
        },
    }


def create_residual_summary(y_test, y_pred):
    """Create a DataFrame with actual values, predicted values, residuals, and absolute errors."""
    comparison = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
    })

    comparison["Residual"] = comparison["Actual"] - comparison["Predicted"]
    comparison["Absolute Error"] = comparison["Residual"].abs()

    return comparison.sort_values("Absolute Error", ascending=False).reset_index(drop=True)


def run_full_analysis(df):
    """Run the full analysis workflow and return all results in one dictionary."""
    regression_result = train_regression_model(df)

    return {
        "data_quality": get_data_quality_summary(df),
        "dataset_summary": get_dataset_summary(df),
        "statistics": perform_statistical_analysis(df),
        "country_summary": generate_country_summary(df),
        "yearly_trends": analyze_trends(df),
        "developed_vs_developing": compare_developed_developing(df),
        "regression": regression_result,
        "residuals": create_residual_summary(
            regression_result["y_test"],
            regression_result["y_pred"],
        ),
        "ols_regression": fit_ols_regression(df),
    }
