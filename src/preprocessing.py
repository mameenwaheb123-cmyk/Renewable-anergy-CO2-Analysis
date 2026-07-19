import pandas as pd

try:
    from .data_loading import load_data as load_dataset
except ImportError:  # pragma: no cover
    from data_loading import load_data as load_dataset


def clean_data(df):
    """
    Match the notebook cleaning steps:
    - convert numeric columns to numeric
    - drop rows missing REC, GDP, CO2, or YR
    - sort by country and year
    """
    if df is None:
        return None

    df = df.copy()

    for col in ["REC", "GDP", "CO2", "YR"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["REC", "GDP", "CO2", "YR"]).reset_index(drop=True)
    df = df.sort_values(["CN", "YR"]).reset_index(drop=True)

    return df


def identify_negative_gdp(df):
    """Return rows with negative GDP values, as checked in the notebook."""
    if df is None:
        return pd.DataFrame()
    return df[df["GDP"] < 0]


def add_group_column(df):
    """Create the developed/developing classification used in the notebook."""
    if df is None:
        return None

    df = df.copy()
    developed_countries = ["Aus", "Can", "UK", "USA"]
    df["Group"] = df["CN"].apply(
        lambda x: "Developed" if x in developed_countries else "Developing"
    )
    return df


def normalize_data(df):
    """Compatibility helper; the notebook does not normalize values."""
    return df


def preprocess_data(file_path=None):
    """Full preprocessing flow matching the notebook."""
    df = load_dataset(file_path)
    df = clean_data(df)
    df = add_group_column(df)
    df = normalize_data(df)
    return df
