from pathlib import Path
import pandas as pd


def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: Could not parse the file.")
        return None


def prepare_data(df):
    if df is None:
        return None

    df = df.copy()

    for col in ["REC", "GDP", "CO2", "YR"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["REC", "GDP", "CO2", "YR"]).reset_index(drop=True)
    df = df.sort_values(["CN", "YR"]).reset_index(drop=True)

    return df


def load_data(file_path=None):
    if file_path is not None:
        path = Path(file_path)
        if path.exists():
            return prepare_data(load_csv(path))
        print(f"Error: The file {path} was not found.")
        return None

    base_folder = Path(__file__).resolve().parents[1]

    candidate_paths = [
        Path(r"D:\DATA\Project1.csv"),
        base_folder / "data" / "raw" / "Renewable_Energy_CO2_Clean.csv",
        base_folder / "Cleaned data" / "Renewable_Energy_CO2_Clean.csv",
        Path.cwd() / "data" / "raw" / "Renewable_Energy_CO2_Clean.csv",
        Path.cwd() / "Cleaned data" / "Renewable_Energy_CO2_Clean.csv",
    ]

    for path in candidate_paths:
        if path.exists():
            df = load_csv(path)
            return prepare_data(df)

    print("Error: Dataset not found in any expected location.")
    return None
