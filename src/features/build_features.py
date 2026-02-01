import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw crime dataframe and returns model-ready features.
    """

    df = df.copy()

    # ---------------------------
    # Drop identifiers & leakage
    # ---------------------------
    drop_cols = [
        "Report Number",
        "Date Case Closed"
    ]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # ---------------------------
    # Encode target
    # ---------------------------
    if "Case Closed" in df.columns:
        df["Case Closed"] = df["Case Closed"].map({"Yes": 1, "No": 0})

    # ---------------------------
    # Handle missing values
    # ---------------------------
    df["Weapon Used"] = df["Weapon Used"].fillna("Unknown")

    # ---------------------------
    # Time features
    # ---------------------------
    # ---------------------------
# Time features
# ---------------------------
    df["Date Reported"] = pd.to_datetime(df["Date Reported"], errors="coerce")
    df["Date of Occurrence"] = pd.to_datetime(df["Date of Occurrence"], errors="coerce")
    df["Time of Occurrence"] = pd.to_datetime(df["Time of Occurrence"], errors="coerce")

    df["Report_Year"] = df["Date Reported"].dt.year.astype("Int64")
    df["Report_Month"] = df["Date Reported"].dt.month.astype("Int64")
    df["Report_DayOfWeek"] = df["Date Reported"].dt.weekday.astype("Int64")

    df["Occurrence_Hour"] = df["Time of Occurrence"].dt.hour.astype("Int64")

# Drop raw datetime columns
    df.drop(
        columns=["Date Reported", "Date of Occurrence", "Time of Occurrence"],
        inplace=True
)

    # ---------------------------
# Handle missing numeric values
# ---------------------------
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # ---------------------------
    # One-hot encode categoricals
    # ---------------------------
    categorical_cols = [
        "City",
        "Crime Description",
        "Crime Domain",
        "Weapon Used",
        "Victim Gender"
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df
