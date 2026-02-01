import joblib
import pandas as pd
from src.features.build_features import build_features

model = joblib.load("models/logistic_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


def predict_case_closure(raw_input: dict):
    df = pd.DataFrame([raw_input])
    df = build_features(df)

    df = df.reindex(columns=feature_columns, fill_value=0)
    probability = model.predict_proba(df)[0][1]

    return probability


if __name__ == "__main__":
    sample_case = {
        "Date Reported": "2024-01-05",
        "Date of Occurrence": "2024-01-04",
        "City": "Delhi",
        "Crime Description": "BURGLARY",
        "Crime Domain": "Other Crime",
        "Weapon Used": "Unknown",
        "Victim Age": 35,
        "Victim Gender": "M",
        "Police Deployed": 12,
        "Case Closed": "No"
    }

    prob = predict_case_closure(sample_case)
    print(f"Predicted closure probability: {prob:.2f}")
