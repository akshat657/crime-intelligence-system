import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.features.build_features import build_features


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = np.asarray(model.predict_proba(X_test))[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    return acc, auc


def train():
    # ---------------------------
    # Load data
    # ---------------------------
    df = pd.read_csv("data/raw/indian_crime_dataset.csv")

    # ---------------------------
    # Feature engineering
    # ---------------------------
    df = build_features(df)

    X = df.drop(columns=["Case Closed"])
    y = df["Case Closed"]

    # ---------------------------
    # Train-test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    os.makedirs("models", exist_ok=True)

    # ============================================================
    # 1️⃣ Logistic Regression (baseline)
    # ============================================================
    log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
    log_reg.fit(X_train, y_train)
    evaluate_model("Logistic Regression", log_reg, X_test, y_test)
    joblib.dump(log_reg, "models/logistic_model.pkl")

    # ============================================================
    # 2️⃣ Decision Tree
    # ============================================================
    tree = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=100,
        random_state=42
    )
    tree.fit(X_train, y_train)
    evaluate_model("Decision Tree", tree, X_test, y_test)
    joblib.dump(tree, "models/decision_tree_model.pkl")

    # ============================================================
    # 3️⃣ Random Forest (ensemble)
    # ============================================================
    forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1
    )

    forest.fit(X_train, y_train)
    evaluate_model("Random Forest", forest, X_test, y_test)
    joblib.dump(forest, "models/random_forest_model.pkl")

    # Save feature columns
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")


if __name__ == "__main__":
    train()
