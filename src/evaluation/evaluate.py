import joblib
import pandas as pd
from sklearn.metrics import classification_report


def evaluate(X_test, y_test):
    model = joblib.load("models/logistic_model.pkl")
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))
