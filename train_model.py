# train_model.py
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import joblib
import os

# 1. CONFIGURE DATASET PATH (change if needed)
CSV_PATH = r"C:\Users\ajayt\Bussines Intelligence\churnanalysis\WA_Fn-UseC_-Telco-Customer-Churn.csv"
def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # Target and features
    X = df.drop(["customerID", "Churn"], axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    return X, y, df

def build_and_train_model(X, y):
    # Identify numeric and categorical columns
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit preprocessor and transform
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # Handle imbalance
    sm = SMOTE(random_state=42)
    X_train_s, y_train_s = sm.fit_resample(X_train_p, y_train)

    # Model
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight=None
    )

    rf.fit(X_train_s, y_train_s)

    # Evaluate
    y_pred = rf.predict(X_test_p)
    acc = accuracy_score(y_test, y_pred)
    print("Random Forest Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # Build final pipeline for inference (preprocess + trained model)
    model_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", rf),
        ]
    )

    return model_pipeline

def main():
    print("Loading data...")
    X, y, df = load_data()
    print(f"Data loaded. Shape: {df.shape}")

    print("Training model...")
    model_pipeline = build_and_train_model(X, y)

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "churn_model.pkl")
    joblib.dump(model_pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
