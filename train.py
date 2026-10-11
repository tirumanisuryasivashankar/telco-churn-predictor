import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Load Data
data_path = "data/telco_churn.csv"
if not os.path.exists(data_path):
    # Fallback to the default Kaggle filename if not renamed
    data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(data_path)

# 2. Data Cleaning
# Drop customerID as it is not predictive
df = df.drop("customerID", axis=1)

# TotalCharges contains empty strings ' ' that cause errors; convert to numeric and drop NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# Map target variable Churn (Yes/No) -> (1/0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify column types
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = [col for col in X.columns if col not in numeric_cols]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ]
)

# 5. MLflow Tracking Setup
mlflow.set_experiment("Telco_Customer_Churn_Prediction")

# Model Hyperparameters
n_estimators = 100
max_depth = 10
random_state = 42

with mlflow.start_run(run_name="RandomForest_Baseline"):
    # Build complete modeling pipeline
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # Train model
    model_pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    # Evaluate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Log Hyperparameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    # Log Metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

   # Log Model Pipeline Artifact with MLflow (explicitly trust tree types)
    mlflow.sklearn.log_model(
        sk_model=model_pipeline,
        artifact_path="churn_model_pipeline",
        registered_model_name="Telco_Churn_RandomForest",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    # Export model locally for Hugging Face deployment
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_pipeline, "models/churn_pipeline.joblib")

    print(f"--- Training Complete ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("Model saved locally to models/churn_pipeline.joblib")