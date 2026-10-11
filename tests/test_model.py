import os
import joblib
import pandas as pd
import pytest

MODEL_PATH = "models/churn_pipeline.joblib" if os.path.exists("models/churn_pipeline.joblib") else "churn_pipeline.joblib"

@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    loaded_model = joblib.load(MODEL_PATH)
    return loaded_model

def test_model_loaded(model):
    """Verify that the model artifact is loaded and is not None."""
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

def test_single_prediction_output(model):
    """Verify inference pipeline input contract and probability shape."""
    sample_payload = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0
    }])
    
    prediction = model.predict(sample_payload)
    probabilities = model.predict_proba(sample_payload)
    
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]
    assert probabilities.shape == (1, 2)
    assert abs(probabilities[0].sum() - 1.0) < 1e-5