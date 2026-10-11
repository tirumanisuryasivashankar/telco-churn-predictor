import gradio as gr
import pandas as pd
import joblib
import os
@spaces.GPU

# Load the trained model pipeline
model_path = "models/churn_pipeline.joblib"
# Fallback for Hugging Face Spaces if models folder structure changes
if not os.path.exists(model_path):
    model_path = "churn_pipeline.joblib" 

model = joblib.load(model_path)

def predict_churn(gender, senior_citizen, partner, dependents, tenure, phone_service, 
                  multiple_lines, internet_service, online_security, online_backup, 
                  device_protection, tech_support, streaming_tv, streaming_movies, 
                  contract, paperless_billing, payment_method, monthly_charges, total_charges):
    
    # Construct a DataFrame with the exact column names the pipeline expects
    input_data = pd.DataFrame([[gender, senior_citizen, partner, dependents, tenure, 
                                phone_service, multiple_lines, internet_service, online_security, 
                                online_backup, device_protection, tech_support, streaming_tv, 
                                streaming_movies, contract, paperless_billing, payment_method, 
                                monthly_charges, total_charges]], 
                              columns=["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", 
                                       "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
                                       "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
                                       "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", 
                                       "MonthlyCharges", "TotalCharges"])
    
    # Extract the probability of class 1 (Churn)
    churn_prob = model.predict_proba(input_data)[0][1]
    
    return {"Churn (High Risk)": float(churn_prob), "Retain (Low Risk)": float(1 - churn_prob)}

# Define Gradio input components
inputs = [
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Radio([0, 1], label="Senior Citizen"),
    gr.Radio(["Yes", "No"], label="Partner"),
    gr.Radio(["Yes", "No"], label="Dependents"),
    gr.Slider(0, 72, step=1, label="Tenure (Months)"),
    gr.Radio(["Yes", "No"], label="Phone Service"),
    gr.Dropdown(["No", "Yes", "No phone service"], label="Multiple Lines"),
    gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
    gr.Dropdown(["No", "Yes", "No internet service"], label="Online Security"),
    gr.Dropdown(["No", "Yes", "No internet service"], label="Online Backup"),
    gr.Dropdown(["No", "Yes", "No internet service"], label="Device Protection"),
    gr.Dropdown(["No", "Yes", "No internet service"], label="Tech Support"),
    gr.Dropdown(["No", "Yes", "No internet service"], label="Streaming TV"),
    gr.Dropdown(["No", "Yes", "No internet service"], label="Streaming Movies"),
    gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
    gr.Radio(["Yes", "No"], label="Paperless Billing"),
    gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method"),
    gr.Number(label="Monthly Charges"),
    gr.Number(label="Total Charges")
]

# Initialize and launch the app
app = gr.Interface(
    fn=predict_churn, 
    inputs=inputs, 
    outputs=gr.Label(num_top_classes=2, label="Prediction Probability"),
    title="Telco Customer Churn Predictor",
    description="Enter customer demographics and account details to predict churn risk."
)

if __name__ == "__main__":
    app.launch()