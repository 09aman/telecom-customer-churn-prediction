from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡", layout="wide")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "churn_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"

FEATURES = ["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines",
"InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
"StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"]

OPTIONS = {
"gender":["Female","Male"], "Partner":["Yes","No"], "Dependents":["Yes","No"],
"PhoneService":["Yes","No"], "MultipleLines":["No phone service","No","Yes"],
"InternetService":["DSL","Fiber optic","No"],
"OnlineSecurity":["No internet service","No","Yes"], "OnlineBackup":["No internet service","No","Yes"],
"DeviceProtection":["No internet service","No","Yes"], "TechSupport":["No internet service","No","Yes"],
"StreamingTV":["No internet service","No","Yes"], "StreamingMovies":["No internet service","No","Yes"],
"Contract":["Month-to-month","One year","Two year"], "PaperlessBilling":["Yes","No"],
"PaymentMethod":["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
}

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return None

if not MODEL_PATH.exists():
    st.error("The saved model file churn_model.joblib is missing. Keep it in the same folder as app.py.")
    st.stop()

model = load_model()
metrics = load_metrics()

st.title("📡 Telecom Customer Churn Predictor")
st.caption("Pre-trained XGBoost model • Interactive Streamlit portfolio project")

tab_predict, tab_performance, tab_about = st.tabs(["🔮 Predict churn", "📊 Model performance", "ℹ️ About"])

with tab_predict:
    st.subheader("Predict churn for a customer")
    st.write("Enter the customer's account and service details. The saved model will return a churn estimate.")
    with st.form("customer_form"):
        customer_id = st.text_input("Customer ID (optional; display only)", "DEMO-001")
        col1, col2, col3 = st.columns(3)
        values = {}
        with col1:
            values["gender"] = st.selectbox("Gender", OPTIONS["gender"])
            values["SeniorCitizen"] = st.selectbox("Senior citizen", [0,1], format_func=lambda x: "Yes" if x else "No")
            values["Partner"] = st.selectbox("Partner", OPTIONS["Partner"])
            values["Dependents"] = st.selectbox("Dependents", OPTIONS["Dependents"])
            values["tenure"] = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            values["PhoneService"] = st.selectbox("Phone service", OPTIONS["PhoneService"])
            values["MultipleLines"] = st.selectbox("Multiple lines", OPTIONS["MultipleLines"])
        with col2:
            for field in ["InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]:
                values[field] = st.selectbox(field, OPTIONS[field])
        with col3:
            values["Contract"] = st.selectbox("Contract", OPTIONS["Contract"])
            values["PaperlessBilling"] = st.selectbox("Paperless billing", OPTIONS["PaperlessBilling"])
            values["PaymentMethod"] = st.selectbox("Payment method", OPTIONS["PaymentMethod"])
            values["MonthlyCharges"] = st.number_input("Monthly charges", min_value=0.0, value=70.0, step=1.0)
            values["TotalCharges"] = st.number_input("Total charges", min_value=0.0, value=840.0, step=10.0)
        submitted = st.form_submit_button("Predict churn", type="primary", use_container_width=True)

    if submitted:
        customer = pd.DataFrame([{name: values[name] for name in FEATURES}], columns=FEATURES)
        probability = float(model.predict_proba(customer)[0, 1])
        will_churn = probability >= 0.5
        st.divider()
        left, right = st.columns([1,2])
        with left:
            if will_churn:
                st.error("### Likely to churn")
            else:
                st.success("### Likely to stay")
            st.metric("Estimated churn probability", f"{probability:.1%}")
        with right:
            st.write(f"**Customer ID:** `{customer_id.strip() or 'Not provided'}`")
            st.progress(probability)
            st.caption("The 50% threshold is a default decision rule. Probability is an estimate, not a guarantee.")

with tab_performance:
    st.subheader("Held-out evaluation")
    if metrics:
        a,b,c = st.columns(3)
        a.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        b.metric("Weighted F1", f"{metrics['weighted_f1']:.1%}")
        c.metric("Macro F1", f"{metrics['macro_f1']:.1%}")
        st.caption(f"Evaluation used a stratified 80/20 split with random_state={metrics['random_state']}; these metrics are from the held-out test split, not the final all-data model.")
        st.markdown("#### Confusion matrix")
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"],
            index=["Actual: No churn","Actual: Churn"],
            columns=["Predicted: No churn","Predicted: Churn"]), use_container_width=True)
        st.markdown("#### Per-class report")
        report = pd.DataFrame(metrics["classification_report"]).T
        st.dataframe(report.round(3), use_container_width=True)
    else:
        st.info("Evaluation metrics were not packaged with this model.")

with tab_about:
    st.subheader("Project overview")
    st.write("A telecom customer churn classifier that estimates whether a customer may leave based on contract, tenure, billing, payment method, and subscribed services.")
    st.markdown("**Workflow**")
    st.markdown("1. Train and evaluate the model before deployment.\n2. Save the fitted preprocessing + XGBoost pipeline.\n3. Load the saved pipeline in Streamlit.\n4. Accept customer details and return a prediction.")
    st.markdown("**Model input features**")
    st.code(", ".join(FEATURES), language="text")
    st.caption("customerID is not used as a model feature. This app is a portfolio demonstration, not a sole basis for consequential customer decisions.")
