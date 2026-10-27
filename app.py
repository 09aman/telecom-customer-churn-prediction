
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# 1. Load saved files
# --------------------------------------------------

model = joblib.load("xgb_churn.pkl")
scaler = joblib.load("churn_scalar.pkl")
expected_columns = joblib.load("churn_expected_columns.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Mobile Service Churn Prediction")
st.write(
    "Enter customer details to predict whether "
    "the customer is likely to leave the service."
)

# --------------------------------------------------
# 2. Customer input form
# --------------------------------------------------

with st.form("churn_prediction_form"):

    st.subheader("Customer Information")

    senior_citizen = st.selectbox(
        "Senior Citizen",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=1,
        step=1
    )

    st.subheader("Service Information")

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    st.subheader("Billing Information")

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=90.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=90.0,
        step=1.0
    )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )

# --------------------------------------------------
# 3. Encode all 38 features
# --------------------------------------------------

if submitted:

    # Check that the expected feature list is correct
    if len(expected_columns) != 38:
        st.error(
            f"Expected 38 features, but found "
            f"{len(expected_columns)}. Check your saved "
            "churn_expected_columns.pkl file."
        )
        st.stop()

    # Create one row containing all expected features
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=expected_columns,
        dtype=float
    )

    # Numerical and binary features
    input_data["SeniorCitizen"] = senior_citizen
    input_data["Partner"] = 1 if partner == "Yes" else 0
    input_data["Dependents"] = 1 if dependents == "Yes" else 0
    input_data["tenure"] = tenure
    input_data["PaperlessBilling"] = (
        1 if paperless_billing == "Yes" else 0
    )
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges

    # Multiple Lines
    input_data["MultipleLines_No"] = int(
        multiple_lines == "No"
    )
    input_data["MultipleLines_No phone service"] = int(
        multiple_lines == "No phone service"
    )
    input_data["MultipleLines_Yes"] = int(
        multiple_lines == "Yes"
    )

    # Internet Service
    input_data["InternetService_DSL"] = int(
        internet_service == "DSL"
    )
    input_data["InternetService_Fiber optic"] = int(
        internet_service == "Fiber optic"
    )
    input_data["InternetService_No"] = int(
        internet_service == "No"
    )

    # Online Security
    input_data["OnlineSecurity_No"] = int(
        online_security == "No"
    )
    input_data["OnlineSecurity_No internet service"] = int(
        online_security == "No internet service"
    )
    input_data["OnlineSecurity_Yes"] = int(
        online_security == "Yes"
    )

    # Online Backup
    input_data["OnlineBackup_No"] = int(
        online_backup == "No"
    )
    input_data["OnlineBackup_No internet service"] = int(
        online_backup == "No internet service"
    )
    input_data["OnlineBackup_Yes"] = int(
        online_backup == "Yes"
    )

    # Device Protection
    input_data["DeviceProtection_No"] = int(
        device_protection == "No"
    )
    input_data["DeviceProtection_No internet service"] = int(
        device_protection == "No internet service"
    )
    input_data["DeviceProtection_Yes"] = int(
        device_protection == "Yes"
    )

    # Tech Support
    input_data["TechSupport_No"] = int(
        tech_support == "No"
    )
    input_data["TechSupport_No internet service"] = int(
        tech_support == "No internet service"
    )
    input_data["TechSupport_Yes"] = int(
        tech_support == "Yes"
    )

    # Streaming TV
    input_data["StreamingTV_No"] = int(
        streaming_tv == "No"
    )
    input_data["StreamingTV_No internet service"] = int(
        streaming_tv == "No internet service"
    )
    input_data["StreamingTV_Yes"] = int(
        streaming_tv == "Yes"
    )

    # Streaming Movies
    input_data["StreamingMovies_No"] = int(
        streaming_movies == "No"
    )
    input_data["StreamingMovies_No internet service"] = int(
        streaming_movies == "No internet service"
    )
    input_data["StreamingMovies_Yes"] = int(
        streaming_movies == "Yes"
    )

    # Payment Method
    input_data[
        "PaymentMethod_Bank transfer (automatic)"
    ] = int(
        payment_method == "Bank transfer (automatic)"
    )

    input_data[
        "PaymentMethod_Credit card (automatic)"
    ] = int(
        payment_method == "Credit card (automatic)"
    )

    input_data[
        "PaymentMethod_Electronic check"
    ] = int(
        payment_method == "Electronic check"
    )

    input_data[
        "PaymentMethod_Mailed check"
    ] = int(
        payment_method == "Mailed check"
    )

    # Contract
    input_data["Contract_Month-to-month"] = int(
        contract == "Month-to-month"
    )
    input_data["Contract_One year"] = int(
        contract == "One year"
    )
    input_data["Contract_Two year"] = int(
        contract == "Two year"
    )

    # --------------------------------------------------
    # 4. Match the exact training feature order
    # --------------------------------------------------

    input_data = input_data.reindex(
        columns=expected_columns,
        fill_value=0
    )

    if input_data.shape[1] != 38:
        st.error("The input must contain exactly 38 features.")
        st.stop()

    

   
    # --------------------------------------------------
    # 6. Predict churn
    # --------------------------------------------------

    prediction = int(model.predict(input_data)[0])

    probabilities = model.predict_proba(input_data)[0]
    classes = list(model.classes_)

    # Assumes class 1 means churn and class 0 means no churn.
    if 1 not in classes or 0 not in classes:
        st.error(
            f"Unexpected model classes: {classes}. "
            "Check your target encoding."
        )
        st.stop()

    churn_probability = float(
        probabilities[classes.index(1)]
    )

    stay_probability = float(
        probabilities[classes.index(0)]
    )

    # --------------------------------------------------
    # 7. Display results
    # --------------------------------------------------

    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.2%}"
        )

    with col2:
        st.metric(
            "Stay Probability",
            f"{stay_probability:.2%}"
        )

    if prediction == 1:
        st.error("⚠️ Prediction: Customer is likely to churn.")
    else:
        st.success("✅ Prediction: Customer is likely to stay.")

    st.progress(churn_probability)

 