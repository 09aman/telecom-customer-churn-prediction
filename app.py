
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide"
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "churn_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"


# --------------------------------------------------
# 2. Model features
# --------------------------------------------------
FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


# --------------------------------------------------
# 3. Input options
# --------------------------------------------------
OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": [
        "No phone service", "No", "Yes"
    ],
    "InternetService": [
        "DSL", "Fiber optic", "No"
    ],
    "OnlineSecurity": [
        "No internet service", "No", "Yes"
    ],
    "OnlineBackup": [
        "No internet service", "No", "Yes"
    ],
    "DeviceProtection": [
        "No internet service", "No", "Yes"
    ],
    "TechSupport": [
        "No internet service", "No", "Yes"
    ],
    "StreamingTV": [
        "No internet service", "No", "Yes"
    ],
    "StreamingMovies": [
        "No internet service", "No", "Yes"
    ],
    "Contract": [
        "Month-to-month", "One year", "Two year"
    ],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}


# --------------------------------------------------
# 4. Load saved model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return None


# --------------------------------------------------
# 5. Check required files
# --------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        "The saved model file 'churn_model.joblib' is missing. "
        "Place it in the same folder as app.py."
    )
    st.stop()

try:
    saved_data = load_model()

    model = saved_data["model"]
    label_encoders = saved_data["label_encoders"]
    scaler = saved_data["scaler"]
    feature_columns = saved_data["feature_columns"]

except Exception as error:
    st.error(f"Unable to load the saved model: {error}")
    st.stop()

metrics = load_metrics()


# --------------------------------------------------
# 6. Application heading
# --------------------------------------------------
st.title("📡 Telecom Customer Churn Predictor")

st.caption(
    "Pre-trained XGBoost model • Interactive Streamlit portfolio project"
)


# --------------------------------------------------
# 7. Tabs
# --------------------------------------------------
tab_predict, tab_performance, tab_about = st.tabs(
    [
        "🔮 Predict churn",
        "📊 Model performance",
        "ℹ️ About"
    ]
)


# ==================================================
# TAB 1: PREDICT CUSTOMER CHURN
# ==================================================
with tab_predict:

    st.subheader("Predict churn for a customer")

    st.write(
        "Enter the customer's account and service details. "
        "The trained model will estimate the likelihood of churn."
    )

    with st.form("customer_form"):

        customer_id = st.text_input(
            "Customer ID (optional; display only)",
            "DEMO-001"
        )

        col1, col2, col3 = st.columns(3)

        values = {}

        # ------------------------------------------
        # Column 1
        # ------------------------------------------
        with col1:

            values["gender"] = st.selectbox(
                "Gender",
                OPTIONS["gender"]
            )

            values["SeniorCitizen"] = st.selectbox(
                "Senior citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            values["Partner"] = st.selectbox(
                "Partner",
                OPTIONS["Partner"]
            )

            values["Dependents"] = st.selectbox(
                "Dependents",
                OPTIONS["Dependents"]
            )

            values["tenure"] = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12
            )

            values["PhoneService"] = st.selectbox(
                "Phone service",
                OPTIONS["PhoneService"]
            )

            values["MultipleLines"] = st.selectbox(
                "Multiple lines",
                OPTIONS["MultipleLines"]
            )

        # ------------------------------------------
        # Column 2
        # ------------------------------------------
        with col2:

            for field in [
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies"
            ]:
                values[field] = st.selectbox(
                    field,
                    OPTIONS[field]
                )

        # ------------------------------------------
        # Column 3
        # ------------------------------------------
        with col3:

            values["Contract"] = st.selectbox(
                "Contract",
                OPTIONS["Contract"]
            )

            values["PaperlessBilling"] = st.selectbox(
                "Paperless billing",
                OPTIONS["PaperlessBilling"]
            )

            values["PaymentMethod"] = st.selectbox(
                "Payment method",
                OPTIONS["PaymentMethod"]
            )

            values["MonthlyCharges"] = st.number_input(
                "Monthly charges",
                min_value=0.0,
                value=70.0,
                step=1.0
            )

            values["TotalCharges"] = st.number_input(
                "Total charges",
                min_value=0.0,
                value=840.0,
                step=10.0
            )

        submitted = st.form_submit_button(
            "Predict churn",
            type="primary",
            use_container_width=True
        )

    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------
    if submitted:

        try:
            customer = pd.DataFrame(
                [{name: values[name] for name in FEATURES}],
                columns=FEATURES
            )

            # Encode categorical columns using the saved encoders.
            for column, encoder in label_encoders.items():

                if column not in customer.columns:
                    continue

                raw_value = str(customer.at[0, column])

                if raw_value not in encoder.classes_:
                    raise ValueError(
                        f"Unexpected value '{raw_value}' "
                        f"for feature '{column}'. "
                        "Check the training data and saved encoder."
                    )

                customer[column] = encoder.transform(
                    customer[column].astype(str)
                )

            # Use the exact feature order expected by the model.
            customer = customer[feature_columns]

            # Apply the saved scaler exactly once.
            customer_scaled = scaler.transform(customer)

            # Predict using the trained classifier.
            prediction = model.predict(customer_scaled)[0]

            # Get the probability for the churn class.
            if not hasattr(model, "predict_proba"):
                raise ValueError(
                    "The saved model does not support predict_proba()."
                )

            classes = list(model.classes_)

            if 1 not in classes:
                raise ValueError(
                    "The model does not contain class 1. "
                    "Check how the churn target was encoded during training."
                )

            churn_index = classes.index(1)

            probability = float(
                model.predict_proba(customer_scaled)[0, churn_index]
            )

            # Default decision threshold.
            will_churn = probability >= 0.5

            st.divider()

            left, right = st.columns([1, 2])

            with left:

                if will_churn:
                    st.error("### Likely to churn")
                else:
                    st.success("### Likely to stay")

                st.metric(
                    "Estimated churn probability",
                    f"{probability:.1%}"
                )

            with right:

                st.write(
                    f"**Customer ID:** "
                    f"`{customer_id.strip() or 'Not provided'}`"
                )

                st.progress(probability)

                st.caption(
                    "The 50% threshold is a default decision rule. "
                    "Probability is an estimate, not a guarantee."
                )

        except Exception as error:
            st.error(
                f"Unable to generate a prediction: {error}"
            )


# ==================================================
# TAB 2: MODEL PERFORMANCE
# ==================================================
with tab_performance:

    st.subheader("Held-out evaluation")

    if metrics:

        # Display the actual keys in the saved JSON.
        # This helps diagnose KeyError issues.
        st.caption(
            "Metrics loaded from model_metrics.json."
        )

        # Support common alternative metric key names.
        def find_metric(*possible_keys):
            for key in possible_keys:
                if key in metrics:
                    return metrics[key]
            return None

        accuracy = find_metric(
        "test_accuracy",
        "accuracy",
        "Accuracy",
        "accuracy_score",
        "Accuracy Score"
        )

        weighted_f1 = find_metric(
            "test_f1_weighted",
            "weighted_f1",
            "weighted F1",
            "weighted_f1_score",
            "f1_weighted"
        )

        macro_f1 = find_metric(
            "test_f1_macro",
            "macro_f1",
            "macro F1",
            "macro_f1_score",
            "f1_macro"
        )

        a, b, c = st.columns(3)

        if accuracy is not None:
            a.metric("Accuracy", f"{float(accuracy):.1%}")
        else:
            a.metric("Accuracy", "Not available")

        if weighted_f1 is not None:
            b.metric(
                "Weighted F1",
                f"{float(weighted_f1):.1%}"
            )
        else:
            b.metric("Weighted F1", "Not available")

        if macro_f1 is not None:
            c.metric(
                "Macro F1",
                f"{float(macro_f1):.1%}"
            )
        else:
            c.metric("Macro F1", "Not available")

        # Evaluation details
        random_state = find_metric(
            "random_state",
            "randomState"
        )

        if random_state is not None:
            st.caption(
                f"Saved evaluation random_state: {random_state}. "
                "Confirm the test split details in your training script."
            )

        # ------------------------------------------
        # Confusion matrix
        # ------------------------------------------
        confusion_matrix = find_metric(
            "confusion_matrix",
            "confusion matrix"
        )

        if confusion_matrix is not None:

            st.markdown("#### Confusion matrix")

            try:
                cm_df = pd.DataFrame(confusion_matrix)

                if cm_df.shape == (2, 2):
                    cm_df.index = [
                        "Actual: No churn",
                        "Actual: Churn"
                    ]

                    cm_df.columns = [
                        "Predicted: No churn",
                        "Predicted: Churn"
                    ]

                st.dataframe(
                    cm_df,
                    use_container_width=True
                )

            except (ValueError, TypeError) as error:
                st.warning(
                    f"Could not display the confusion matrix: {error}"
                )

        # ------------------------------------------
        # Classification report
        # ------------------------------------------
        classification_report = find_metric(
            "classification_report",
            "classification report",
            "class_report"
        )

        if classification_report is not None:

            st.markdown("#### Per-class report")

            try:
                report_df = pd.DataFrame(
                    classification_report
                ).T

                st.dataframe(
                    report_df.round(3),
                    use_container_width=True
                )

            except (ValueError, TypeError) as error:
                st.warning(
                    f"Could not display the classification report: {error}"
                )

        # Diagnostic information if metrics are missing.
        missing_metrics = []

        if accuracy is None:
            missing_metrics.append("accuracy")

        if weighted_f1 is None:
            missing_metrics.append("weighted_f1")

        if macro_f1 is None:
            missing_metrics.append("macro_f1")

        if missing_metrics:
            st.warning(
                "Some evaluation metrics are missing: "
                + ", ".join(missing_metrics)
                + ". Check model_metrics.json and your training script."
            )

      
    else:
        st.info(
            "Evaluation metrics were not packaged with this model. "
            "Run your model evaluation script to generate "
            "model_metrics.json."
        )


# ==================================================
# TAB 3: ABOUT
# ==================================================
with tab_about:

    st.subheader("Project overview")

    st.write(
        "A telecom customer churn classifier that estimates "
        "whether a customer may leave based on contract, tenure, "
        "billing, payment method, and subscribed services."
    )

    st.markdown("#### Workflow")

    st.markdown(
        """
        1. Train and evaluate the model before deployment.
        2. Save the fitted model and preprocessing objects.
        3. Load the saved model in Streamlit.
        4. Accept customer details and generate a prediction.
        5. Display churn probability and evaluation metrics.
        """
    )

    st.markdown("#### Model input features")

    st.code(", ".join(FEATURES), language="text")

    st.markdown("#### Technologies used")

    st.write(
        "Python • Pandas • Scikit-learn • XGBoost • Joblib • Streamlit"
    )

    st.caption(
        "customerID is not used as a model feature. "
        "This app is a portfolio demonstration, not a sole basis "
        "for consequential customer decisions."
    )
