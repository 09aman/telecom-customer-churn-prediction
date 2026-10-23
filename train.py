import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


# Load the dataset from the same folder as this script.
DATA_PATH = Path(__file__).resolve().parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

# Clean TotalCharges as in the notebook.
df["TotalCharges"] = df["TotalCharges"].str.strip().replace("", "0")
df["TotalCharges"] = df["TotalCharges"].astype("float")

# Remove customerID and encode Churn as in the notebook.
df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Label-encode the same categorical columns as in the notebook.
columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

label_encoders = {}
for column in columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder

# Drop the same columns as in the notebook.
df = df.drop(["PhoneService", "gender"], axis=1)

# Separate X and y, then apply StandardScaler as in the notebook.
x = df.drop("Churn", axis=1)
y = df["Churn"]

feature_columns = list(x.columns)
scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)
x = pd.DataFrame(x_scaled, columns=feature_columns, index=x.index)

# Use the same stratified split from the notebook's ensemble section.
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Use the same cross-validation setup and XGBoost search parameters.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_model = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)

xgb_params = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 7],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma": [0, 0.1, 0.3]
}

xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_params,
    n_iter=20,
    scoring="f1_weighted",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    refit=True
)

xgb_search.fit(x1_train, y1_train)

# Evaluate the tuned XGBoost model using the same test split.
best_model = xgb_search.best_estimator_
y1_pred = best_model.predict(x1_test)

accuracy = accuracy_score(y1_test, y1_pred)
f1_weighted = f1_score(y1_test, y1_pred, average="weighted")
f1_macro = f1_score(y1_test, y1_pred, average="macro")

print("Best XGBoost Parameters:", xgb_search.best_params_)
print("Best CV F1-score:", xgb_search.best_score_)
print("Test Accuracy:", accuracy)
print("Test F1 Score (Weighted):", f1_weighted)
print("Test F1 Score (Macro):", f1_macro)

# Save the model and the preprocessing objects used by this notebook.
# The preprocessing objects are included so the same transformations can
# be applied when making predictions on new customer data.
model_artifact = {
    "model": best_model,
    "label_encoders": label_encoders,
    "scaler": scalar,
    "feature_columns": feature_columns,
}

joblib.dump(model_artifact, Path(__file__).resolve().parent / "churn_model.joblib")

metrics = {
    "model": "XGBoost",
    "best_parameters": xgb_search.best_params_,
    "best_cv_f1_weighted": float(xgb_search.best_score_),
    "test_accuracy": float(accuracy),
    "test_f1_weighted": float(f1_weighted),
    "test_f1_macro": float(f1_macro),
}
with (Path(__file__).resolve().parent / "model_metrics.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=2)

print("Saved churn_model.joblib and model_metrics.json")
