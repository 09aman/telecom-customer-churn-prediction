# Telecom Customer Churn Prediction — Pre-trained Streamlit App

This version loads a **pre-trained XGBoost pipeline**. The training CSV is used before deployment and is **not required when the app is opened or used for predictions**.

## Project files

- `app.py` — Streamlit prediction interface
- `churn_model.joblib` — fitted preprocessing + XGBoost pipeline
- `model_metrics.json` — held-out evaluation metrics
- `requirements.txt` — Python dependencies
- `README.md` — these instructions

## Run locally (macOS / Linux)

1. Extract the ZIP and open Terminal in the `telco_churn_pretrained_app` folder.
2. Create and activate an environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

## Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Training and evaluation

The attached Telco dataset (7,043 records) was used to train and package `churn_model.joblib` before running the app. The saved artifact includes preprocessing and the XGBoost classifier, so the deployed app does not need to read or retrain on the CSV.

Evaluation metrics were calculated separately on a stratified 80/20 holdout split (random_state=42). After evaluation, the final prediction pipeline was fitted on the full dataset and saved for use in the app. The metrics shown in the app refer to the holdout evaluation model, not an in-sample score from the final model.

The pipeline handles numeric missing values and one-hot encodes categorical features. `customerID` is not used as an input feature.

## Suggested CV description

**Telecom Customer Churn Prediction | Python, Scikit-learn, XGBoost, Streamlit**
- Trained and packaged an XGBoost classification pipeline for telecom churn prediction.
- Built a Streamlit interface that loads the saved model and predicts customer-level churn probability without requiring the training dataset at runtime.
- Evaluated performance using accuracy, weighted F1, macro F1, and a confusion matrix.

Predictions are estimates, not guaranteed outcomes. This is a portfolio demonstration.
