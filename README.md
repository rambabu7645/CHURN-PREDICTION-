# Signal — Customer Churn Intelligence

A complete customer churn prediction project: a trained machine learning pipeline, a responsive Flask website, an API, and a local history of customer risk analyses.

## Website

The Signal workspace includes:

- An overview dashboard with held-out model metrics and analysis count.
- A structured customer profile form with input validation and immediate risk feedback.
- Risk levels, probability scores, and a suggested next step for high-risk accounts.
- A prediction history with recent scores, account details, and risk summaries.
- A JSON prediction API for integrations.

The app uses the included model and dataset. Start it from this project directory:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. Prediction history is stored locally in `predictions.db`.

## API

Send the customer fields in `sample.json` to the prediction endpoint:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d @sample.json
```

The response includes `churn_probability`, `risk`, and `churn`. Invalid or incomplete input returns HTTP 400 with an `error` message.

## Project structure

```text
app.py                         Flask website and JSON API
templates/                     Shared layout, overview, and prediction history
static/                         Responsive design system and form interactions
data/Telco-Customer-Churn.csv   IBM Telco churn dataset (7,043 rows)
models/churn_model.joblib       Trained preprocessing and prediction pipeline
reports/                        Model results, EDA, and evaluation visuals
train.py                         Training and model comparison pipeline
eda.py                           Exploratory analysis and charts
```

## Model summary

The pipeline cleans `TotalCharges`, engineers service and tenure features, scales numeric fields, encodes categorical fields, and handles class imbalance with SMOTE during training. Random Forest and XGBoost candidates are tuned with stratified cross-validation. The decision threshold is selected from training predictions, then final metrics are recorded on held-out test data. See `reports/final_model.csv` and `reports/model_comparison.csv` for results.

Retrain from the project directory with `python train.py`. Regenerate exploratory charts with `python eda.py`.
