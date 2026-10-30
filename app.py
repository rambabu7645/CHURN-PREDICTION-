"""Signal: customer churn prediction web application."""
import sqlite3
import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, instance_relative_config=True)
app.config["DATABASE"] = str(ROOT / "predictions.db")

bundle = joblib.load(ROOT / "models" / "churn_model.joblib")
PIPE, THR = bundle["pipeline"], float(bundle["threshold"])

SERVICES = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
REQUIRED_CATEGORIES = {
    "gender": {"Female", "Male"}, "Partner": {"Yes", "No"}, "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"}, "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {"Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"},
}


def connect_db():
    return sqlite3.connect(app.config["DATABASE"])


with connect_db() as conn:
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions
        (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT CURRENT_TIMESTAMP,
         tenure INT, contract TEXT, monthly REAL, probability REAL, risk TEXT)""")


def normalize_input(form):
    """Validate user input and return normalized features for the model."""
    missing = [key for key in REQUIRED_CATEGORIES if key not in form]
    if missing:
        raise ValueError("Please complete all customer profile fields.")
    row = {key: str(form[key]).strip() for key in REQUIRED_CATEGORIES}
    for key, allowed in REQUIRED_CATEGORIES.items():
        if row[key] not in allowed:
            raise ValueError(f"Choose a valid option for {key}.")
    try:
        row["SeniorCitizen"] = int(form.get("SeniorCitizen", 0))
        row["tenure"] = int(form["tenure"])
        row["MonthlyCharges"] = float(form["MonthlyCharges"])
        total_raw = str(form.get("TotalCharges", "")).strip()
        row["TotalCharges"] = float(total_raw) if total_raw else row["MonthlyCharges"] * row["tenure"]
    except (TypeError, ValueError, KeyError):
        raise ValueError("Enter valid numbers for tenure and charges.") from None
    if not all(math.isfinite(row[key]) for key in ("MonthlyCharges", "TotalCharges")):
        raise ValueError("Charges must be valid finite numbers.")
    if row["SeniorCitizen"] not in (0, 1):
        raise ValueError("Choose a valid option for senior citizen.")
    if not 0 <= row["tenure"] <= 72:
        raise ValueError("Tenure must be between 0 and 72 months.")
    if not 0 <= row["MonthlyCharges"] <= 200 or row["TotalCharges"] < 0:
        raise ValueError("Charges must be zero or greater; monthly charges must not exceed $200.")
    row["NumServices"] = sum(row[service] == "Yes" for service in SERVICES)
    row["AvgChargePerMonth"] = row["TotalCharges"] / max(row["tenure"], 1)
    t = row["tenure"]
    row["TenureGroup"] = "0-12" if t <= 12 else "13-24" if t <= 24 else "25-48" if t <= 48 else "49-72"
    return pd.DataFrame([row])


def predict(form):
    probability = float(PIPE.predict_proba(normalize_input(form))[:, 1][0])
    risk = "High" if probability >= max(THR, 0.6) else "Medium" if probability >= THR * 0.6 else "Low"
    return probability, risk


def dashboard_data():
    report_path = ROOT / "reports" / "final_model.csv"
    if report_path.exists():
        metrics = pd.read_csv(report_path).iloc[0]
        model_metrics = {"accuracy": f"{metrics['Accuracy'] * 100:.1f}",
                         "recall": f"{metrics['Recall'] * 100:.1f}",
                         "auc": f"{metrics['ROC_AUC'] * 100:.1f}"}
    else:
        model_metrics = {"accuracy": "—", "recall": "—", "auc": "—"}
    with connect_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    return model_metrics, total


@app.route("/")
def index():
    metrics, total = dashboard_data()
    return render_template("index.html", metrics=metrics, total_predictions=total,
                           threshold=round(THR * 100, 1), form=None)


@app.route("/predict", methods=["POST"])
def predict_form():
    form = request.form
    try:
        probability, risk = predict(form)
    except ValueError as exc:
        metrics, total = dashboard_data()
        return render_template("index.html", metrics=metrics, total_predictions=total,
                               threshold=round(THR * 100, 1), form=form, error=str(exc)), 400
    with connect_db() as conn:
        conn.execute("INSERT INTO predictions (tenure, contract, monthly, probability, risk) VALUES (?,?,?,?,?)",
                     (int(form["tenure"]), form["Contract"], float(form["MonthlyCharges"]), probability, risk))
    metrics, total = dashboard_data()
    result = {"prob": round(probability * 100, 1), "risk": risk, "will_churn": probability >= THR}
    return render_template("index.html", metrics=metrics, total_predictions=total,
                           threshold=round(THR * 100, 1), form=form, result=result)


@app.route("/api/predict", methods=["POST"])
def api():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send customer details as a JSON object."), 400
    try:
        probability, risk = predict(data)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(churn_probability=round(probability, 4), risk=risk, churn=probability >= THR)


@app.route("/history")
def history():
    with connect_db() as conn:
        rows = conn.execute("SELECT ts, tenure, contract, monthly, probability, risk FROM predictions ORDER BY id DESC LIMIT 20").fetchall()
        total, high, average = conn.execute("SELECT COUNT(*), SUM(CASE WHEN risk='High' THEN 1 ELSE 0 END), AVG(probability) FROM predictions").fetchone()
    stats = {"total": total, "high": high or 0, "avg": f"{(average or 0) * 100:.1f}"}
    return render_template("history.html", rows=rows, stats=stats)


if __name__ == "__main__":
    app.run(debug=True)
