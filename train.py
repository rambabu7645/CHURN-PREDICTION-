"""Steps 2-7: preprocessing, imbalance handling, model comparison, tuning, SHAP, saving."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

RS = 42
df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
df = df.drop(columns=["customerID"])
y = (df.pop("Churn") == "Yes").astype(int)

# ---- Feature engineering
service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)
df["AvgChargePerMonth"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
df["TenureGroup"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices", "AvgChargePerMonth"]
cat_cols = [c for c in df.columns if c not in num_cols]
pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=RS)
cv = StratifiedKFold(5, shuffle=True, random_state=RS)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1),
    "SVM": SVC(probability=True, random_state=RS),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=RS, n_jobs=-1),
}

def metrics(name, mode, pipe):
    p = pipe.predict(X_test); pr = pipe.predict_proba(X_test)[:, 1]
    return dict(Model=name, Imbalance=mode, Accuracy=accuracy_score(y_test, p),
                Precision=precision_score(y_test, p), Recall=recall_score(y_test, p),
                F1=f1_score(y_test, p), ROC_AUC=roc_auc_score(y_test, pr))

# ---- Model comparison: no balancing vs SMOTE
rows = []
for name, m in models.items():
    for mode in ["Baseline", "SMOTE"]:
        steps = [("pre", pre)] + ([("smote", SMOTE(random_state=RS))] if mode == "SMOTE" else []) + [("clf", m)]
        pipe = ImbPipeline(steps).fit(X_train, y_train)
        rows.append(metrics(name, mode, pipe)); print(rows[-1])
res = pd.DataFrame(rows).round(4)
res.to_csv("reports/model_comparison.csv", index=False)

# ---- Tune top candidates (selected by ROC-AUC) with SMOTE inside CV (no leakage)
grids = {
    "Random Forest": {"clf__n_estimators": [200, 400], "clf__max_depth": [6, 10, None], "clf__min_samples_leaf": [1, 3, 5]},
    "XGBoost": {"clf__n_estimators": [150, 300], "clf__max_depth": [3, 4, 6], "clf__learning_rate": [0.03, 0.1], "clf__subsample": [0.8, 1.0]},
}
best_pipe, best_auc, best_name = None, 0, ""
for name, grid in grids.items():
    pipe = ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=RS)), ("clf", models[name])])
    rs = RandomizedSearchCV(pipe, grid, n_iter=8, scoring="roc_auc", cv=cv, random_state=RS, n_jobs=-1).fit(X_train, y_train)
    auc = roc_auc_score(y_test, rs.predict_proba(X_test)[:, 1])
    print(f"Tuned {name}: CV AUC={rs.best_score_:.4f}, test AUC={auc:.4f}, params={rs.best_params_}")
    if auc > best_auc: best_pipe, best_auc, best_name = rs.best_estimator_, auc, name

# ---- Threshold tuning: maximise F1 using CV predictions on training data (not test)
from sklearn.model_selection import cross_val_predict
cvp = cross_val_predict(best_pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
ths = np.arange(0.2, 0.7, 0.01)
thr = float(ths[np.argmax([f1_score(y_train, cvp >= t) for t in ths])])
proba = best_pipe.predict_proba(X_test)[:, 1]; pred = (proba >= thr).astype(int)
final = dict(Model=best_name, Threshold=round(thr, 2), Accuracy=accuracy_score(y_test, pred),
             Precision=precision_score(y_test, pred), Recall=recall_score(y_test, pred),
             F1=f1_score(y_test, pred), ROC_AUC=roc_auc_score(y_test, proba))
print("FINAL:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in final.items()})
pd.DataFrame([final]).round(4).to_csv("reports/final_model.csv", index=False)

# ---- Plots
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Stay", "Churn"]).plot(cmap="Blues")
plt.title(f"{best_name} (threshold={thr:.2f})"); plt.savefig("reports/confusion_matrix.png", dpi=120); plt.close()
plt.figure(figsize=(6, 5))
for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
    p = ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=RS)), ("clf", models[name])]).fit(X_train, y_train)
    f, t, _ = roc_curve(y_test, p.predict_proba(X_test)[:, 1]); plt.plot(f, t, label=name)
f, t, _ = roc_curve(y_test, proba); plt.plot(f, t, "k--", label=f"Tuned {best_name}")
plt.plot([0, 1], [0, 1], ":", c="grey"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC curves")
plt.savefig("reports/roc_curves.png", dpi=120); plt.close()

# ---- SHAP explainability (on the tuned tree model)
try:
    import shap
    Xt = best_pipe.named_steps["pre"].transform(X_test)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    names = best_pipe.named_steps["pre"].get_feature_names_out()
    sv = shap.TreeExplainer(best_pipe.named_steps["clf"]).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if getattr(sv, "ndim", 2) == 3 else sv)
    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=15)
    plt.tight_layout(); plt.savefig("reports/shap_summary.png", dpi=120); plt.close()
except Exception as e:
    print("SHAP skipped:", e)

joblib.dump({"pipeline": best_pipe, "threshold": thr, "model_name": best_name}, "models/churn_model.joblib")
print("Saved models/churn_model.joblib")
