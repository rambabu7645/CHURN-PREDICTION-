"""Step 1: Exploratory Data Analysis. Saves plots to reports/."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df.shape); print(df.isna().sum()[df.isna().sum() > 0])
print(df["Churn"].value_counts(normalize=True))

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
sns.countplot(data=df, x="Churn", ax=ax[0, 0]); ax[0, 0].set_title("Churn distribution")
sns.countplot(data=df, x="Contract", hue="Churn", ax=ax[0, 1]); ax[0, 1].set_title("Churn by contract")
sns.histplot(data=df, x="tenure", hue="Churn", kde=True, ax=ax[1, 0]); ax[1, 0].set_title("Tenure vs churn")
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax[1, 1]); ax[1, 1].set_title("Monthly charges vs churn")
plt.tight_layout(); plt.savefig("reports/eda_overview.png", dpi=120)

plt.figure(figsize=(7, 5))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.tight_layout(); plt.savefig("reports/correlation_heatmap.png", dpi=120)
print("EDA plots saved in reports/")
