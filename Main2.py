import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection       import train_test_split
from sklearn.preprocessing         import StandardScaler
from sklearn.ensemble              import RandomForestClassifier
from sklearn.feature_selection     import SelectFromModel
from xgboost                       import XGBClassifier
from sklearn.metrics               import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

df = pd.read_excel("Telco_Customer_Churn.xlsx", engine="openpyxl")
dfss = df.copy()

dfss = dfss.drop(["CustomerID", "Zip Code", "Lat Long",  "Churn Reason", "Churn Label"], axis=1)

dfss['Total Charges'] = pd.to_numeric(dfss['Total Charges'], errors='coerce')
dfss['calc_charges'] = dfss['Monthly Charges'] * dfss['Tenure Months']
dfss['Total Charges'] = dfss['Total Charges'].fillna(dfss['calc_charges'])

dfss.drop(columns=['calc_charges'], inplace=True)

dfss.columns = (
    dfss.columns
      .str.strip()
      .str.lower()
      .str.replace(' ', '_')
)

for col in dfss.select_dtypes(include='object').columns:
    dfss[col] = dfss[col].astype('category')

print(dfss.info())
print(dfss.isnull().sum())
print(dfss.head(3))

X = pd.get_dummies(dfss.drop(columns=["churn_value"]), drop_first=True)
y = dfss["churn_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

rf_initial = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
rf_initial.fit(X_train_scaled, y_train)

sfm = SelectFromModel(rf_initial, threshold="mean", prefit=True)

selected_idx  = sfm.get_support(indices=True)
selected_feats = X.columns[selected_idx]
print(f"Original column amount: {X.shape[1]}")
print(f"Selected column amount: {len(selected_feats)}")
print("Selected columns:\n", list(selected_feats))

X_train_sel = sfm.transform(X_train_scaled)
X_test_sel  = sfm.transform(X_test_scaled)

# — Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
rf.fit(X_train_sel, y_train)

# — XGBoost
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train_sel, y_train)

y_pred_rf  = rf.predict(X_test_sel)
y_pred_xgb = xgb.predict(X_test_sel)

def evaluate(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred))
    print("MCC      :", matthews_corrcoef(y_true, y_pred))
    print("Conf Mat :\n", confusion_matrix(y_true, y_pred))

evaluate("Random Forest", y_test, y_pred_rf)
evaluate("XGBoost", y_test, y_pred_xgb)

plt.figure(figsize=(18, 12))             
corr = dfss.apply(lambda x: pd.factorize(x)[0]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",    
    annot_kws={"size": 9,}, 
    linewidths=0.5,
    cmap="coolwarm",
    vmin=-1, vmax=1,
    cbar_kws={"shrink": .5}
)
plt.xticks(rotation=45, ha="right")     # tilt labels so they don’t overlap
plt.yticks(rotation=0)
plt.title("Feature Correlation Matrix", pad=16, fontsize=14)
plt.tight_layout()
plt.show()
