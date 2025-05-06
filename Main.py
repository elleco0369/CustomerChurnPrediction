import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)

## ilk hamle 
df = pd.read_excel("Telco_Customer_Churn.xlsx", engine="openpyxl")
dfss = df.copy()

print("Shape of the data:")
print(dfss.shape)

print("Data Information:")
print(dfss.info())

##Seem's like something's wrong in the "Total Charges" column.
# It only contains numeric values but the data type is object. Now we have to convert it into numeric type for future development.

## Data düzenleme 

dfss['Total Charges'] = pd.to_numeric(dfss['Total Charges'], errors='coerce')

print(dfss.isnull().sum())


# Total Charges’ı numeric’e çevir ve boşları hesapla
dfss['Total Charges'] = pd.to_numeric(dfss['Total Charges'], errors='coerce')
dfss['calc_charges'] = dfss['Monthly Charges'] * dfss['Tenure Months']
dfss['Total Charges'] = dfss['Total Charges'].fillna(dfss['calc_charges'])
dfss.drop(columns=['calc_charges'], inplace=True)

# Null check
print(dfss.isnull().sum())

# bu kısma bakıcalak
dfss = dfss.drop(["CustomerID", "Count","City", "Zip Code", "Country", "State", "Lat Long", "Churn Score", "CLTV", "Churn Reason",
                      "Contract", "Payment Method", "Churn Label", "Gender"], axis=1)


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

##Egitim 

X = dfss.drop(columns=['churn_value'], axis=1)
y = dfss['churn_value']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

rf  = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_scaled, y_train)

importances = rf.feature_importances_        
feat_names  = X.columns
feat_imp = pd.Series(importances, index=feat_names) \
    .sort_values(ascending=False)
top_k = 10
top_features = feat_imp.iloc[:top_k].index.tolist()
print(f"\nTop {top_k} features:\n", top_features)
X_train_sel = X_train[top_features]
X_test_sel  = X_test[top_features]
rf_sel = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
X_test_sel_scaled = scaler.fit_transform(X_test_sel)
X_train_sel_scaled = scaler.fit_transform(X_train_sel)
rf_sel.fit(X_train_sel_scaled, y_train)


xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test_scaled)
y_pred_rf_sel = rf_sel.predict(X_test_sel_scaled)
y_pred_xgb = xgb.predict(X_test)

def evaluate(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred))
    print("MCC      :", matthews_corrcoef(y_true, y_pred))
    print("Conf Mat :\n", confusion_matrix(y_true, y_pred))

evaluate("Random Forest",  y_test, y_pred_rf)
evaluate("XGBoost",        y_test, y_pred_xgb)
evaluate("Random Forest with sel",  y_test, y_pred_rf_sel)