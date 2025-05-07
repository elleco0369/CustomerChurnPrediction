import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection       import train_test_split
from sklearn.preprocessing         import StandardScaler
from sklearn.ensemble              import RandomForestClassifier
from sklearn.feature_selection     import SelectFromModel
from xgboost                       import XGBClassifier
from sklearn.feature_selection     import RFE
from sklearn.linear_model          import LogisticRegression
from sklearn.metrics               import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)


# Read data and make a working copy
df = pd.read_excel("Telco_Customer_Churn.xlsx", engine="openpyxl")
dfss = df.copy()

# Drop unneeded columns
drop_cols = ["CustomerID", "Zip Code", "Lat Long", "Churn Reason", "Churn Label"]
dfss.drop(columns=drop_cols, inplace=True)

# Fix Total Charges
dfss['Total Charges'] = pd.to_numeric(dfss['Total Charges'], errors='coerce')
dfss['calc_charges'] = dfss['Monthly Charges'] * dfss['Tenure Months']
dfss['Total Charges'] = dfss['Total Charges'].fillna(dfss['calc_charges'])
dfss.drop(columns=['calc_charges'], inplace=True)

# Normalize column names
dfss.columns = (
    dfss.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
)

# Convert object to category
for col in dfss.select_dtypes(include='object').columns:
    dfss[col] = dfss[col].astype('category')



# === 3. Encoding & Split ===
X = pd.get_dummies(dfss.drop(columns=["churn_value"]), drop_first=True)
y = dfss["churn_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# === 4. Scaling ===
# cıkartalım test ederek
def standart_scaller(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = standart_scaller(X_train, X_test)



# === 5. Feature Selection ===
def compute_threshold_from_rf(rf_model: RandomForestClassifier) -> float:
    imps = rf_model.feature_importances_
    return imps.mean() + imps.std()

# Initial RandomForest for importances
def feature_importances(X_train_scaled, X_test_scaled, y_train, X_cols):

    rf_initial = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    rf_initial.fit(X_train_scaled, y_train)
    threshold_val = compute_threshold_from_rf(rf_initial)

    sfm = SelectFromModel(rf_initial, threshold=threshold_val, prefit=True)

    selected_feats = X.columns[sfm.get_support()]
    print(f"Original columns: {X.shape[1]}")
    print(f"Selected columns: {len(selected_feats)}")
    print(sorted(list(selected_feats)))

    X_train_sel = sfm.transform(X_train_scaled)
    X_test_sel = sfm.transform(X_test_scaled)

    return X_train_sel , X_test_sel;

X_train_sel, X_test_sel = feature_importances(X_train_scaled, X_test_scaled, y_train, X.columns)


def RFE_selection(X_train, X_test, y_train, n_features=20):
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    X_train_sel = rfe.transform(X_train)
    X_test_sel = rfe.transform(X_test)
    return X_train_sel, X_test_sel


# === 6. Model Training ===

# Tuned RandomForest
def Random_Forest( X_train_sel , y_train):
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    rf.fit(X_train_sel, y_train)
    return rf

# — XGBoost
def XGB_Classifier(X_train_sel ,y_train ) :
    xgb = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(X_train_sel, y_train)
    return xgb


def RFE_model_run (X_train_sel ,y_train  ):
    rf_RFE = Random_Forest(X_train_sel, y_train)
    xgb_RFE = XGB_Classifier(X_train_sel, y_train)
    return rf_RFE , xgb_RFE

def feature_importances_model_run (X_train_sel ,y_train  ):
    rf_fi = Random_Forest(X_train_sel, y_train)
    xgb_fi = XGB_Classifier(X_train_sel, y_train)
    return rf_fi, xgb_fi

# === 7. Evaluation ===

def evaluate(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred))
    print("MCC      :", matthews_corrcoef(y_true, y_pred))
    print("Conf Mat :\n", confusion_matrix(y_true, y_pred))

# Feature Importance yöntemi ile seçim
X_train_fi, X_test_fi = feature_importances(X_train_scaled, X_test_scaled, y_train, X.columns)
rf_fi, xgb_fi = feature_importances_model_run(X_train_fi, y_train)

# RFE yöntemi ile seçim
X_train_rfe, X_test_rfe = RFE_selection(X_train_scaled, X_test_scaled, y_train)
rf_rfe, xgb_rfe = RFE_model_run(X_train_rfe, y_train)

# Tahminler ve değerlendirme
evaluate("RandomForest + FeatureImportance", y_test, rf_fi.predict(X_test_fi))
evaluate("XGBoost     + FeatureImportance", y_test, xgb_fi.predict(X_test_fi))

evaluate("RandomForest + RFE", y_test, rf_rfe.predict(X_test_rfe))
evaluate("XGBoost     + RFE", y_test, xgb_rfe.predict(X_test_rfe))


# Feature Correlation Matrix
plt.figure(figsize=(18, 12))
corr = dfss.apply(lambda x: pd.factorize(x)[0]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr,mask=mask,annot=True,fmt=".2f",annot_kws={"size": 9,},linewidths=0.5,cmap="coolwarm",vmin=-1, vmax=1,cbar_kws={"shrink": .5})
plt.xticks(rotation=45, ha="right")     # tilt labels so they don’t overlap
plt.yticks(rotation=0)
plt.title("Feature Correlation Matrix", pad=16, fontsize=14)
plt.tight_layout()
plt.show()

yes_total = (df['Churn Label']=="Yes").sum()
no_total  = (df['Churn Label']=="No").sum()
yes_f = ((df['Gender']=="Female") & (df['Churn Label']=="Yes")).sum()
yes_m = ((df['Gender']=="Male")   & (df['Churn Label']=="Yes")).sum()
no_f  = ((df['Gender']=="Female") & (df['Churn Label']=="No")).sum()
no_m  = ((df['Gender']=="Male")   & (df['Churn Label']=="No")).sum()

outer_vals   = [yes_total, no_total]
outer_labels = ["Churn: Yes", "Churn: No"]
outer_colors = ['#ff6666', '#66b3ff']

inner_vals   = [yes_f, yes_m, no_f, no_m]
inner_labels = ["Yes - F", "Yes - M", "No - F", "No - M"]
inner_colors = ['#c2c2f0','#ffb3e6','#c2c2f0','#ffb3e6']

fig, ax = plt.subplots(figsize=(8, 8))
size = 0.3
# Outer
ax.pie(outer_vals,radius=1,labels=outer_labels,colors=outer_colors,wedgeprops=dict(width=size, edgecolor='white'),autopct='%1.1f%%',pctdistance=0.8,labeldistance=1.15,textprops=dict(fontsize=12))
#Inner
ax.pie(inner_vals,radius=1-size,labels=inner_labels,colors=inner_colors,wedgeprops=dict(width=size, edgecolor='white'),autopct='%1.1f%%',pctdistance=0.35,labeldistance=0.65,textprops=dict(fontsize=10, fontweight='bold'))
ax.set(aspect="equal")
plt.title('Churn Distribution by Gender', y=1.05)
plt.tight_layout()
plt.show()


# Customer contract distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df,x="Churn Label",hue="Contract",palette="Set2")
plt.title("Customer contract distribution", fontsize=16)
plt.xlabel("Churn")
plt.ylabel("Count")
plt.legend(title="Contract", fontsize=12, title_fontsize=12)
plt.tight_layout()
plt.show()

#Dependents Distribution by Churn
color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
plt.figure(figsize=(10, 6))
sns.countplot(data=df,x="Churn Label",hue="Dependents",palette=color_map)
plt.title("Dependents Distribution by Churn")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.legend(title="Dependents")
plt.tight_layout()
plt.show()

# Monthly Charges Distribution by Churn
fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(df["Monthly Charges"][df["Churn Label"] == 'No'],fill=True,alpha=0.5,label="No churn",ax=ax)
sns.kdeplot(df["Monthly Charges"][df["Churn Label"] == 'Yes'],fill=True,alpha=0.5,label="Churned",ax=ax)
ax.set_title("Monthly Charges Distribution by Churn")
ax.set_xlabel("Monthly Charges")
ax.legend()
plt.tight_layout()
plt.show()