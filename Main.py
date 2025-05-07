import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

plt.figure(figsize=(10, 6))
sns.countplot(data=df,x="Churn Label",hue="Contract",palette="Set2")
plt.title("Customer contract distribution", fontsize=16)
plt.xlabel("Churn")
plt.ylabel("Count")
plt.legend(title="Contract", fontsize=12, title_fontsize=12)
plt.tight_layout()
plt.show() 

color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
plt.figure(figsize=(10, 6))
sns.countplot(data=df,x="Churn Label",hue="Dependents",palette=color_map)
plt.title("Dependents Distribution by Churn")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.legend(title="Dependents")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(df["Monthly Charges"][df["Churn Label"] == 'No'],fill=True,alpha=0.5,label="No churn",ax=ax)
sns.kdeplot(df["Monthly Charges"][df["Churn Label"] == 'Yes'],fill=True,alpha=0.5,label="Churned",ax=ax)
ax.set_title("Monthly Charges Distribution by Churn")
ax.set_xlabel("Monthly Charges")
ax.legend()
plt.tight_layout()
plt.show()