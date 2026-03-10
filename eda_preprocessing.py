import catppuccin
import matplotlib as mpl

mpl.style.use(catppuccin.PALETTE.latte.identifier)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load data
df = pd.read_csv("diabetic_data.csv")
df.replace('?', np.nan, inplace=True)
print(df.head())

numeric_cols = [
    "time_in_hospital", "num_medications", "num_lab_procedures",
    "num_procedures", "number_diagnoses", "number_inpatient"
]
numeric_labels = [
    "Time in Hospital", "Num Medications", "Num Lab Procedures",
    "Num Procedures", "Num Diagnoses", "Prior Inpatient Visits"
]

#Distribution of numeric variables
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, (col, label) in enumerate(zip(numeric_cols, numeric_labels)):
    axes[i].hist(df[col].dropna(), bins=30, edgecolor="black")
    axes[i].set_title(label)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")

plt.suptitle("Distributions of Numeric Variables", fontsize=13)
plt.tight_layout()
plt.savefig("distributions.png", dpi=150)
plt.show()

#Correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    fmt=".2f",
    xticklabels=numeric_labels,
    yticklabels=numeric_labels
)
plt.title("Correlation Matrix of Numeric Variables")
plt.tight_layout()
plt.savefig("correlation.png", dpi=150)
plt.show()

print("\nDescriptive Statistics")
print(df[numeric_cols].describe().round(2))

print("\nMissing Values")
print(df[numeric_cols].isnull().sum())

#Preprocessing

#Binary outcome: 0 = not readmitted, 1 = readmitted 
df["outcome"] = (df["readmitted"] != "NO").astype(int)

#Binary treatment: 0 = not tested, 1 = tested
df["A1C_tested"] = (df["A1Cresult"].isin(["Norm", ">7", ">8"])).astype(int) 

#Age: convert interval strings to average
age_map = {
    "[0-10)": 5,   "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95
}
df["age_num"] = df["age"].map(age_map)

#Gender: drop Unknown/Invalid, encode Male=1 Female=0
df = df[df["gender"].isin(["Male", "Female"])].copy()
df["gender_bin"] = (df["gender"] == "Male").astype(int)

#Diabetes medication: Yes=1, No=0
df["diabetesMed_bin"] = (df["diabetesMed"] == "Yes").astype(int)

#Remove duplicate patients - keep first encounter only
df = df.sort_values("encounter_id").drop_duplicates(
    subset="patient_nbr", keep="first"
).reset_index(drop=True)

#Cap numeric outliers at 99th percentile
for col in numeric_cols:
    cap = df[col].quantile(0.99)
    df[col + "_capped"] = df[col].clip(upper=cap)

#Drop rows with missing values in key variables
key_vars = [
    "outcome", "A1C_tested", "age_num", "gender_bin",
    "race", "diabetesMed_bin",
    "time_in_hospital_capped", "num_medications_capped",
    "num_lab_procedures_capped", "num_procedures_capped",
    "number_diagnoses_capped", "number_inpatient_capped"
]
df = df.dropna(subset=key_vars).reset_index(drop=True)

print(f"\nClean dataset shape: {df.shape}")
print(f"\nOutcome distribution:")
print(df["outcome"].value_counts())
print(f"\nA1C testing distribution:")
print(df["A1C_tested"].value_counts())

#Visualisation of preprocessed data
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

axes[0].bar(["Not Readmitted","Readmitted"], df["outcome"].value_counts())
axes[0].set_title("Outcome Balance")

axes[1].bar(["Not Tested","Tested"], df["A1C_tested"].value_counts())
axes[1].set_title("HbA1c Tested")

axes[2].hist(df["age_num"], bins=10, edgecolor="black")
axes[2].set_title("Age Distribution")

axes[3].hist(df["num_lab_procedures_capped"], bins=30, edgecolor="black")
axes[3].set_title("Num of Procedures (capped at 99)")


readmit_rate = df.groupby("A1C_tested")["outcome"].mean() * 100
axes[4].bar(["Not Tested","Tested"], readmit_rate)
axes[4].set_title("Readmission Rate by A1C Test")
axes[4].set_ylabel("Readmitted (%)")

age_rate = df.groupby("age_num")["outcome"].mean() * 100
axes[5].plot(age_rate.index, age_rate.values, marker="o")
axes[5].set_title("Readmission Rate by Age")
axes[5].set_xlabel("Age")

plt.suptitle("Preprocessing & Descriptive Overview")
plt.tight_layout()
plt.savefig("preprocessing_plots.png", dpi=150)
plt.show()

#New csv that other scripts can use
df.to_csv("diabetic_clean.csv", index=False)
