"""Perform a causal inference analysis by constructing an appropriate graphical causal model for your dataset, 
deriving an adjustment set or instrumental variable, and performing a covariate adjustment or IV analysis."""

"""Generate DAG in https://dagitty.net/

Code for it:
dag {
A1C [exposure]
Age 
DiabetesMeds 
NumDiagnoses
NumInpatient 
Readmission [outcome]
A1C -> Readmission
Age -> A1C
Age -> Readmission
DiabetesMeds -> A1C
DiabetesMeds -> Readmission
NumDiagnoses -> A1C
NumDiagnoses -> Readmission
NumInpatient -> Readmission
}
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

#Covariate adjustment

df = pd.read_csv("diabetic_clean.csv")

# Unadjusted
m_unadj = smf.logit("outcome ~ A1C_tested", data=df).fit(disp=0)

# Adjusted using DAGitty adjustment set: { age, diabetesMed, number_diagnoses }
m_adj = smf.logit(
    "outcome ~ A1C_tested + age_num + diabetesMed_bin + number_diagnoses_capped",
    data=df
).fit(disp=0)

or_unadj = np.exp(m_unadj.params["A1C_tested"])
ci_unadj = np.exp(m_unadj.conf_int().loc["A1C_tested"])
p_unadj  = m_unadj.pvalues["A1C_tested"]

or_adj = np.exp(m_adj.params["A1C_tested"])
ci_adj = np.exp(m_adj.conf_int().loc["A1C_tested"])
p_adj  = m_adj.pvalues["A1C_tested"]

print("Unadjusted")
print(f"OR: {or_unadj:.4f} (95% CI: {ci_unadj[0]:.4f}-{ci_unadj[1]:.4f}, p={p_unadj:.4f})")
print("\nAdjusted (DAGitty: age, diabetesMed, number_diagnoses)")
print(f"OR: {or_adj:.4f} (95% CI: {ci_adj[0]:.4f}-{ci_adj[1]:.4f}, p={p_adj:.4f})")

