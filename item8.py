"""8. Make a choice between two or more models for the same 
data using either a likelihood-based or a (possibly Bayesian) 
simulation-based approach. Which model is better?"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import catppuccin
import matplotlib as mpl
import statsmodels.formula.api as smf
from scipy import stats


mpl.style.use(catppuccin.PALETTE.latte.identifier)

df = pd.read_csv("diabetic_clean.csv")

#Nested models
    #M1: treatment variable alone
    #M2: + demographics
    #M3: + clinical severity
    #M4: + prior utilisation
models = {
    "M1: A1C only":
        "outcome ~ A1C_tested",
    "M2: + Demographics":
        "outcome ~ A1C_tested + age_num + gender_bin",
    "M3: + Clinical":
        "outcome ~ A1C_tested + age_num + gender_bin + time_in_hospital_capped + number_diagnoses_capped",
    "M4: Full model":
        "outcome ~ A1C_tested + age_num + gender_bin + time_in_hospital_capped + number_diagnoses_capped + number_inpatient_capped + num_medications_capped"
}

results = []
fitted_models = {}

for name, formula in models.items():
    m = smf.logit(formula=formula, data=df).fit(disp=0)
    fitted_models[name] = m
    results.append({
        "Model": name,
        "Num Params": len(m.params),
        "Log-Likelihood": round(m.llf, 2),
        "AIC": round(m.aic, 2),
        "BIC": round(m.bic, 2),
        "Pseudo R2": round(m.prsquared, 4),
        "A1C OR": round(np.exp(m.params["A1C_tested"]), 4),
        "A1C p-value": round(m.pvalues["A1C_tested"], 4)
    })

results_df = pd.DataFrame(results)
print("Model Comparison Table")
print(results_df.to_string(index=False))

best_aic = results_df.loc[results_df["AIC"].idxmin(), "Model"]
best_bic = results_df.loc[results_df["BIC"].idxmin(), "Model"]
print(f"\nBest model by AIC: {best_aic}")
print(f"Best model by BIC: {best_bic}")


#Visualisation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x = range(len(results_df))
labels = [f"M{i+1}" for i in x]

axes[0].plot(labels, results_df["AIC"], marker="o", label="AIC")
axes[0].plot(labels, results_df["BIC"], marker="o", label="BIC")
axes[0].set_title("AIC and BIC by Model")
axes[0].set_xlabel("Model")
axes[0].set_ylabel("Score (lower is better)")
axes[0].legend()

axes[1].set_ylim(0.85, 1.05)
axes[1].bar(labels, results_df["A1C OR"])
axes[1].axhline(y=1, color="red", linestyle="--", linewidth=1.2, label="OR = 1")
axes[1].set_title("A1C Odds Ratio Across Models")
axes[1].set_xlabel("Model")
axes[1].set_ylabel("Odds Ratio")
axes[1].legend()

plt.suptitle("Item 8: Likelihood-Based Model Selection")
plt.tight_layout()
plt.savefig("item8_model_selection.png", dpi=150)
plt.show()